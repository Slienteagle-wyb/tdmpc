import ray
from ray import tune, air
from ray.air import session
import warnings
import pathlib
import matplotlib.pyplot as plt
import tqdm
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import torch
import gym
import time

gym.logger.set_level(40)
import random
from pathlib import Path
from cfg import parse_cfg
from envs.quad_envs import make_quadrotor_env_racing
from algorithm.tdsim_drnn_racing import TdMpcSimDssmR
from algorithm.helper import Episode, RolloutBuffer
from gym_art.quadrotor_single.quad_utils import *
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
PARENT_PATH = pathlib.Path(__file__).parent.parent.absolute()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_length = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            if t == 0 or t % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan(obs, hidden, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        episode_length.append(t)
        if video:
            video.save(env_step)
    return {'episode_reward': np.nanmean(episode_rewards),
            'episode_length': int(np.nanmean(episode_length))}


def evaluate_pi(env, agent, num_episodes, cfg, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
            z = agent.model.h(obs)
            action = agent.model.pi(z)
            obs, reward, done, info = env.step(action.squeeze().detach().cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards)


def train(config):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(PARENT_PATH / __CONFIG__)
    set_seed(config['seed'])
    cfg.seed = config['seed']
    cfg.progress_coef = config['progress_coef']
    cfg.safety_coef = config['safety_coef']
    cfg.wandb_exp_name = cfg.wandb_exp_name + '_' + str(cfg.progress_coef) + '_' + str(cfg.safety_coef)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env = make_quadrotor_env_racing(cfg)
    agent, buffer = TdMpcSimDssmR(cfg), RolloutBuffer(cfg)
    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    ctrl_step, iters = 0, 0
    while iters < cfg.train_steps:
        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        hidden = None
        external_reward_mean_list = []
        current_std_mean_list = []
        while not episode.done:
            if episode.first or iters % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, plan_metrics = agent.plan(obs, hidden, step=ctrl_step, t0=episode.first)
            external_reward_mean_list.append(plan_metrics['external_reward_mean'])
            current_std_mean_list.append(plan_metrics['current_std'])
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            ctrl_step += 1
        episode_length = len(episode)
        buffer += episode

        # Update model
        train_metrics = {}
        if ctrl_step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if iters == 0 else episode_length
            for i in range(num_updates):
                iters += 1
                train_metrics.update(agent.update(buffer, iters))

        # Log training episode
        episode_idx += 1
        env_step = int(ctrl_step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': ctrl_step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'episode_length': len(episode),
            'external_reward_mean': np.mean(external_reward_mean_list),
            'current_std': np.mean(current_std_mean_list),
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if (episode_idx-1) % cfg.eval_freq_episodes == 0:
            common_metrics.update(evaluate(env, agent, cfg.eval_episodes, iters, env_step, L.video))
            common_metrics['episode_reward_pi'] = evaluate_pi(env, agent, cfg.eval_episodes, cfg, env_step, L.video)
            L.log(common_metrics, category='eval')
            session.report({'episode_reward': common_metrics['episode_reward']})

        # save model every save epoch interval
        if episode_idx % int(cfg.save_interval) == 0 and episode_idx >= 600:
            L.save_model(agent, episode_idx)

    L.finish(agent)
    print('Training completed successfully')


def main(num_samples=1, gpus_per_trail=0.25):
    object_store_memory = 24 * 1024 * 1024 * 1024
    ray.init(object_store_memory=object_store_memory)
    search_space = {
        'progress_coef': tune.grid_search([200.0, ]),
        'safety_coef': tune.grid_search([2.5, 5.0, 10.0]),
        'seed': tune.randint(10, 2000),
    }
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train),
                            resources={'cpu': 4, 'gpu': gpus_per_trail}),
        tune_config=tune.TuneConfig(
            metric='episode_reward',
            mode='max',
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(stop={'training_iteration': 500, 'episode_reward': 3000}),
        param_space=search_space,
    )
    try:
        results = tuner.fit()
        best_result = results.get_best_result('episode_reward', 'max')
        print('Best result: {}'.format(best_result))
    except KeyboardInterrupt:
        print('Keyboard interrupt received, exiting.')
    finally:
        ray.shutdown()


def test_gym_art(cfg):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_quadrotor_env_racing(cfg), TdMpcSimDssmR(cfg), RolloutBuffer(cfg)
    # load the model for test
    fp = os.path.join(work_dir, cfg.model_path)
    agent.load(fp)
    episode_rewards = []
    num_rollouts = 10
    plot_thrusts = False
    plot_step = None
    plot_obs = False

    start_time = time.time()
    for rollout_id in tqdm.tqdm(range(num_rollouts)):
        s, done, ep_reward, t = env.reset(), False, 0, 0
        done = False
        step_count, r_sum = 0, 0
        observations = []
        actions = []
        thrusts = []
        csv_data = []
        while not done:
            if cfg.env.render and (step_count % 2 == 0):
                env.render()
            if t == 0 or t % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan(s, hidden, eval_mode=True, step=0, t0=step_count == 0)
            s, reward, done, info = env.step(action.cpu().numpy())
            r_sum += reward
            actions.append((action.cpu().numpy() + np.ones(4)) * 0.5)
            thrusts.append(env.env.dynamics.thrust_cmds_damp)
            observations.append(s)
            # record the relative pos to target and attitude represented by quaternion
            quat = R2quat(rot=s[6:15])
            csv_data.append(np.concatenate([np.array([1.0 / env.control_freq * step_count]), s[0:3], quat]))

            if plot_step is not None and step_count % plot_step == 0:
                plt.clf()
                if plot_obs:
                    observations_arr = np.array(observations)
                    # print('observations array shape', observations_arr.shape)
                    dimenstions = observations_arr.shape[1]
                    for dim in range(15, 18, 1):
                        plt.plot(observations_arr[:, dim])
                    plt.legend([str(x) for x in range(observations_arr.shape[1])])

                plt.pause(0.05)  # have to pause otherwise does not draw
                plt.draw()

            step_count += 1
        print('the episode reward is: ', r_sum)
        episode_rewards.append(r_sum)
        # print(np.nanmean(episode_rewards))

        if plot_thrusts:
            plt.figure(3, figsize=(10, 10))
            ep_time = np.linspace(0, env.control_freq, cfg.episode_length)
            actions = np.array(actions)
            thrusts = np.array(thrusts)
            for i in range(2):
                plt.plot(ep_time, actions[:, i], label="Thrust desired %d" % i)
                plt.plot(ep_time, thrusts[:, i], label="Thrust produced %d" % i)
            plt.legend()
            plt.show(block=False)
            input("Press Enter to continue...")
        print("##############################################################")
        print("Total time: ", time.time() - start_time)


if __name__ == '__main__':
    main()
    # train(parse_cfg(Path().cwd() / __CONFIG__))
    # test_gym_art(parse_cfg(Path().cwd() / __CONFIG__))