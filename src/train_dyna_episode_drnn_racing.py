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
from algorithm.tdsim_drnn_racing_extend_vis import TdMpcSimDssmRE
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
    complete_rate, mean_traverse_ticks = [], []
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
        complete_rate.append(info['complete_rate'])
        mean_traverse_ticks.append(info['mean_traverse_ticks'])
        if video:
            video.save(env_step)
    return {'episode_reward': np.nanmean(episode_rewards),
            'episode_length': int(np.nanmean(episode_length)),
            'complete_rate': np.nanmean(complete_rate),
            'mean_traverse_ticks': np.nanmean(mean_traverse_ticks)}


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
    cfg.noise_beta = config['noise_beta']
    cfg.horizon = config['horizon']
    cfg.safety_coef = config['safety_coef']
    cfg.wandb_exp_name = cfg.wandb_exp_name + '_' + str(cfg.noise_beta) + '_' + str(cfg.horizon)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_quadrotor_env_racing(cfg), TdMpcSimDssmR(cfg), RolloutBuffer(cfg)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    ctrl_step, iters = 0, 0
    while iters < cfg.train_steps:
        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        hidden = None
        external_reward_mean_list, current_std_mean_list = [], []
        complete_rate_list, mean_traverse_ticks_list = [], []
        while not episode.done:
            if episode.first or iters % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, plan_metrics = agent.plan(obs, hidden, step=ctrl_step, t0=episode.first)
            external_reward_mean_list.append(plan_metrics['external_reward_mean'])
            current_std_mean_list.append(plan_metrics['current_std'])
            obs, reward, done, info = env.step(action.cpu().numpy())
            complete_rate_list.append(info['complete_rate'])
            mean_traverse_ticks_list.append(info['mean_traverse_ticks'])
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
            'complete_rate': np.mean(complete_rate_list),
            'mean_traverse_ticks': np.mean(mean_traverse_ticks_list),
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if (episode_idx-1) % cfg.eval_freq_episodes == 0:
            common_metrics.update(evaluate(env, agent, cfg.eval_episodes, iters, env_step, L.video))
            # common_metrics['episode_reward_pi'] = evaluate_pi(env, agent, cfg.eval_episodes, cfg, env_step, L.video)
            L.log(common_metrics, category='eval')
            session.report({'episode_reward': common_metrics['episode_reward']})

        # save model every save epoch interval
        if episode_idx % int(cfg.save_interval) == 0 and episode_idx >= 600:
            L.save_model(agent, episode_idx)

    L.finish(agent)
    print('Training completed successfully')


def main(num_samples=1, gpus_per_trail=1.0):
    object_store_memory = 24 * 1024 * 1024 * 1024
    ray.init(object_store_memory=object_store_memory)
    search_space = {
        'horizon': tune.grid_search([5, ]),
        'noise_beta': tune.grid_search([0, ]),
        'safety_coef': tune.grid_search([5.0, ]),
        'seed': tune.randint(2500, 2600),
    }
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train),
                            resources={'cpu': 8, 'gpu': gpus_per_trail}),
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


if __name__ == '__main__':
    main()
    # train(parse_cfg(Path().cwd() / __CONFIG__))
    # test_gym_art(parse_cfg(Path().cwd() / __CONFIG__))
