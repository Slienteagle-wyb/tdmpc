import ray
from ray import tune, air
from ray.air import session
import warnings
import pathlib
warnings.filterwarnings('ignore')
import os
from cfg import parse_cfg
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from envs.env import make_env
from algorithm.tdmpc_icem_similarity_drnn import TdICemSimDssm
from algorithm.helper import Episode, ReplayBuffer
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
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        hidden = None
        while not done:
            if t == 0 or t % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan(obs, hidden, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)


def evaluate_pi(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
            action = agent.model.pi(agent.model.h(obs))
            obs, reward, done, _ = env.step(action.squeeze().detach().cpu().numpy())
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
    cfg.horizon = config['horizon']
    set_seed(config['seed'])
    work_dir = PARENT_PATH / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_env(cfg), TdICemSimDssm(cfg), ReplayBuffer(cfg, latent_plan=True)

    # Run training
    cfg.wandb_exp_name = cfg.wandb_exp_name + '_' + str(cfg.noise_beta)
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        hidden = None
        total_train_step = step
        external_reward_mean_list = []
        current_std_mean_list = []
        while not episode.done:
            # reset the hidden state for gru every cfg.horizon step.
            if episode.first or total_train_step % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, plan_metrics = agent.plan(obs, hidden, step=step, t0=episode.first)
            external_reward_mean_list.append(plan_metrics['external_reward_mean'])
            current_std_mean_list.append(plan_metrics['current_std'])
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            total_train_step += 1
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'external_reward_mean': np.mean(external_reward_mean_list),
            'current_std': np.mean(current_std_mean_list), }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            common_metrics['episode_reward_pi'] = evaluate_pi(env, agent, cfg.eval_episodes, step, env_step, L.video)
            L.log(common_metrics, category='eval')
            session.report({'episode_reward': common_metrics['episode_reward']})

    L.finish(agent)
    print('Training completed successfully')


def main(num_samples=2, gpus_per_trail=0.8):
    object_store_memory = 4 * 1024 * 1024 * 1024
    ray.init(object_store_memory=object_store_memory)
    search_space = {
        'horizon': tune.grid_search([8, ]),
        'seed': tune.randint(200, 2000),
    }
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train),
                            resources={'cpu': 6, 'gpu': gpus_per_trail}),
        tune_config=tune.TuneConfig(
            metric='episode_reward',
            mode='max',
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(stop={'training_iteration': 100, 'episode_reward': 900}),
        param_space=search_space,
    )
    try:
        results = tuner.fit()
        best_result = results.get_best_result('episode_reward', 'max')
        print('Best result: {}'.format(best_result))
    except KeyboardInterrupt:
        print('Keyboard interrupt received, exiting.')
        tuner.stop()
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
