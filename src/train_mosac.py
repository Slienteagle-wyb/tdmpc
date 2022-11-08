import warnings

warnings.filterwarnings('ignore')
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.mosac import MoSac
from algorithm.helper import Episode, ReplayBuffer
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
FORWARD_SEARCH_HORIZON = 15


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
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0)
            action = agent.model.pi(agent.model.h(obs))
            obs, reward, done, _ = env.step(action.detach().cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, env_buffer = make_env(cfg), MoSac(cfg), ReplayBuffer(cfg)
    model_buffer = ReplayBuffer(cfg, latent_plan=True)
    model_env = make_env(cfg)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # Collect trajectory from the environment using optimized policy (actor-critic)
        obs = env.reset()
        env_episode = Episode(cfg, obs)
        model_obs = model_env.reset()
        model_episode = Episode(cfg, model_obs)

        while not env_episode.done:
            obs = torch.tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            model_obs = torch.tensor(model_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            with torch.no_grad():
                z = agent.model.h(obs)
                pi_action = agent.model.pi(z)
                model_z = agent.model.h(model_obs)
                plan_action = agent.latent_plan(model_z, step=step, t0=model_episode.first)
            obs, reward, done, _ = env.step(pi_action.detach().cpu().numpy())
            env_episode += (obs, pi_action, reward, done)
            model_obs, model_reward, model_done, _ = model_env.step(plan_action.detach().cpu().numpy())
            model_episode += (model_obs, plan_action, model_reward, model_done)
        assert len(env_episode) == cfg.episode_length
        env_buffer += env_episode
        model_buffer += model_episode
        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            num_updates = int(num_updates / cfg.update_freq)
            for i in range(num_updates):
                train_metrics.update(agent.update_interval(env_buffer, model_buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': env_episode.cumulative_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            L.log(common_metrics, category='eval')

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
