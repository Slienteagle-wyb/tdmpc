import warnings
import matplotlib.pyplot as plt
import tqdm
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from tdmpc.envs.env import make_env
from tdmpc.envs.quad_envs import make_quadrotor_env_single, make_quadrotor_env_multi
from algorithm.tdmpc import TDMPC
from algorithm.tdmpc_similarity import TDMPCSIM
from algorithm.helper import Episode, ReplayBuffer, RolloutBuffer
from gym_art.quadrotor_single.quad_utils import *
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_rewards_mean = []
    episode_length = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        episode_rewards_mean.append(ep_reward/t)
        episode_length.append(t)
        if video:
            video.save(env_step)
    return {'episode_reward': np.nanmean(episode_rewards),
            'episode_reward_mean': np.nanmean(episode_rewards_mean),
            'episode_length': int(np.nanmean(episode_length))}


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


def train(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    model_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.pretrained_seed)
    # env, agent, buffer = make_quadrotor_env_single(cfg), TDMPCSIM(cfg), RolloutBuffer(cfg)
    env, agent, buffer = make_quadrotor_env_multi(cfg), TDMPCSIM(cfg), RolloutBuffer(cfg)
    demo_buffer = RolloutBuffer(cfg)
    # load the pretrained model for fine_tuning
    fp = os.path.join(model_dir, cfg.model_path)
    agent.load(fp)
    print('Have loaded the pretrained model successfully!!')
    if cfg.freeze_encoder:
        print('Have frozen the pretrained encoder head')
        agent.model.freeze_encoder(enable=False)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    ctrl_step, iters = 0, 0
    while iters < cfg.train_steps:
        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=ctrl_step, t0=episode.first, fine_tuning=True)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            ctrl_step += 1
        episode_length = len(episode)
        if ctrl_step >= cfg.seed_steps:
            buffer += episode
        else:
            demo_buffer += episode

        # Update model
        train_metrics = {}
        if ctrl_step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if iters == 0 else episode_length
            for i in range(num_updates):
                iters += 1
                train_metrics.update(agent.finetune(buffer, iters, demo_buffer))

        # Log training episode
        episode_idx += 1
        env_step = int(ctrl_step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': ctrl_step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'episode_reward_mean': episode.episode_reward_mean,
            'episode_length': len(episode)}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if (episode_idx-1) % (cfg.eval_freq / 250) == 0:
            common_metrics.update(evaluate(env, agent, cfg.eval_episodes, iters, env_step, L.video))
            common_metrics['episode_reward_pi'] = evaluate_pi(env, agent, cfg.eval_episodes, iters, env_step, L.video)
            L.log(common_metrics, category='eval')

        # save model every save epoch interval
        if episode_idx % int(cfg.save_interval) == 0:
            L.save_model(agent, episode_idx)

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
