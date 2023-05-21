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
from envs.mujoco_envs import make_robohive_env
from algorithm.tdmpc_icem_similarity_drnn import TdICemSimDssm
from algorithm.helper import Episode, RolloutBuffer
import logger
from robohive.utils import tensor_utils

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards, paths = [], []
    for i in range(num_episodes):
        env_infos = []
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, _ = agent.plan_icem(obs, hidden, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, env_info = env.step(action.cpu().numpy())
            env_infos.extend(env_info)
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video:
            video.save(env_step)
        path = dict(env_infos=tensor_utils.stack_tensor_dict_list(env_infos))
        paths.append(path)
    success_percentage = env.env.evaluate_success(paths)
    return {'episode_reward': np.nanmean(episode_rewards),
            'success_rate': success_percentage}


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
    env, agent, buffer = make_robohive_env(cfg), TdICemSimDssm(cfg), RolloutBuffer(cfg)

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
            hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden, plan_metrics = agent.plan_icem(obs, hidden, step=ctrl_step, t0=episode.first)
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
            'episode_length': len(episode)}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if (episode_idx - 1) % cfg.eval_freq_episodes == 0:
            common_metrics.update(evaluate(env, agent, cfg.eval_episodes, iters, env_step, L.video))
            # common_metrics['episode_reward_pi'] = evaluate_pi(env, agent, cfg.eval_episodes, iters, env_step, L.video)
            L.log(common_metrics, category='eval')

        # save model every save epoch interval
        if episode_idx % int(cfg.save_interval) == 0 and episode_idx >= 500:
            L.save_model(agent, episode_idx)

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
