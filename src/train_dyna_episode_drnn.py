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
from tdmpc.envs.quad_envs import make_quadrotor_env_single, make_pybullet_drone_env
from algorithm.tdmpc_similarity_drnn import TdMpcSimDssm
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
            if t == 0 or t % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden = agent.plan(obs, hidden, eval_mode=True, step=step, t0=t == 0)
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
    env, agent, buffer = make_quadrotor_env_single(cfg), TdMpcSimDssm(cfg), RolloutBuffer(cfg)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    ctrl_step, iters = 0, 0
    while iters < cfg.train_steps:
        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        hidden = None
        while not episode.done:
            if episode.first or iters % 1 == 0:
                hidden = agent.model.init_hidden_state(batch_size=1, device='cuda')
            action, hidden = agent.plan(obs, hidden, step=ctrl_step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
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


def test_gym_art(cfg):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_quadrotor_env_single(cfg), TdMpcSimDssm(cfg), RolloutBuffer(cfg)
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
            action, hidden = agent.plan(s, hidden, eval_mode=True, step=0, t0=step_count == 0)
            s, reward, done, info = env.step(action.cpu().numpy())
            r_sum += reward
            actions.append((action.cpu().numpy() + np.ones(4)) * 0.5)
            thrusts.append(env.dynamics.thrust_cmds_damp)
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
        print(r_sum)
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
    # train(parse_cfg(Path().cwd() / __CONFIG__))
    test_gym_art(parse_cfg(Path().cwd() / __CONFIG__))
