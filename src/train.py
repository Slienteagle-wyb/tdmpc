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
from envs.env import make_env, make_mujoco_env
from envs.quad_envs import make_quadrotor_env_single, make_pybullet_drone_env
from algorithm.tdmpc import TDMPC
from algorithm.tdmpc_similarity import TDMPCSIM
from algorithm.helper import Episode, ReplayBuffer
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


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_mujoco_env(cfg), TDMPC(cfg), ReplayBuffer(cfg, latent_plan=True)
	# env, agent, buffer = make_quadrotor_env_single(cfg), TDMPCSIM(cfg), ReplayBuffer(cfg, latent_plan=True)
	# env, agent, buffer = make_mujoco_env(cfg), TDMPCSIM(cfg), ReplayBuffer(cfg, latent_plan=True)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
			common_metrics['episode_reward_pi'] = evaluate_pi(env, agent, cfg.eval_episodes, step, env_step, L.video)
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')


def test_dm_control(cfg):
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg, latent_plan=True)
	# load the model for test
	fp = os.path.join(work_dir, cfg.model_path)
	agent.load(fp)
	episode_rewards = []
	max_frame = cfg.episode_length
	for i in range(2):
		video = np.zeros((max_frame, 384, 384, 3), dtype=np.uint8)
		frame_count = 0
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			action = agent.plan(obs, eval_mode=True, step=0, t0=t == 0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			video[frame_count] = env.render(mode='rgb_array', height=384, width=384, camera_id=0)
			frame_count += 1
			ep_reward += reward
			t += 1
		episode_rewards.append(ep_reward)
		print(np.nanmean(episode_rewards))
		for j in range(max_frame):
			image = plt.imshow(video[j])
			plt.pause(0.01)
			plt.draw()


def test_gym_art(cfg):
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_quadrotor_env_single(cfg), TDMPCSIM(cfg), ReplayBuffer(cfg, latent_plan=True)
	# load the model for test
	fp = os.path.join(work_dir, cfg.model_path)
	agent.load(fp)
	episode_rewards = []
	num_rollouts = 2
	plot_thrusts = False
	plot_step = 2
	plot_obs = True

	start_time = time.time()
	for rollout_id in tqdm.tqdm(range(num_rollouts)):
		s = env.reset()
		done = False
		step_count, r_sum = 0, 0
		observations = []
		actions = []
		thrusts = []
		csv_data = []
		while not done:
			if cfg.env.render and (step_count % 2 == 0):
				env.render()
			action = agent.plan(s, eval_mode=True, step=0, t0=step_count == 0)
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
					for dim in range(0, 3, 1):
						plt.plot(observations_arr[:, dim])
					plt.legend([str(x) for x in range(observations_arr.shape[1])])

				plt.pause(0.05)  # have to pause otherwise does not draw
				plt.draw()

			step_count += 1
		episode_rewards.append(r_sum)
		print(np.nanmean(episode_rewards))

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
	train(parse_cfg(Path().cwd() / __CONFIG__))
	# test_dm_control(parse_cfg(Path().cwd() / __CONFIG__))
	# test_gym_art(parse_cfg(Path().cwd() / __CONFIG__))
