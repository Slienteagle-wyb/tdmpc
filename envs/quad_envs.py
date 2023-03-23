import gym
import numpy as np
from gym.wrappers import RecordEpisodeStatistics
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING_SINGLE, DEFAULT_QUAD_REWARD_SHAPING
from gym_art.quadrotor_single.quadrotor import QuadrotorEnv
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
from gym_art.quadrotor_multi.quadrotor_racing import QuadrotorEnvRacing


def make_quadrotor_env_single(cfg):
    sampler_1 = None
    dyn_randomization_ratio = cfg.env.dynamics_randomization_ratio
    if cfg.env.dynamics_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    reward_coeff = DEFAULT_QUAD_REWARD_SHAPING_SINGLE['quad_rewards']
    reward_coeff.update(dict(reward_scale=1.0, pos=1.0, crash=0.0, action_change=0.0, orient=1.0))
    # dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))
    dynamics_change = None

    env = QuadrotorEnv(dynamics_params=cfg.env.quad_type, raw_control=cfg.env.raw_control,
                       raw_control_zero_middle=cfg.env.raw_control_zero_middle,
                       dynamics_randomize_every=cfg.env.dynamics_randomize_every, dynamics_change=dynamics_change,
                       dyn_sampler_1=sampler_1, sense_noise=cfg.env.sense_noise, obs_repr=cfg.env.obs_repr,
                       init_random_state=cfg.env.init_random_state, ep_time=cfg.env.episode_duration,
                       rew_coeff=reward_coeff, excite=cfg.env.excite, resample_goal=cfg.env.resample_goal,
                       controller_type=cfg.env.controller_type)
    env = QuadObsWrapper(env)
    env = ActRepeatWrapper(env, cfg.action_repeat)

    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env


def make_quadrotor_env_multi(cfgs):
    sampler_1 = None
    cfg = cfgs.env
    dyn_randomization_ratio = cfg.dynamics_randomization_ratio
    room_dims = (10, 10, 10)

    if cfg.dynamics_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    # dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))
    dynamics_change = None

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    rew_coeff.update(dict(pos=1.0, crash=5.0, action_change=0.0, orient=1.0, precede=1.0))

    extended_obs = cfg.neighbor_obs_type
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents,
        dynamics_params=cfg.quad_type, raw_control=cfg.raw_control, raw_control_zero_middle=cfg.raw_control_zero_middle,
        dynamics_randomize_every=cfg.dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=cfg.sense_noise, init_random_state=cfg.init_random_state, ep_time=cfg.episode_duration, room_length=room_dims[0],
        room_width=room_dims[1], room_height=room_dims[2], rew_coeff=rew_coeff,
        quads_mode=cfg.quads_mode, quads_formation=cfg.quads_formation, quads_formation_size=cfg.quads_formation_size,
        swarm_obs=extended_obs, quads_use_numba=cfg.quads_use_numba, quads_settle=cfg.quads_settle, quads_settle_range_meters=cfg.quads_settle_range_meters,
        quads_vel_reward_out_range=cfg.quads_vel_reward_out_range, quads_obstacle_mode=cfg.quads_obstacle_mode,
        quads_view_mode=cfg.quads_view_mode, quads_obstacle_num=cfg.quads_obstacle_num, quads_obstacle_type=cfg.quads_obstacle_type, quads_obstacle_size=cfg.quads_obstacle_size,
        adaptive_env=cfg.quads_adaptive_env, obstacle_traj=cfg.quads_obstacle_traj, local_obs=cfg.quads_local_obs, obs_repr=cfg.quads_obs_repr,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius, collision_falloff_radius=cfg.quads_collision_falloff_radius,
        local_metric=cfg.quads_local_metric, controller_type=cfg.controller_type,
        local_coeff=cfg.quads_local_coeff,  # how much velocity matters in "distance" calculation
        use_replay_buffer=use_replay_buffer, obstacle_obs_mode=cfg.quads_obstacle_obs_mode,
        obst_penalty_fall_off=cfg.quads_obst_penalty_fall_off,
    )

    env = CompatibilityWrapper(env)
    env = ActRepeatWrapper(env, cfgs.action_repeat)

    cfgs.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfgs.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfgs.action_dim = env.action_space.shape[0]

    return env


def make_pybullet_drone_env(cfg):
    domain, task = cfg.task.replace('-', '_').split('_', 1)
    env_id = task + str('-aviary-v0')
    env = gym.make(env_id)
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.SIM_FREQ * env.EPISODE_LEN_SEC
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    return env


class QuadObsWrapper(gym.Wrapper):
    def __init__(self, env: QuadrotorEnv):
        super().__init__(env)
        self._env = env

    def _modify_obs(self, obs):
        clipped_relative_pos = np.clip(obs[0:3], -2.0 * self._env.box, 2.0 * self._env.box)  # box hw is 2.0
        normalized_relative_pos = clipped_relative_pos / (2.0 * self._env.box)
        obs[0:3] = normalized_relative_pos
        vxyz = np.clip(obs[3:6], -self._env.dynamics.vxyz_max, self._env.dynamics.vxyz_max)
        obs[3:6] = vxyz
        # clipped_vxyz = np.clip(obs[3:6], -self._env.dynamics.vxyz_max, self._env.dynamics.vxyz_max)
        # clipped_w = np.clip(obs[15:18], -self._env.dynamics.omega_max, self._env.dynamics.omega_max)
        # normalized_vxyz = clipped_vxyz / self._env.dynamics.vxyz_max
        # normalized_w = clipped_w / (0.25 * self._env.dynamics.omega_max)
        # obs[3:6] = normalized_vxyz
        # obs[15:18] = clipped_w
        return obs

    def resset(self):
        obs = self._env.reset()
        return self._modify_obs(obs)

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        dist = np.linalg.norm(obs[0:3])
        obs = self._modify_obs(obs)
        # if the drone is out of bound, then break.
        if dist >= 4.0:
            print('out of range!!!!')
            done = True
        return obs, rew, done, info


class ActRepeatWrapper(gym.Wrapper):
    def __init__(self, env: QuadrotorEnv, num_repeat):
        super().__init__(env)
        self._env = env
        self._num_repeat = num_repeat

    def step(self, action):
        reward = 0.0
        discount = 0.99
        for i in range(self._num_repeat):
            obs, rew, done, info = self._env.step(action)
            reward += rew * discount
            if done:
                break
        return obs, reward, done, info

    def reset(self):
        return self._env.reset()


class CompatibilityWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CompatibilityWrapper, self).__init__(env)
        self._env = env

    def _modify_obs(self, obs):
        clipped_relative_pos = np.clip(obs[0:3], -1.5 * self._env.envs[0].box, 1.5 * self._env.envs[0].box)
        normalized_relative_pos = clipped_relative_pos / (1.5 * self._env.envs[0].box)
        obs[0:3] = normalized_relative_pos
        vxyz = np.clip(obs[3:6], -self._env.envs[0].dynamics.vxyz_max, self._env.envs[0].dynamics.vxyz_max)
        obs[3:6] = vxyz
        return obs

    def reset(self):
        obs = self._env.reset()
        obs = self._modify_obs(obs[0])
        return obs

    def step(self, action):
        action_list = [action]
        obs, rew, done, info = self._env.step(action_list)
        dist = np.linalg.norm(obs[0][0:3])
        obs = self._modify_obs(obs[0])
        # if the drone is out of bound, then break.
        if dist >= 4.0:
            print('out of range!!!!')
            done[0] = True
            rew[0] -= 0.25
        return obs, rew[0], done[0], info[0]
