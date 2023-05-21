import gym
import robohive
from gym.wrappers import RecordEpisodeStatistics, NormalizeReward
from mjrl.utils.gym_env import GymEnv
from mj_envs import hand_manipulation_suite


class ActRepeatWrapper(gym.Wrapper):
    """
    Wrapper for mujoco environments to repeat actions.
    """

    def __init__(self, env, act_repeat):
        gym.Wrapper.__init__(self, env)
        self.act_repeat = act_repeat

    def step(self, action):
        if self.act_repeat == 1:
            obs, cum_reward, done, info = self.env.step(action)
            env_infos = [info]
        else:
            cum_reward = 0
            env_infos = []
            for _ in range(self.act_repeat):
                obs, reward, done, info = self.env.step(action)
                cum_reward += reward
                env_infos.append(info)
                if done:
                    break
        return obs, cum_reward, done, env_infos


def make_hms_env(cfg):
    """
    Make Hands manipulation suite environment for TD-MPC experiments.
    This simulation is powered by mujoco adapted from mj_envs
    """
    env_id = cfg.task
    env = GymEnv(env_id, act_repeat=cfg.action_repeat)
    cfg.obs_shape = (env.spec.observation_dim,)
    cfg.action_shape = (env.spec.action_dim,)
    cfg.action_dim = env.spec.action_dim
    cfg.buffer_shape = cfg.obs_shape
    return env


def make_robohive_env(cfg):
    env_id = cfg.task
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    env = ActRepeatWrapper(env, cfg.action_repeat)
    cfg.obs_shape = env.observation_space.shape
    cfg.buffer_shape = cfg.obs_shape
    cfg.action_shape = env.action_space.shape
    cfg.action_dim = env.action_space.shape[0]
    return env


if __name__ == '__main__':
    import robohive
    env = gym.make('pen-v1')
    env.reset()
    cum_reward = 0
    counts = 0
    for t in range(1000):
        # env.render()
        o, r, d, info = env.step(env.action_space.sample())
        cum_reward += r
        counts += 1
        print(r)
        if d:
            print('episode_done', info["solved"], info["done"], cum_reward, counts)
            env.reset()
            d = False
            cum_reward = 0
            counts = 0
