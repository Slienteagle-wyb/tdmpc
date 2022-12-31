import time
import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from quad_envs import make_quadrotor_env_single
from tdmpc.src.cfg import parse_cfg
from gym_art.quadrotor_single.quad_utils import *
from gym_art.quadrotor_single.quadrotor import DummyPolicy, UpDownPolicy

__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def test_rollout(cfgs):
    # manual cfg for testing
    traj_num = 2
    policy_type = 'mellinger'  # 'updown', 'mellinger'
    render_each = 2
    plot_step = 2
    plot_obs = True
    plot_dyn_change = False
    plot_thrusts = False

    if policy_type == "mellinger":
        cfgs.env.raw_control = False
        cfgs.env.raw_control_zero_middle = True
        policy = DummyPolicy()  # since internal Mellinger takes care of the policy
    elif policy_type == "updown":
        cfgs.env.raw_control = True
        cfgs.env.raw_control_zero_middle = False
        policy = UpDownPolicy()

    env = make_quadrotor_env_single(cfgs)
    policy.dt = 1. / env.control_freq
    print('Reseting env ...')
    print("Obs repr: ", env.obs_repr)
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high, "size:",
              env.observation_space.high.size)
        print('Action space:', env.action_space.low, env.action_space.high, "size:", env.observation_space.high.size)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high,
              "size:", env.observation_space[0].spaces[0].high.size)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high, "size:",
              env.action_space[0].spaces[0].high.size)
    # input('Press any key to continue ...')
    # Collected statistics for dynamics
    dyn_param_names = [
        "mass",
        "inertia",
        "thrust_to_weight",
        "torque_to_thrust",
        "thrust_noise_ratio",
        "vel_damp",
        "damp_omega_quadratic",
        "torque_to_inertia"
    ]

    dyn_param_stats = [[] for i in dyn_param_names]

    action = np.array([0.0, 0.5, 0.0, 0.5])
    rollouts_id = 0
    start_time = time.time()
    # while rollouts_id < rollouts_num:
    for rollouts_id in tqdm.tqdm(range(traj_num)):
        s = env.reset()
        policy.reset()
        # Diagnostics
        observations = []
        velocities = []
        actions = []
        thrusts = []
        csv_data = []

        # Collecting dynamics params each episode
        if plot_dyn_change:
            for par_i, par in enumerate(dyn_param_names):
                dyn_param_stats[par_i].append(np.array(getattr(env.dynamics, par)).flatten())
                # print(par, dyn_param_stats[par_i][-1])

        t = 0
        r_sum = 0
        while True:
            if cfgs.env.render and (t % render_each == 0):
                env.render()
            action = policy.step(s)
            s, r, done, info = env.step(action)
            r_sum += r
            actions.append(action)
            thrusts.append(env.dynamics.thrust_cmds_damp)
            observations.append(s)
            # print('Step: ', t, ' Obs:', s)
            quat = R2quat(rot=s[6:15])
            csv_data.append(np.concatenate([np.array([1.0 / env.control_freq * t]), s[0:3], quat]))

            if plot_step is not None and t % plot_step == 0:
                plt.clf()

                if plot_obs:
                    observations_arr = np.array(observations)
                    # print('observations array shape', observations_arr.shape)
                    dimenstions = observations_arr.shape[1]
                    for dim in range(3, 6, 1):
                        plt.plot(observations_arr[:, dim])
                    plt.legend([str(x) for x in range(observations_arr.shape[1])])

                plt.pause(0.05)  # have to pause otherwise does not draw
                plt.draw()

            if done:
                break
            t += 1

        if plot_thrusts:
            plt.figure(3, figsize=(10, 10))
            ep_time = np.linspace(0, policy.dt * len(actions), len(actions))
            actions = np.array(actions)
            thrusts = np.array(thrusts)
            for i in range(4):
                plt.plot(ep_time, actions[:, i], label="Thrust desired %d" % i)
                plt.plot(ep_time, thrusts[:, i], label="Thrust produced %d" % i)
            plt.legend()
            plt.show(block=False)
            input("Press Enter to continue...")

        if plot_dyn_change:
            dyn_par_normvar = []
            dyn_par_means = []
            dyn_par_var = []
            plt.figure(2, figsize=(10, 10))
            for par_i, par in enumerate(dyn_param_stats):
                plt.subplot(3, 3, par_i + 1)
                par = np.array(par)

                ## Compute stats
                # print(dyn_param_names[par_i], par)
                dyn_par_means.append(np.mean(par, axis=0))
                dyn_par_var.append(np.std(par, axis=0))
                dyn_par_normvar.append(dyn_par_var[-1] / dyn_par_means[-1])

                if par.shape[1] > 1:
                    for vi in range(par.shape[1]):
                        plt.plot(par[:, vi])
                else:
                    plt.plot(par)
                # plt.title(dyn_param_names[par_i] + "\n Normvar: %s" % str(dyn_par_normvar[-1]))
                plt.title(dyn_param_names[par_i])
                print(dyn_param_names[par_i], "NormVar: ", dyn_par_normvar[-1])

        print("##############################################################")
        print("Total time: ", time.time() - start_time)

        # print('Rollouts are done!')
        # plt.pause(2.0)
        # plt.waitforbuttonpress()
        if plot_step is not None or plot_dyn_change:
            plt.show(block=False)
            input("Press Enter to continue...")


if __name__ == '__main__':
    cfgs = parse_cfg(Path().cwd() / __CONFIG__)
    test_rollout(cfgs)
