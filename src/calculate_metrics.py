import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R


def calculate_dist_mean():
    with open('/home/yibo/spaces/set_points/state_sequences_100.pkl', 'rb') as f:
        states_seqs = pickle.load(f)
    full_seques = []
    j = 0
    for i in range(len(states_seqs)):
        if states_seqs[i].shape[0] == 1000:
            full_seques.append(states_seqs[i])
            j += 1
    print(j)

    dist_list = []
    for k in range(len(full_seques)):
        traj = full_seques[k]
        dist = np.linalg.norm(traj[:, 1:4] - traj[:, -3:], axis=1)
        stable_mean_dist = np.mean(dist[-100:])
        dist_list.append(stable_mean_dist)
    print(np.mean(dist_list))


def plot_3d_traj_colored_by_vel():
    with open('/home/yibo/spaces/set_points/flat_state_sequences_test1.pkl', 'rb') as f:
        flat_seqs = pickle.load(f)
    with open('/home/yibo/spaces/set_points/tdmpc_state_sequences_test1.pkl', 'rb') as f:
        rl_seqs = pickle.load(f)
    flat_traj = flat_seqs[0]
    rl_traj = rl_seqs[0]

    flat_ticks = flat_traj[:, 0]
    rl_ticks = rl_traj[:500, 0]
    flat_pos = flat_traj[:, 1:4]
    rl_pos = rl_traj[:500, 1:4]
    flat_vel = flat_traj[:, 4:7]
    rl_vel = rl_traj[:500, 4:7]

    vel_amp_flat = np.linalg.norm(flat_vel, axis=1)
    vel_amp_rl = np.linalg.norm(rl_vel, axis=1)
    all_vel_amp = np.concatenate([vel_amp_flat, vel_amp_rl])

    norm = Normalize()
    norm.autoscale(all_vel_amp)
    color_map = cm.viridis

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca(projection='3d')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    for i in range(len(rl_ticks) - 1):
        ax.plot(rl_pos[i:i + 2, 0], rl_pos[i:i + 2, 1], rl_pos[i:i + 2, 2], color=color_map(norm(vel_amp_rl[i])))
    for i in range(len(flat_ticks) - 1):
        ax.plot(flat_pos[i:i + 2, 0], flat_pos[i:i + 2, 1], flat_pos[i:i + 2, 2],
                color=color_map(norm(vel_amp_flat[i])))

    sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array(all_vel_amp)
    plt.colorbar(sm, label='速度幅值', shrink=0.4, pad=0.1)

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')

    start_marker = '*'
    target_marker = 'P'
    start_color = 'blue'
    target_color = 'red'
    marker_size = 100

    start_scatter = ax.scatter(flat_pos[0, 0], flat_pos[0, 1], flat_pos[0, 2], c=start_color, marker=start_marker,
                               s=marker_size)
    target_scatter = ax.scatter(rl_pos[-1, 0], rl_pos[-1, 1], rl_pos[-1, 2], c=target_color, marker=target_marker,
                                s=marker_size)

    legend1 = ax.legend([start_scatter, target_scatter], ['起始点', '目标点'], loc='upper right', numpoints=1,
                        scatterpoints=1, fontsize=12)
    ax.add_artist(legend1)

    plt.savefig('/home/yibo/spaces/set_points/rl_traj_test1.png', dpi=300)


def plot_rpy_omega_comparison():
    with open('/home/yibo/spaces/set_points/flat_success_state_sequences_test1.pkl', 'rb') as f:
        flat_seqs = pickle.load(f)
    with open('/home/yibo/spaces/set_points/tdmpc_success_state_sequences_test1.pkl', 'rb') as f:
        rl_seqs = pickle.load(f)
    flat_traj = flat_seqs[0]
    rl_traj = rl_seqs[0]

    flat_ticks = flat_traj[:, 0]
    flat_quat = flat_traj[:, 7:16]
    flat_rpy = R.from_matrix(flat_quat.reshape(-1, 3, 3)).as_euler('xyz', degrees=True)
    flat_omega = flat_traj[:, 16:19] * 180 / np.pi
    rl_ticks = rl_traj[:, 0]
    rl_quat = rl_traj[:, 7:16]
    rl_rpy = R.from_matrix(rl_quat.reshape(-1, 3, 3)).as_euler('xyz', degrees=True)
    rl_omega = rl_traj[:, 16:19] * 180 / np.pi

    matplotlib.rcParams['font.family'] = ['SimHei']  # Replace 'SimHei' with the name of the Chinese font you installed
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    axs[0, 0].grid(True)
    axs[0, 0].plot(flat_ticks, flat_rpy[:, 0], color='#54b9c5', label='微分平坦')
    axs[0, 0].plot(rl_ticks, rl_rpy[:, 0], color='#d34e38', label='强化学习')
    axs[0, 0].set_ylabel('滚转/度')
    axs[0, 0].set_xlabel('时间/秒')
    axs[0, 0].legend()

    axs[0, 1].grid(True)
    axs[0, 1].plot(flat_ticks, flat_rpy[:, 1], color='#54b9c5', label='强化学习')
    axs[0, 1].plot(rl_ticks, rl_rpy[:, 1], color='#d34e38', label='强化学习')
    axs[0, 1].set_ylabel('俯仰/度')
    axs[0, 1].set_xlabel('时间/秒')

    axs[1, 0].grid(True)
    axs[1, 0].plot(flat_ticks, flat_omega[:, 2], color='#54b9c5', label='强化学习')
    axs[1, 0].plot(rl_ticks, rl_omega[:, 2], color='#d34e38', label='强化学习')
    axs[1, 0].set_ylabel('滚度/秒)')
    axs[1, 0].set_xlabel('时间/秒')

    axs[1, 1].grid(True)
    axs[1, 1].plot(flat_ticks, flat_omega[:, 1], color='#54b9c5', label='强化学习')
    axs[1, 1].plot(rl_ticks, rl_omega[:, 1], color='#d34e38', label='强化学习')
    axs[1, 1].set_ylabel('俯(度秒)')
    axs[1, 1].set_xlabel('时间/秒')

    # plt.show()
    plt.savefig('/home/yibo/spaces/set_points/rl_omegas.png', dpi=300)


def calculate_mean_max_vel():
    with open('/home/yibo/spaces/set_points/tdmpc_racing_sequences_test1.pkl', 'rb') as f:
    # with open('/home/yibo/spaces/set_points/flat_racing_sequences_test1.pkl', 'rb') as f:
        states_seqs = pickle.load(f)
    acc_run_time = []
    vel_mean_list = []
    vel_max_list = []
    for i in range(len(states_seqs)):
        acc_run_time.append(states_seqs[i][-1][0])
        vel = states_seqs[i][:, 4:7]
        vel_amp = np.linalg.norm(vel, axis=1)
        vel_mean = np.mean(vel_amp)
        vel_max = np.max(vel_amp)
        vel_mean_list.append(vel_mean)
        vel_max_list.append(vel_max)
    mean_run_time = np.mean(acc_run_time)
    global_mean_vel = np.mean(vel_mean_list)
    global_max_vel = np.max(vel_max_list)
    print('mean run time: ', mean_run_time, 'global mean vel: ', global_mean_vel, 'global max vel: ', global_max_vel)


def plot_racing_traj_colorized_by_vel():
    with open('/home/yibo/spaces/set_points/tdmpc_racing_sequences_test1.pkl', 'rb') as f:
        rl_seqs = pickle.load(f)
    rl_traj = rl_seqs[0]

    rl_ticks = rl_traj[:, 0]
    rl_pos = rl_traj[:, 1:4]
    rl_vel = rl_traj[:, 4:7]
    setpoints = rl_traj[:, 1:4]

    vel_amp_rl = np.linalg.norm(rl_vel, axis=1)

    norm = Normalize()
    norm.autoscale(vel_amp_rl)
    color_map = cm.viridis

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca(projection='3d')
    ax.set_box_aspect([1, 1, 0.3])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    for i in range(len(rl_ticks) - 1):
        ax.plot(rl_pos[i:i + 2, 0], rl_pos[i:i + 2, 1], rl_pos[i:i + 2, 2], color=color_map(norm(vel_amp_rl[i])))

    sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array(vel_amp_rl)
    plt.colorbar(sm, label='速度幅值', shrink=0.4, pad=0.1)

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')

    start_marker = '*'
    target_marker = 'P'
    start_color = 'blue'
    target_color = 'red'
    marker_size = 10

    start_scatter = ax.scatter(rl_pos[0, 0], rl_pos[0, 1], rl_pos[0, 2], c=start_color, marker=start_marker,
                               s=marker_size)
    target_scatter = ax.scatter(rl_pos[-1, 0], rl_pos[-1, 1], rl_pos[-1, 2], c=target_color, marker=target_marker,
                                s=marker_size)

    set_points = [ax.scatter(setpoints[i, 0], setpoints[i, 1], setpoints[i, 2], c='green', marker='o', s=marker_size) for i in range(100, 700, 150)]

    legend1 = ax.legend([start_scatter, set_points[0], target_scatter], ['起始点', '中间目标', '目标点'], loc='upper right', numpoints=1,
                        scatterpoints=1, fontsize=12)
    ax.add_artist(legend1)

    # plt.show()
    plt.savefig('/home/yibo/spaces/set_points/rl_racing_traj1.png', dpi=300)


if __name__ == '__main__':
    plot_rpy_omega_comparison()
