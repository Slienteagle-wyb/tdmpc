import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt


def calculate_z_score(traj_sequences_dir, obs_dim):
    """Calculate the z-score of the trajectory sequences.
    Args:
        traj_sequences_dir: the directory of the trajectory sequences.
        obs_dim: the dimension of the observation.
    Returns:
        z_score: the z-score of the trajectory sequences.
    """
    # load the trajectory sequences
    traj_sequences = pickle.load(open(traj_sequences_dir, 'rb'))
    # calculate the mean and std
    print(len(traj_sequences))
    traj_sequences = np.concatenate(traj_sequences, axis=0)
    print(traj_sequences.shape)
    mean = np.mean(traj_sequences, axis=0)
    std = np.std(traj_sequences, axis=0)
    # calculate the z-score
    # z_score = (traj_sequences - mean) / std
    print('mean', mean, 'std', std)
    return mean, std


if __name__ == '__main__':
    plt.grid(True)
    # symlog = lambda x: torch.sign(x) * torch.log(1 + torch.abs(x))
    # symlog1 = lambda x: 2.0 * torch.sign(x) * torch.log(1 + torch.abs(1.0 * x))
    identity = lambda x: x
    pos_reward = lambda x: -2.0 * np.log(1.0 + x) + 1
    x_in = np.linspace(0, 4, 1000)
    y_in = pos_reward(x_in)
    # y_in = symlog(x_in)
    # y_in_1 = symlog1(x_in)
    y_identity = identity(x_in)
    y_exp_pos = np.exp(-x_in ** 2)
    # plt.plot(x_in.numpy(), y_in.numpy(), color='red')
    # plt.plot(x_in.numpy(), y_in_1.numpy(), color='blue')
    plt.plot(x_in, y_identity, color='black')
    plt.plot(x_in, y_in, color='green')
    plt.plot(x_in, y_exp_pos, color='red')
    plt.show()

    # work_dir = '/home/yibo/spaces/racing_traj/z_score/obs_sequences6_1000.pkl'
    # mean, std = calculate_z_score(work_dir, 18)
