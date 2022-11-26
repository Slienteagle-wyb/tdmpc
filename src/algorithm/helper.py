import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm

__REDUCE__ = lambda b: 'mean' if b else 'none'
LOG_STD_MAX = 2
LOG_STD_MIN = -5


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enc(cfg):
    """Returns a TOLD encoder."""
    if cfg.modality == 'pixels':
        C = int(3 * cfg.frame_stack)
        layers = [NormalizeImg(),
                  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
    else:
        layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
                  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
    return nn.Sequential(*layers)


def enc_norm(cfg):
    """Returns a TOLD encoder."""
    if cfg.modality == 'pixels':
        C = int(3 * cfg.frame_stack)
        norm = init_normalization(cfg.mlp_dim, type_id=cfg.norm_type, one_d=True)
        layers = [NormalizeImg(),
                  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.BatchNorm2d(cfg.num_channels), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.BatchNorm2d(cfg.num_channels), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.BatchNorm2d(cfg.num_channels), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.BatchNorm2d(cfg.num_channels), nn.ReLU()]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.mlp_dim), norm, nn.ELU(),
                       nn.Linear(cfg.mlp_dim, cfg.latent_dim)])
    else:
        norm = init_normalization(cfg.enc_dim, type_id=cfg.norm_type, one_d=True)
        layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), norm, nn.ELU(),
                  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
    return nn.Sequential(*layers)


def dmlab_enc_norm(cfg):
    if cfg.modality == 'pixels':
        image_shape = (3*cfg.frame_stack, cfg.img_size, cfg.img_size)
        encoder = DmlabEncoderModelNorm(image_shape=image_shape, latent_size=cfg.latent_dim,
                                        hidden_sizes=cfg.mlp_dim)
    else:
        norm = init_normalization(cfg.enc_dim, type_id=cfg.norm_type, one_d=True)
        layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), norm, nn.ELU(),
                  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
        encoder = nn.Sequential(*layers)
    return encoder


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]), act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        nn.Linear(mlp_dim[1], out_dim))


def mlp_norm(in_dim, hidden_dim, out_dim, cfg, act_fn=nn.ELU(), norm_type='bn'):
    norm = init_normalization(hidden_dim, type_id=norm_type, one_d=True)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), norm, act_fn,
        nn.Linear(hidden_dim, out_dim)
    )


def q(cfg, act_fn=nn.ELU()):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
                         nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
                         nn.Linear(cfg.mlp_dim, 1))


def soft_q(cfg):
    return nn.Sequential(nn.Linear(cfg.obs_shape[0] + cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim),
                         nn.Tanh(),
                         nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
                         nn.Linear(cfg.mlp_dim, 1))


class SoftActor(nn.Module):
    def __init__(self, cfg, act_fn=nn.ELU()):
        super(SoftActor, self).__init__()
        if cfg.latent_policy:
            in_dim = cfg.latent_dim
        else:
            in_dim = cfg.obs_shape[0]
        self.fc1 = nn.Linear(in_dim, cfg.mlp_dim)
        self.act_fn1 = act_fn
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.mlp_dim)
        self.act_fn2 = act_fn
        self.fc_mean = nn.Linear(cfg.mlp_dim, cfg.action_dim)
        self.fc_logstd = nn.Linear(cfg.mlp_dim, cfg.action_dim)

    def forward(self, z):
        z = self.act_fn1(self.fc1(z))
        z = self.act_fn2(self.fc2(z))
        mean = self.fc_mean(z)
        log_std = self.fc_logstd(z)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # low and high bound rescale
        return mean, log_std

    def get_action(self, z):
        mean, log_std = self(z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action_prim = normal.rsample()
        action = torch.tanh(action_prim)
        log_prob = normal.log_prob(action_prim)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class RandomShiftsAug(nn.Module):
    """
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""

    def __init__(self, cfg):
        super().__init__()
        self.pad = int(cfg.img_size / 21) if cfg.modality == 'pixels' else None

    def forward(self, x):
        if not self.pad:
            return x
        shape_len = len(x.size())
        if shape_len == 5:
            t, n, c, h, w = x.size()
            x = x.reshape(t*n, c, h, w)  # apply the same aug to the same traj.

        n_stacked, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n_stacked, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n_stacked, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        shifted = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        if shape_len == 5:
            shifted = shifted.reshape(t, n, c, h, w)

        return shifted


class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        self.obs = torch.empty((cfg.episode_length + 1, *init_obs.shape), dtype=dtype, device=self.device)
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done):
        self.obs[self._idx + 1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


class ModelRollout(Episode):
    def __init__(self, cfg, init_latent):
        super(ModelRollout, self).__init__(cfg, init_latent)
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        self.obs = torch.empty((cfg.horizon+1, *init_latent.shape), dtype=dtype, device=self.device)
        self.obs[0] = torch.tensor(init_latent, dtype=dtype, device=self.device)
        self.action = torch.empty((cfg.horizon, cfg.dream_trace, cfg.action_dim), dtype=torch.float32, device=cfg.device)
        self.reward = torch.empty((cfg.horizon, cfg.dream_trace, 1), dtype=torch.float32, device=cfg.device)

    def add(self, z_dream, action, reward_pred, done):
        self.obs[self._idx+1] = z_dream
        self.action[self._idx] = action
        self.done = done
        self._idx += 1


class ReplayBuffer:
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default.
    """

    def __init__(self, cfg, latent_plan=False):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(cfg.device)
        self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        if cfg.modality == 'state':
            obs_shape = cfg.obs_shape
        else:
            obs_shape = (3, *cfg.obs_shape[-2:])
        self._obs = torch.empty((self.capacity + 1, *obs_shape), dtype=dtype, device=self.device)
        self._last_obs = torch.empty((self.capacity // self.cfg.episode_length, *cfg.obs_shape), dtype=dtype,
                                     device=self.device)  # last obs of an episodic trajectory
        self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self._eps = 1e-6
        self._full = False
        self.idx = 0
        if not latent_plan:
            self.batch_size = self.cfg.batch_size
            self.horizon = self.cfg.env_horizon
        else:
            self.batch_size = self.cfg.batch_size
            self.horizon = self.cfg.horizon

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self._obs[self.idx:self.idx + self.cfg.episode_length] = episode.obs[
                                                                 :-1] if self.cfg.modality == 'state' else episode.obs[
                                                                                                           :-1, -3:]
        self._last_obs[self.idx // self.cfg.episode_length] = episode.obs[-1]
        self._action[self.idx:self.idx + self.cfg.episode_length] = episode.action
        self._reward[self.idx:self.idx + self.cfg.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
        mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length - self.cfg.horizon
        new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
        new_priorities[mask] = 0
        self._priorities[self.idx:self.idx + self.cfg.episode_length] = new_priorities
        self.idx = (self.idx + self.cfg.episode_length) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        if self.cfg.modality == 'state':
            return arr[idxs]
        obs = torch.empty((self.batch_size, 3 * self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype,
                          device=torch.device('cuda'))
        obs[:, -3:] = arr[idxs].cuda()
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i + 1) * 3:-i * 3] = arr[_idxs].cuda()
        return obs.float()

    def sample(self):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(total, self.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = self._last_obs.shape[1:] if self.cfg.modality == 'state' else (
            3 * self.cfg.frame_stack, *self._last_obs.shape[-2:])
        next_obs = torch.empty((self.horizon + 1, self.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((self.horizon + 1, self.batch_size, *self._action.shape[1:]), dtype=torch.float32,
                             device=self.device)
        reward = torch.empty((self.horizon + 1, self.batch_size), dtype=torch.float32, device=self.device)
        for t in range(self.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs + 1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs + 1) % self.cfg.episode_length == 0
        next_obs[-1, mask] = self._last_obs[_idxs[mask] // self.cfg.episode_length].cuda().float()
        if not action.is_cuda:
            action, reward, idxs, weights = \
                action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights


class RolloutBuffer:
    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(cfg.device)
        self.capacity = int(min(cfg.train_steps, cfg.max_buffer_size) / self.cfg.dream_trace)
        self._obs = torch.empty((self.capacity + 1, cfg.dream_trace, cfg.latent_dim), dtype=torch.float32,
                                device=self.device)
        self._last_obs = torch.empty((self.capacity // self.cfg.horizon, cfg.dream_trace, cfg.latent_dim),
                                     dtype=torch.float32, device=self.device)  # last obs of a dream rollout
        self._action = torch.empty((self.capacity, cfg.dream_trace, cfg.action_dim),
                                   dtype=torch.float32, device=cfg.device)
        self._reward = torch.empty((self.capacity, cfg.dream_trace, 1), dtype=torch.float32, device=self.device)
        self._eps = 1e-6
        self._full = False
        self.idx = 0
        self.batch_size = self.cfg.batch_size // self.cfg.dream_trace
        self.horizon = self.cfg.env_horizon

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self._obs[self.idx:self.idx + self.cfg.horizon] = episode.obs[:-1]
        self._last_obs[self.idx // self.cfg.horizon] = episode.obs[-1]
        self._action[self.idx:self.idx + self.cfg.horizon] = episode.action
        self._reward[self.idx:self.idx + self.cfg.horizon] = episode.reward
        self.idx = (self.idx + self.cfg.horizon) % self.capacity
        self._full = self._full or self.idx == 0

    def sample(self):
        idxs = torch.from_numpy(np.random.choice(self.capacity if self._full else self.idx, self.batch_size,
                                                 replace=not self._full)).to(self.device)
        obs = self._obs[idxs]
        next_obs_shape = self._last_obs.shape[1:]
        next_obs = torch.empty((self.horizon + 1, self.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((self.horizon + 1, self.batch_size, *self._action.shape[1:]),
                             dtype=torch.float32, device=self.device)
        reward = torch.empty((self.horizon + 1, self.batch_size, *self._reward.shape[1:]), dtype=torch.float32, device=self.device)
        for t in range(self.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = self._obs[_idxs + 1]
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
        if not action.is_cuda:
            action, reward, idxs = action.cuda(), reward.cuda(), idxs.cuda()

        return obs, next_obs, action, reward, idxs


def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration, start = [float(g) for g in match.groups()]
            mix = np.clip((step-start) / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


def init_normalization(channels, type_id="bn", affine=True, one_d=False):
    assert type_id in ["bn", "ln", "in", "gn", "max", "none", None]
    if type_id == "bn":
        if one_d:
            return torch.nn.BatchNorm1d(channels, affine=affine)
        else:
            return torch.nn.BatchNorm2d(channels, affine=affine)
    elif type_id == "ln":
        if one_d:
            return torch.nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return torch.nn.GroupNorm(1, channels, affine=affine)
    elif type_id == "in":
        return torch.nn.GroupNorm(channels, channels, affine=affine)
    elif type_id == "gn":
        groups = max(min(32, channels//4), 1)
        return torch.nn.GroupNorm(groups, channels, affine=affine)
    elif type_id == "none" or type_id is None:
        return torch.nn.Identity()
