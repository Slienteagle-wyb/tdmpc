import torch
import torch.nn as nn
import src.algorithm.helper as h
from src.models.rssm import RSSMCell
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class RSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.dmlab_enc_norm(cfg)
        self._decoder = h.decoder(cfg)
        self._dynamics = RSSMCell(cfg)
        self._reward = h.mlp(cfg.latent_dim+cfg.hidden_dim, cfg.mlp_dim, 1, cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward]:
            m[-1].weight.data.zero_()
            m[-1].bias.data.zero_()

    def h(self, obs):
        if self.cfg.modality == 'pixel':
            lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
            obs = obs.view(T * B, *img_shape)
            obs_embed = self._encoder(obs)
            obs_embed = restore_leading_dims(obs_embed, lead_dim, T, B)
        else:
            raise 'only pixel input is supported'
        return obs_embed

    def next(self, init_states, action, obs_embed, mode):
        prior, prior_sample, posterior, posterior_sample, belief = self._dynamics(init_states, action, obs_embed, mode)
        if obs_embed is not None:
            latent_state = posterior_sample
        else:
            latent_state = prior_sample
        reward_pred = self._reward(torch.cat([belief, latent_state], dim=-1))
        return (belief, latent_state), reward_pred, prior, posterior


class PlaNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.std = h.linear_schedule(cfg.std_schedule, 0)  # std schedule for cem planning
        self.model = RSSM(cfg).to(self.device)
        self.model.eval()
        self.plan_horizon = 1  # add a warmup schedule for planning horizon
        self.batch_size = cfg.batch_size

        total_epochs = int(cfg.train_steps / cfg.episode_length)
