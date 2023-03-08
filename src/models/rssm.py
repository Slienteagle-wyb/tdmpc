import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import src.algorithm.helper as h
from copy import deepcopy
from src.models.rnns import NormGRUCell


class RSSMCell(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.min_std = 0.1
        self.max_std = 2.0
        self.state_act_proj = nn.Linear(cfg.action_dim + cfg.latent_dim, cfg.hidden_dim)
        self.sa_norm = torch.nn.LayerNorm(cfg.hidden_dim, eps=1e-3)
        self.prior_mlp = h.mlp_norm(cfg.hidden_dim, cfg.mlp_dim, 2 * cfg.latent_dim, cfg)
        self.post_mlp = h.mlp_norm(cfg.hidden_dim, cfg.mlp_dim, 2 * cfg.latent_dim, cfg)
        if cfg.norm_cell:
            self.gru_cell = NormGRUCell(cfg.latent_dim, cfg.hidden_dim)
        else:
            self.gru_cell = nn.GRUCell(cfg.latent_dim, cfg.hidden_dim)

    # prepare the init belief and latent state for rssm cell
    def init_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.cfg.belief_dim), device=device),
                torch.zeros((batch_size, self.cfg.latent_dim), device=device))

    def forward(self, init_states, action, obs_embed, mode='train'):
        belief, latent_state = init_states
        batch_size = belief.shape[0]
        # prepare the projected sa_embed as the input of deterministic path
        sa_embed = self.state_act_proj(torch.cat([latent_state, action], dim=-1))
        sa_embed = F.elu(self.sa_norm(sa_embed))
        belief = self.gru_cell(sa_embed, belief)
        # calculate the prio dist of next latent state using current belief
        prior = self.prior_mlp(belief)  # prior logits
        prior_dist = self.latent_state_dist(prior)
        prior_sample = prior_dist.rsample().reshape(batch_size, -1)
        # infer the posterior dist of next latent state using the transited belief and next obs_embed
        if mode == 'train':
            assert obs_embed is not None
            posterior = self.post_mlp(torch.cat([belief, obs_embed], dim=-1))
            posterior_dist = self.latent_state_dist(posterior)
            posterior_sample = posterior_dist.rsample().reshape(batch_size, -1)
        else:
            posterior = None
            posterior_sample = None

        return prior, prior_sample, posterior, posterior_sample, belief

    # separate the rollout process when planning
    @torch.no_grad()
    def dream(self, init_states, action, obs_embed=None, mode='rollout'):
        return self.forward(init_states, action, obs_embed, mode)

    def latent_state_dist(self, pp, mode='gaussian'):
        # pp: prior or posterior
        if mode == 'gaussian':
            mean, std = pp.chunk(2, dim=-1)
            std = self.max_std * torch.sigmoid(std) + self.min_std
            return D.independent.Independent(D.normal.Normal(mean, std), 1)
        elif mode == 'category':
            pass
