import torch
import torch.nn as nn
import algorithm.helper as h
import torch.nn.functional as F
from copy import deepcopy


class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input, state):
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h


class DGruDyna(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = deepcopy(cfg)
        # self.a_proj = nn.Linear(cfg.action_dim, cfg.action_latent, bias=False)
        self.prior_mlp = h.mlp_norm(cfg.hidden_dim, cfg.mlp_dim, cfg.latent_dim, cfg)
        if cfg.norm_cell:
            self.gru_cell = NormGRUCell(cfg.latent_dim+cfg.action_dim, cfg.hidden_dim)
        else:
            self.gru_cell = nn.GRUCell(cfg.latent_dim+cfg.action_dim, cfg.hidden_dim)

    # prepare the init hidden state for gru cell
    def init_hidden_state(self, batch_size, device):
        return torch.zeros((batch_size, self.cfg.hidden_dim), device=device)

    def forward(self, obs_embed, action, h_prev):
        # action_embed = self.a_proj(action)
        x = torch.cat([obs_embed, action], dim=-1)
        h = self.gru_cell(x, h_prev)
        z_pred = self.prior_mlp(h)
        return z_pred, h
