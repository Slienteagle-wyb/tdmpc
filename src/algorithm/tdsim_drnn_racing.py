import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorednoise
from copy import deepcopy
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.algos.utils.optim_factory import create_optimizer
import src.algorithm.helper as h
from src.models.gru_dyna import DGruDyna


class DSSMR(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._dynamics = DGruDyna(cfg)
        self._reward = h.mlp(cfg.hidden_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
        # self.state_adaptor = h.mlp_norm(in_dim=cfg.latent_dim + cfg.env.num_vis_gates * 4,
        #                                 hidden_dim=cfg.hidden_dim, out_dim=cfg.latent_dim, cfg=cfg)
        if self.cfg.normalize:
            self._encoder = h.dmlab_enc_norm(cfg)
            self._predictor = h.mlp_norm(cfg.latent_dim, cfg.mlp_dim, cfg.latent_dim, cfg)
        else:
            self._encoder = h.enc(cfg)
            self._predictor = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.latent_dim)
        self.apply(h.orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def freeze_encoder(self, enable=False):
        h.set_requires_grad(self._encoder, enable)

    def track_model_grad(self, enable=True):
        for m in [self._Q1, self._Q2, self._reward, self._dynamics]:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        if self.cfg.modality == 'pixels':
            lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
            obs = obs.view(T*B, *img_shape)
            latent_feature, _ = self._encoder(obs)
            latents = restore_leading_dims(latent_feature, lead_dim, T, B)
        else:
            latents = self._encoder(obs)
        return latents

    def next(self, z, a, h_prev):
        z, hidden = self._dynamics(z, a, h_prev)
        reward_pred = self._reward(hidden)
        return z, hidden, reward_pred

    def init_hidden_state(self, batch_size, device):
        return self._dynamics.init_hidden_state(batch_size, device)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)

    def pred_z(self, z):
        return self._predictor(z)


class TdMpcSimDssmR:
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, 0)
        self.model = DSSMR(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()

        self.plan_horizon = 1
        self.batch_size = cfg.batch_size
        total_epochs = int(cfg.train_steps / cfg.episode_length)
        self._optim_initialize(total_epochs)

    def _optim_initialize(self, total_epochs):
        self.optim = create_optimizer(model=self.model, optim_id=self.cfg.optim_id,
                                      lr=self.cfg.lr)
        self.pi_optim = create_optimizer(model=self.model._pi, optim_id=self.cfg.optim_id,
                                         lr=self.cfg.pi_lr)

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict(),
                'optim': self.optim.state_dict(), }

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])
        # self.optim.load_state_dict(d['optim'])

    @torch.no_grad()
    def estimate_value(self, z, actions, hidden):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.plan_horizon):
            z, hidden, reward = self.model.next(z, actions[t], hidden)
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G.nan_to_num_(0), float(reward.mean().item())

    def sample_action_sequence(self, num_samples, mean, std):
        if self.cfg.noise_beta > 0.:
            noise = colorednoise.powerlaw_psd_gaussian(self.cfg.noise_beta,
                                                       size=(num_samples,
                                                             self.cfg.action_dim, 12))
            noise = torch.from_numpy(noise).float().to(self.device).permute(2, 0, 1)[:self.plan_horizon]
        else:
            noise = torch.randn(self.plan_horizon, num_samples,
                                self.cfg.action_dim, device=self.device)
        actions_sampled = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * noise, -1, 1)
        return actions_sampled

    @torch.no_grad()
    def plan(self, obs, hidden, eval_mode=False, step=None, t0=True, fine_tuning=False):
        intrinsic_reward_mean, reward_mean = 0, 0
        plan_metrics = {'external_reward_mean': 0.0, 'current_std': 0.0}

        if step < self.cfg.seed_steps and not eval_mode and not fine_tuning:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1), None, plan_metrics

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # obs_drone, obs_gate = torch.split(obs, [self.cfg.obs_shape[0], 4 * self.cfg.env.num_vis_gates], dim=-1)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        if horizon != self.plan_horizon and t0:
            self.plan_horizon = horizon
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, step)
        num_pi_trajs = int(self.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(self.plan_horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z_pi = self.model.h(obs).repeat(num_pi_trajs, 1)
            hidden_pi = hidden.repeat(num_pi_trajs, 1)
            for t in range(self.plan_horizon):
                pi_actions[t] = self.model.pi(z_pi, self.cfg.min_std)
                z_pi, hidden_pi, _ = self.model.next(z_pi, pi_actions[t], hidden_pi)

        # Initialize state and parameters
        z_plan = self.model.h(obs).repeat(self.cfg.num_samples + num_pi_trajs, 1)
        hidden_plan = hidden.repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(self.plan_horizon, self.cfg.action_dim, device=self.device)
        std = 2.0 * torch.ones(self.plan_horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            # parameterized action of mpc
            actions = self.sample_action_sequence(self.cfg.num_samples, mean, std)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value, reward_mean = self.estimate_value(z_plan, actions, hidden_plan)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]  # select action of high value

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                    score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        # step the model to calculate the next hidden state
        z, hidden, _ = self.model.next(z_plan[0:1], a.unsqueeze(0), hidden)

        plan_metrics.update({'external_reward_mean': reward_mean,
                             'current_std': std.mean().item()})

        return a, hidden, plan_metrics

    @torch.no_grad()
    def plan_icem(self, obs, hidden, eval_mode=False, step=None, t0=True, fine_tuning=False):
        intrinsic_reward_mean, reward_mean = 0, 0
        extend_horizon = False
        plan_metrics = {'external_reward_mean': 0.0, 'current_std': 0.0}

        if step < self.cfg.seed_steps and not eval_mode and not fine_tuning:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32,
                               device=self.device).uniform_(-1, 1), None, plan_metrics

        # obs_drone, obs_gate = torch.split(obs, [self.cfg.obs_shape[0], 4 * self.cfg.env.num_vis_gates], dim=-1)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        if horizon != self.plan_horizon and t0:
            self.plan_horizon = horizon
            extend_horizon = True
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, step)

        mean = torch.zeros(self.plan_horizon, self.cfg.action_dim, device=self.device)
        std = 0.5 * torch.ones(self.plan_horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]

        num_samples = self.cfg.num_samples
        num_pi_trajs = int(self.mixture_coef * num_samples)
        assert num_pi_trajs > 0
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.h(obs)
        pi_actions = torch.empty(self.plan_horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
        z_pi = z.repeat(num_pi_trajs, 1)
        hidden_pi = hidden.repeat(num_pi_trajs, 1)
        for t in range(self.plan_horizon):
            pi_actions[t] = self.model.pi(z_pi, self.cfg.min_std)
            z_pi, hidden_pi, _ = self.model.next(z_pi, pi_actions[t], hidden_pi)

        # Iterate CEM
        for i in range(self.cfg.iterations):
            # apply schedule to decrease the number of samples
            if i > 0:
                num_samples = max(2 * self.cfg.num_elites, int(num_samples / self.cfg.factor_decrease_num))
                num_pi_trajs = int(self.mixture_coef * num_samples)
                assert num_pi_trajs > 0
            # reuse th elite actions from the previous iteration or step
            if self.cfg.fraction_elites_reused > 0 and hasattr(self, '_elite_actions'):
                num_elite_trajs = int(self.cfg.fraction_elites_reused * self.cfg.num_elites)
                num_trajs = num_samples + num_pi_trajs + num_elite_trajs
            else:
                num_elite_trajs = 0
                num_trajs = num_samples + num_pi_trajs

            zs_plan = z.repeat(num_trajs, 1)
            hidden_plan = hidden.repeat(num_trajs, 1)
            actions_sampled = self.sample_action_sequence(num_samples, mean, std)

            if i == self.cfg.iterations - 1:
                actions_sampled[:, 0] = mean  # use the mean of the last iteration (icem_best-a)
            # fetch the previous elite actions if reuse them from history
            # reused the elite actions from previous planning step
            if i == 0 and self.cfg.shift_elites_over_time and hasattr(self, '_elite_actions'):
                num_elite_trajs = int(self.cfg.fraction_elites_reused * self.cfg.num_elites)
                reused_actions = self._elite_actions[1:, :num_elite_trajs]
                if extend_horizon:
                    last_actions = self.sample_action_sequence(num_elite_trajs, mean, std)[-2:]
                else:
                    last_actions = self.sample_action_sequence(num_elite_trajs, mean, std)[-1:]
                reused_actions = torch.cat([reused_actions, last_actions], dim=0)
            # reuse the elite actions from previous iteration
            if i > 0 and self.cfg.keep_previous_elites:
                reused_actions = self._elite_actions[:, :num_elite_trajs]

            # concatenate the actions
            if num_elite_trajs > 0:
                actions = torch.cat([actions_sampled, reused_actions, pi_actions[:, :num_pi_trajs]], dim=1)
            else:
                actions = torch.cat([actions_sampled, pi_actions[:, :num_pi_trajs]], dim=1)

            # Compute elite actions
            value, reward_mean = self.estimate_value(zs_plan, actions, hidden_plan)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]  # select action of high value
            self._elite_actions = elite_actions

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                    score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        # step the model to calculate the next hidden state
        z, hidden, _ = self.model.next(z[0:1], a.unsqueeze(0), hidden)

        plan_metrics.update({'external_reward_mean': reward_mean,
                             'current_std': std.mean().item()})

        return a, hidden, plan_metrics

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z, a))
            # pi_loss += -Q.mean() * (self.cfg.rho ** t)
            pi_loss += -Q.mean()

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        # next_z = self.model.h(next_obs)
        q = torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
        td_target = reward + self.cfg.discount * q
        return td_target

    def similarity_loss(self, queries, keys):
        queries = torch.cat(queries, dim=0)  # (cfg.horizon*batch_size, latent_dim)
        queries = self.model.pred_z(queries)
        keys = torch.cat(keys, dim=0)
        queries_norm = F.normalize(queries, dim=-1, p=2)
        keys_norm = F.normalize(keys, dim=-1, p=2)
        return 2.0 - 2.0 * (queries_norm * keys_norm).sum(dim=-1)  # (cfg.horizon*batch_size, )

    def consistency_loss(self, z_pred, z_target):
        seg_len = len(z_pred)
        z_pred = torch.cat(z_pred, dim=0)  # (cfg.horizon*batch_size, latent_dim)
        z_target = torch.cat(z_target, dim=0)
        seg_loss = torch.mean(h.mse(z_pred, z_target), dim=1, keepdim=True)
        seg_loss_mean = seg_loss.reshape(seg_len, self.cfg.batch_size, -1).mean(dim=0)
        return seg_loss_mean

    @torch.no_grad()
    def model_rollout(self, z, actions):
        T, B = actions.shape[:2]
        zs_latent = []
        hidden = self.model.init_hidden_state(z.shape[0], z.device)
        for t in range(T):
            z, hidden, _ = self.model.next(z, actions[t], hidden)
            zs_latent.append(z)
        zs_latent = torch.stack(zs_latent, dim=0)
        return zs_latent

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        # obs [batch, state_dim], actions [horizon+1, batch, act_dim]
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        self.optim.zero_grad(set_to_none=True)

        # Representation
        z = self.model.h(self.aug(obs))
        next_zs = self.model_target.h(self.aug(next_obses))
        online_next_zs = self.model.h(self.aug(next_obses))
        # input_zs = torch.cat([z.unsqueeze(0), online_next_zs], dim=0)

        similarity_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        hidden = self.model.init_hidden_state(z.shape[0], z.device)
        zs = [z.detach()]  # obs embedding for policy learning

        for t in range(self.cfg.horizon):
            # Predictions
            rho = (self.cfg.rho ** t)
            zs_query, zs_key = [], []
            reward_loss_shooting = 0
            Q1, Q2 = self.model.Q(z, action[t])
            z, hidden, reward_pred = self.model.next(z, action[t], hidden)
            with torch.no_grad():
                td_target = self._td_target(online_next_zs[t], reward[t])
            reward_loss_shooting += h.mse(50*reward_pred, 50*reward[t])
            value_loss += (h.mse(Q1, td_target) + h.mse(Q2, td_target)) * rho
            priority_loss += (h.l1(Q1, td_target) + h.l1(Q2, td_target)) * rho

            zs_query.append(z)
            zs_key.append(next_zs[t].detach())
            zs.append(z.detach())

            # conduct overshooting for the next states embedding
            shoot_hidden = torch.clone(hidden)
            shoot_z = torch.clone(z)
            shoot_count = 0
            for j in range(t+1, self.cfg.horizon):
                shoot_count += 1
                shoot_z, shoot_hidden, reward_pred = self.model.next(shoot_z, action[j], shoot_hidden)
                zs_query.append(shoot_z)
                zs_key.append(next_zs[j].detach())
                reward_loss_shooting += h.mse(50*reward_pred, 50*reward[j])
            reward_loss += reward_loss_shooting / (shoot_count+1)
            similarity_loss += self.similarity_loss(zs_query, zs_key).reshape(self.cfg.horizon-t, self.cfg.batch_size, -1).sum(dim=0)  # (batch_size, )
            # consistency_loss += self.consistency_loss(zs_query, zs_key)

        # Optimize model
        total_loss = self.cfg.similarity_coef * similarity_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4)
        weighted_loss = (total_loss * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm,
                                                   error_if_nonfinite=False)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        # pi_loss = self.analytic_update_pi(zs[0], action[0])
        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {'consistency_loss': float(similarity_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                'grad_norm': float(grad_norm),
                'mixture_coef': self.mixture_coef}

    def finetune(self, replay_buffer, step, demo_buffer=None):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        # obs [batch, state_dim], actions [horizon+1, batch, act_dim]
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.demo_batch_size = int(h.linear_schedule(self.cfg.demo_schedule, step) * self.batch_size)
        replay_buffer.batch_size = self.batch_size - self.demo_batch_size
        demo_buffer.batch_size = self.demo_batch_size

        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        demo_obs, demo_next_obses, demo_action, demo_reward, demo_idxs, demo_weights = demo_buffer.sample()

        obs, next_obses, action, reward, idxs, weights = (
            torch.cat([obs, demo_obs]),
            torch.cat([next_obses, demo_next_obses], dim=1),
            torch.cat([action, demo_action], dim=1),
            torch.cat([reward, demo_reward], dim=1),
            torch.cat([idxs, demo_idxs]),
            torch.cat([weights, demo_weights])
        )

        self.model.train()
        self.optim.zero_grad(set_to_none=True)

        # Representation
        z = self.model.h(self.aug(obs))
        next_zs = self.model_target.h(self.aug(next_obses))
        online_next_zs = self.model.h(self.aug(next_obses))
        # input_zs = torch.cat([z.unsqueeze(0), online_next_zs], dim=0)
        zs = [z.detach()]  # obs embedding for policy learning

        similarity_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        hidden = self.model.init_hidden_state(z.shape[0], z.device)
        for t in range(self.cfg.horizon):
            # Predictions
            rho = (self.cfg.rho ** t)
            zs_query, zs_key = [], []
            reward_loss_shooting = 0
            Q1, Q2 = self.model.Q(z, action[t])
            z, hidden, reward_pred = self.model.next(z, action[t], hidden)
            with torch.no_grad():
                td_target = self._td_target(online_next_zs[t], reward[t])
            reward_loss_shooting += h.mse(50*reward_pred, 50*reward[t])
            value_loss += (h.mse(Q1, td_target) + h.mse(Q2, td_target)) * rho
            priority_loss += (h.l1(Q1, td_target) + h.l1(Q2, td_target)) * rho

            zs_query.append(z)
            zs_key.append(next_zs[t].detach())
            zs.append(z.detach())

            # conduct overshooting for the next states embedding
            shoot_hidden = torch.clone(hidden)
            shoot_z = torch.clone(z)
            shoot_count = 0
            for j in range(t+1, self.cfg.horizon):
                shoot_count += 1
                shoot_z, shoot_hidden, reward_pred = self.model.next(shoot_z, action[j], shoot_hidden)
                zs_query.append(shoot_z)
                zs_key.append(next_zs[j].detach())
                reward_loss_shooting += h.mse(50*reward_pred, 50*reward[j])
            reward_loss += reward_loss_shooting / (shoot_count+1)
            similarity_loss += self.similarity_loss(zs_query, zs_key).reshape(self.cfg.horizon-t, self.cfg.batch_size, -1).sum(dim=0)  # (batch_size, )
            # consistency_loss += self.consistency_loss(zs_query, zs_key)

        # Optimize model
        total_loss = self.cfg.similarity_coef * similarity_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4)
        weighted_loss = (total_loss * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm,
                                                   error_if_nonfinite=False)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        # pi_loss = self.analytic_update_pi(zs[0], action[0])
        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {'consistency_loss': float(similarity_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                'grad_norm': float(grad_norm),
                'mixture_coef': self.mixture_coef}
