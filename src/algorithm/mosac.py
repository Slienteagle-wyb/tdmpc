import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from algorithm.helper import Episode
import algorithm.helper as h


class SofTold(nn.Module):
    def __init__(self, cfg):
        super(SofTold, self).__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = h.SoftActor(cfg)
        self._soft_q1, self._soft_q2 = h.q(cfg), h.q(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, self._soft_q1, self._soft_q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        for m in [self._soft_q1, self._soft_q2]:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        return self._encoder(obs)

    def next(self, z, a):
        x = torch.cat([z, a], dim=1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z):
        action, _, _ = self._pi.get_action(z)
        return action

    def get_action(self, z):
        return self._pi.get_action(z)

    def Q(self, z, a):
        x = torch.cat([z, a], dim=1)
        return self._soft_q1(x), self._soft_q2(x)

    def pred_z(self, z):
        return self._predictor(z)


class MoSac:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = SofTold(cfg).to(cfg.device)
        self.model_target = deepcopy(self.model)
        self.target_entropy = -torch.prod(torch.tensor(cfg.action_dim).to(cfg.device)).item()
        self.log_temp = torch.zeros(1, requires_grad=True, device=cfg.device)
        self.temp = torch.exp(self.log_temp).item()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=cfg.pi_lr)
        self.temp_optim = torch.optim.Adam([self.log_temp], lr=cfg.lr)

        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()

    def static_dict(self):
        return {
            'model': self.model.state_dict(),
            'model_target': self.model_target.state_dict()
        }

    def save(self, fp):
        torch.save(self.static_dict(), fp)

    def load(self, fp):
        state_dict = torch.load(fp)
        self.model.load_state_dict(state_dict['model'])
        self.model_target.load_state_dict(state_dict['model_target'])

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z)))
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            # parameterized action of mpc
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device),
                                  -1, 1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)  # forward shoot plus q_value
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
        return a

    @torch.no_grad()
    def latent_plan(self, z, eval_mode=False, step=None, t0=True):
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
        # Sample policy trajectories
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        # Initialize state and parameters
        z_trace = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            # parameterized action of mpc
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device),
                                  -1, 1)

            # Compute elite actions
            value = self.estimate_value(z_trace, actions, horizon).nan_to_num_(0)  # forward shoot plus q_value
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
        return a

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.cfg.discount * \
            torch.min(*self.model_target.Q(next_z, self.model.pi(next_z)))
        return td_target

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a, log_pi, _ = self.model.get_action(z)
            q = torch.min(*self.model.Q(z, a))
            pi_loss += ((self.temp * log_pi) - q).mean() * (self.cfg.rho ** t)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()

        with torch.no_grad():
            _, log_pi, _ = self.model.get_action(zs[0])
        temp_loss = (-self.log_temp * (log_pi + self.target_entropy)).mean()
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        self.temp = torch.exp(self.log_temp).item()

        self.model.track_q_grad(True)
        return pi_loss.item(), temp_loss.item()

    # recurrent objective introduced by td-mpc
    def recurrent_loss(self, buffer, overshoot_horizon):
        obs, next_obses, actions, rewards, idxs, weights = buffer.sample()
        z = self.model.h(self.aug(obs))
        zs = [z.detach()]
        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0

        for t in range(overshoot_horizon + 1):
            q1, q2 = self.model.Q(z, actions[t])
            z, reward_pred = self.model.next(z, actions[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                td_target = self._td_target(next_obs, rewards[t])
                next_z = self.model_target.h(next_obs)
            zs.append(z.detach())
            # calculate cul loss through time
            rho = self.cfg.rho ** t
            consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)  # (batch, 1)
            reward_loss += rho * h.mse(reward_pred, rewards[t])  # (batch, 1)
            value_loss += rho * (h.mse(q1, td_target) + h.mse(q2, td_target))  # (batch, 1)
            priority_loss += rho * (h.l1(q1, td_target) + h.l1(q2, td_target))
        # buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())
        return zs, consistency_loss, reward_loss, value_loss, weights

    def update_interval(self, env_buffer, plan_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # model loss using datas sampled by planning
        zs_plan, consistency_loss_plan, reward_loss_plan, value_loss_plan, weights_plan = \
            self.recurrent_loss(plan_buffer, self.cfg.horizon)
        dyna_model_loss = self.cfg.consistency_coef * consistency_loss_plan.clamp(max=1e4) + \
            self.cfg.reward_coef * reward_loss_plan.clamp(
            max=1e4) + self.cfg.value_coef * value_loss_plan.clamp(max=1e4)
        weighted_dyna_model_loss = (weights_plan * dyna_model_loss).mean()
        self.optim.zero_grad(set_to_none=True)
        weighted_dyna_model_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_dyna_model_loss.backward()
        grad_norm_plan = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm,
                                                        error_if_nonfinite=False)
        self.optim.step()
        pi_loss_plan, temp_loss_plan = self.update_pi(zs_plan)

        # model loss using datas sampled by pi
        zs_env, consistency_loss_env, reward_loss_env, value_loss_env, weights_env = \
            self.recurrent_loss(env_buffer, self.cfg.env_horizon)
        env_model_loss = self.cfg.consistency_coef * consistency_loss_env.clamp(max=1e4) + \
            self.cfg.reward_coef * reward_loss_env.clamp(max=1e4) + self.cfg.value_coef * value_loss_env.clamp(max=1e4)
        weighted_env_model_loss = (weights_env * env_model_loss).mean()
        self.optim.zero_grad(set_to_none=True)
        weighted_env_model_loss.backward()
        grad_norm_env = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm,
                                                       error_if_nonfinite=False)
        self.optim.step()
        if self.cfg.env_horizon > 0:
            pi_loss, temp_loss = self.update_pi(zs_env)
        else:
            pi_loss, temp_loss = self.update_pi([zs_env[0]])

        weighted_total_loss = weighted_dyna_model_loss + weighted_env_model_loss
        # Update policy + target network
        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)
        self.model.eval()
        return {'consistency_loss_plan': float(consistency_loss_plan.mean().item()),
                'consistency_loss_env': float(consistency_loss_env.mean().item()),
                'reward_loss_plan': float(reward_loss_plan.mean().item()),
                'reward_loss_env': float(reward_loss_env.mean().item()),
                'value_loss': float(value_loss_plan.mean().item()),
                'value_loss_plan': float(value_loss_plan.mean().item()),
                'pi_loss': pi_loss,
                'pi_loss_model': pi_loss_plan,
                'temp_loss_plan': temp_loss_plan,
                'temp_loss': temp_loss,
                'cur_temp': float(self.temp),
                'dyna_model_loss': float(dyna_model_loss.mean().item()),
                'env_model_loss': float(env_model_loss.mean().item()),
                'weighted_loss': float(weighted_total_loss.item()),
                'grad_norm_plan': float(grad_norm_plan),
                'grad_norm_env': float(grad_norm_env),
                'explore_std': float(self.std)}
