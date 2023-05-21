import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.algos.utils.optim_factory import create_optimizer
from rlpyt.ul.algos.utils.scheduler_factory import create_scheduler
import src.algorithm.helper as h
from src.models.gru_dyna import DGruDyna, OneStepDyna
from gym.wrappers.normalize import RunningMeanStd


class DSSM(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._dynamics = DGruDyna(cfg)
        self._reward = h.mlp(cfg.hidden_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
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
            # obs = h.symlog(obs)
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
            return h.TruncatedNormal(mu, std).sample(clip=0.3)[0]
        return mu

    def pi_log_prob(self, z, std=0.05):
        mu = torch.tanh(self._pi(z))
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std).sample(clip=0.3)

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)

    def pred_z(self, z):
        return self._predictor(z)


class CqMpcSimDssm:
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, 0)
        self.model = DSSM(cfg).cuda()
        self.model_target = deepcopy(self.model)

        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()
        self.reward_rms = RunningMeanStd()
        self.intrinsic_reward_rms = RunningMeanStd()
        self.plan_horizon = 1
        self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=cfg.device)

        total_epochs = int(cfg.train_steps / cfg.episode_length)
        self._optim_initialize(total_epochs)

    def _optim_initialize(self, total_epochs):
        self.optim = create_optimizer(model=self.model, optim_id=self.cfg.optim_id,
                                      lr=self.cfg.lr)
        self.pi_optim = create_optimizer(model=self.model._pi, optim_id=self.cfg.optim_id,
                                         lr=self.cfg.pi_lr)
        self.alpha_prime_optim = torch.optim.AdamW([self.log_alpha_prime], lr=self.cfg.alpha_lr)

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict()}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])

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

    @torch.no_grad()
    def plan(self, obs, hidden, eval_mode=False, step=None, t0=True):
        intrinsic_reward_mean, reward_mean = 0, 0
        plan_metrics = {'intrinsic_reward_mean': 0.0, 'external_reward_mean': 0.0, 'current_std': 0.0}

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1), None, plan_metrics

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        if horizon != self.plan_horizon and t0:
            self.plan_horizon = horizon
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, step)
        num_pi_trajs = int(self.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(self.plan_horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            hidden_pi = hidden.repeat(num_pi_trajs, 1)
            for t in range(self.plan_horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, hidden_pi, _ = self.model.next(z, pi_actions[t], hidden_pi)

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples + num_pi_trajs, 1)
        hidden_plan = hidden.repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(self.plan_horizon, self.cfg.action_dim, device=self.device)
        std = 2 * torch.ones(self.plan_horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            # parameterized action of mpc
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(self.plan_horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device),
                                  -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            if self.cfg.plan2expl and not eval_mode and i < 6:
                value, rewards_expl, intrinsic_reward_mean, reward_mean = self.plan2explore(z, actions, hidden_plan)
                value += rewards_expl
                elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            else:
                value, reward_mean = self.estimate_value(z, actions, hidden_plan)
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
        z, hidden, _ = self.model.next(z[0:1], a.unsqueeze(0), hidden)

        plan_metrics.update({'intrinsic_reward_mean': intrinsic_reward_mean,
                             'external_reward_mean': reward_mean,
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

    def cql_loss(self, z, next_z, q1, q2):
        """
        update the q function using the conservative regularize
        """
        # rollout the model to get the next latent state
        batch_size = z.shape[0]
        num_repeat = self.cfg.cql_n_actions
        z = z.unsqueeze(1).repeat(1, num_repeat, 1).view(batch_size * num_repeat, -1)
        next_z = next_z.unsqueeze(1).repeat(1, num_repeat, 1).view(batch_size * num_repeat, -1)
        # prepare action query from random uniform dist and pi policy
        with torch.no_grad():
            cql_rand_actions = torch.FloatTensor(int(batch_size*num_repeat), self.cfg.action_dim).uniform_(-1, 1).to(self.device)
            cql_pi_actions, cql_pi_log_prob = self.model.pi_log_prob(z, std=self.cfg.min_std)
            cql_next_pi_actions, cql_next_pi_log_prob = self.model.pi_log_prob(next_z, std=self.cfg.min_std)
        # calculate the estimated q values
        cql_q1_rand, cql_q2_rand = self.model.Q(z, cql_rand_actions.detach())
        cql_q1_rand, cql_q2_rand = cql_q1_rand.reshape(batch_size, num_repeat, -1), cql_q2_rand.reshape(batch_size, num_repeat, -1)
        cql_q1_pi_actions, cql_q2_pi_actions = self.model.Q(z, cql_pi_actions.detach())
        cql_q1_pi_actions, cql_q2_pi_actions = cql_q1_pi_actions.reshape(batch_size, num_repeat, -1), cql_q2_pi_actions.reshape(batch_size, num_repeat, -1)
        cql_q1_next_pi_actions, cql_q2_next_pi_actions = self.model.Q(z, cql_next_pi_actions.detach())
        cql_q1_next_pi_actions, cql_q2_next_pi_actions = cql_q1_next_pi_actions.reshape(batch_size, num_repeat, -1), cql_q2_next_pi_actions.reshape(batch_size, num_repeat, -1)

        cql_q1_cats = torch.cat([cql_q1_rand, cql_q1_pi_actions, cql_q1_next_pi_actions], dim=1)
        cql_q2_cats = torch.cat([cql_q2_rand, cql_q2_pi_actions, cql_q2_next_pi_actions], dim=1)

        cql_q1_ood = torch.logsumexp(cql_q1_cats / self.cfg.cql_tmp, dim=1).mean() * self.cfg.cql_min_q_weight * self.cfg.cql_tmp
        cql_q2_ood = torch.logsumexp(cql_q2_cats / self.cfg.cql_tmp, dim=1).mean() * self.cfg.cql_min_q_weight * self.cfg.cql_tmp

        cql_q1_diff = (cql_q1_ood - q1.mean()) * self.cfg.cql_min_q_weight
        cql_q2_diff = (cql_q2_ood - q2.mean()) * self.cfg.cql_min_q_weight

        alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime), min=0.0, max=1000000.0)
        cql_min_q1_loss = alpha_prime * (cql_q1_diff - self.cfg.cql_target_action_gap)
        cql_min_q2_loss = alpha_prime * (cql_q2_diff - self.cfg.cql_target_action_gap)

        cql_min_loss = cql_min_q1_loss + cql_min_q2_loss
        alpha_prime_loss = 0.5 * (-cql_min_q1_loss - cql_min_q2_loss)

        return cql_min_loss, alpha_prime_loss

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
        z = self.model.h(obs)
        next_zs = self.model_target.h(next_obses)
        online_next_zs = self.model.h(next_obses)
        # input_zs = torch.cat([z.unsqueeze(0), online_next_zs], dim=0)
        zs = [z.detach()]  # obs embedding for policy learning
        beliefs = []

        similarity_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        alpha_prime_total_loss = 0
        hidden = self.model.init_hidden_state(z.shape[0], z.device)
        for t in range(self.cfg.horizon):
            # Predictions
            rho = (self.cfg.rho ** t)
            zs_query, zs_key = [], []
            reward_loss_shooting = 0
            Q1, Q2 = self.model.Q(z, action[t])
            cql_loss, alpha_prime_loss = self.cql_loss(z, online_next_zs[t], Q1, Q2)
            z, hidden, reward_pred = self.model.next(z, action[t], hidden)
            reward_loss_shooting += h.mse(reward_pred, reward[t])
            with torch.no_grad():
                td_target = self._td_target(online_next_zs[t], reward[t])
            bellman_error = h.mse(Q1, td_target) + h.mse(Q2, td_target)
            value_loss += (bellman_error + cql_loss) * 0.5
            priority_loss += h.l1(Q1, td_target) + h.l1(Q2, td_target)

            zs_query.append(z)
            zs_key.append(next_zs[t].detach())
            zs.append(z.detach())
            beliefs.append(hidden)  # will be detached internally

            # conduct overshooting for the next states embedding
            shoot_hidden = torch.clone(hidden)
            shoot_z = torch.clone(z)
            shoot_count = 0
            for j in range(t+1, self.cfg.horizon):
                shoot_count += 1
                shoot_z, shoot_hidden, reward_pred = self.model.next(shoot_z, action[j], shoot_hidden)
                zs_query.append(shoot_z)
                zs_key.append(next_zs[j].detach())
                reward_loss_shooting += h.mse(reward_pred, reward[j])
            reward_loss += reward_loss_shooting / (shoot_count+1)
            similarity_loss += self.similarity_loss(zs_query, zs_key).reshape(self.cfg.horizon-t, self.cfg.batch_size, -1).sum(dim=0)  # (batch_size, )

        # Optimize model
        self.alpha_prime_optim.zero_grad()
        alpha_prime_loss.backward(retain_graph=True)
        self.alpha_prime_optim.step()

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
                'aplha_prime_loss': float(alpha_prime_loss.item()),
                'grad_norm': float(grad_norm),
                'mixture_coef': self.mixture_coef}
