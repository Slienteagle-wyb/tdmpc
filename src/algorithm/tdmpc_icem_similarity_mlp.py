import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorednoise
from copy import deepcopy
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.algos.utils.optim_factory import create_optimizer
import src.algorithm.helper as h
from gym.wrappers.normalize import RunningMeanStd


class DSSM(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._dynamics = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
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
            obs = obs.view(T * B, *img_shape)
            latent_feature, _ = self._encoder(obs)
            latents = restore_leading_dims(latent_feature, lead_dim, T, B)
        else:
            # obs = h.symlog(obs)
            latents = self._encoder(obs)
        return latents

    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

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


class TdICemSimMlp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, 0)
        self.model = DSSM(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.reward_rms = RunningMeanStd()

        self.model.eval()
        self.model_target.eval()

        self.plan_horizon = 1
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
    def estimate_value(self, z, actions):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.plan_horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G.nan_to_num_(0), float(reward.mean().item())

    def sample_action_sequence(self, num_samples, mean, std):
        if self.cfg.noise_beta > 0.:
            noise = colorednoise.powerlaw_psd_gaussian(self.cfg.noise_beta,
                                                       size=(num_samples,
                                                             self.cfg.action_dim,
                                                             self.plan_horizon))
            noise = torch.from_numpy(noise).float().to(self.device).permute(2, 0, 1)
        else:
            noise = torch.randn(self.plan_horizon, num_samples,
                                self.cfg.action_dim, device=self.device)
        actions_sampled = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * noise, -1, 1)
        return actions_sampled

    def sample_mix_action_sequence(self, num_samples, mean, std):
        quotient = num_samples // 3
        remainder = num_samples % 3
        if remainder == 0:
            sample_list = [quotient, quotient, quotient]
        elif remainder == 1:
            sample_list = [quotient, quotient + 1, quotient]
        else:
            sample_list = [quotient, quotient + 1, quotient + 1]
        noise_white = torch.randn(self.plan_horizon, sample_list[0], self.cfg.action_dim, device=self.device)
        noise_pink = colorednoise.powerlaw_psd_gaussian(1.0, size=(sample_list[1], self.cfg.action_dim,
                                                                   self.cfg.horizon))
        noise_pink = torch.from_numpy(noise_pink).float().to(self.device).permute(2, 0, 1)[:self.plan_horizon]
        noise_brown = colorednoise.powerlaw_psd_gaussian(2.5, size=(sample_list[2], self.cfg.action_dim,
                                                                    self.cfg.horizon))
        noise_brown = torch.from_numpy(noise_brown).float().to(self.device).permute(2, 0, 1)[:self.plan_horizon]
        noise = torch.cat([noise_white, noise_pink, noise_brown], dim=1)
        actions_sampled = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * noise, -1, 1)
        return actions_sampled

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        intrinsic_reward_mean, reward_mean = 0, 0
        extend_horizon = False
        plan_metrics = {'external_reward_mean': 0.0, 'current_std': 0.0}
        # initialize seeds steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32,
                               device=self.device).uniform_(-1, 1), plan_metrics
        # # prepare params at the start of rollout
        # schedule the horizon and mixture coefficient of policy
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        if horizon != self.plan_horizon and t0:
            self.plan_horizon = horizon
            extend_horizon = True
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, step)
        # initialize mean and std
        mean = torch.zeros(self.plan_horizon, self.cfg.action_dim, device=self.device)
        std = 0.5 * torch.ones(self.plan_horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]
        # set up the num of rollout trajectory
        num_samples = self.cfg.num_samples
        num_pi_trajs = int(self.mixture_coef * num_samples)
        assert num_pi_trajs > 0
        # rollout the policy to get pi action sequences
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.h(obs)
        pi_actions = torch.empty(self.plan_horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
        zs_pi = z.repeat(num_pi_trajs, 1)
        # if not eval_mode:
        #     zs_pi = self.aug(zs_pi)
        for t in range(self.plan_horizon):
            pi_actions[t] = self.model.pi(zs_pi, self.cfg.min_std)
            zs_pi, _ = self.model.next(zs_pi, pi_actions[t])
            # if not eval_mode:
            #     zs_pi = self.aug(zs_pi)

        # Iterate iCEM
        for i in range(self.cfg.iterations):
            if i > 0:
                num_samples = max(2 * self.cfg.num_elites, int(num_samples / self.cfg.factor_decrease_num))
                num_pi_trajs = int(self.mixture_coef * num_samples)
                assert num_pi_trajs > 0
            if self.cfg.fraction_elites_reused > 0 and hasattr(self, '_elite_actions'):
                num_elite_trajs = int(self.cfg.fraction_elites_reused * self.cfg.num_elites)
                num_trajs = num_samples + num_pi_trajs + num_elite_trajs
            else:
                num_elite_trajs = 0
                num_trajs = num_samples + num_pi_trajs + num_elite_trajs
            zs_plan = z.repeat(num_trajs, 1)
            # sample actions from the current distribution
            actions_sampled = self.sample_mix_action_sequence(num_samples, mean, std)
            if i == self.cfg.iterations - 1:
                actions_sampled[:, 0] = mean  # use the mean of the last iteration (icem_best-a)

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
            value, reward_mean = self.estimate_value(zs_plan, actions)
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
        z, _ = self.model.next(z[0:1], a.unsqueeze(0))

        plan_metrics.update({'external_reward_mean': reward_mean,
                             'current_std': std.mean().item()})

        return a, plan_metrics

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            # Q = torch.min(*self.model.Q(z, a)) * (self.cfg.rho ** t)
            Q = torch.min(*self.model.Q(z, a))
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
        seg_len = z_pred.shape[0]
        z_pred = z_pred.reshape(seg_len*self.cfg.batch_size, -1)
        z_target = z_target.reshape(seg_len*self.cfg.batch_size, -1)
        # seg_len = len(z_pred)
        # z_pred = torch.cat(z_pred, dim=0)  # (cfg.horizon*batch_size, latent_dim)
        # z_target = torch.cat(z_target, dim=0)
        seg_loss = torch.mean(h.mse(z_pred, z_target), dim=1, keepdim=True)
        seg_loss_mean = seg_loss.reshape(seg_len, self.cfg.batch_size, -1).mean(dim=0)
        return seg_loss_mean

    @torch.no_grad()
    def model_rollout(self, z, actions):
        T, B = actions.shape[:2]
        zs_latent = []
        for t in range(T):
            z, _ = self.model.next(z, actions[t])
            zs_latent.append(z)
        zs_latent = torch.stack(zs_latent, dim=0)
        return zs_latent

    @torch.no_grad()
    def intrinsic_rewards(self, obs, next_obses, actions):
        zs_target = self.model_target.h(next_obses)
        z_traj = self.model.h(torch.cat([obs.unsqueeze(0), next_obses], dim=0))
        model_uncertainty = torch.zeros((self.cfg.horizon+1, self.cfg.horizon+1, self.cfg.batch_size, 1), requires_grad=False).cuda()
        for t in range(0, z_traj.shape[0]-1):
            zs_latent = self.model_rollout(z_traj[t], actions[t:t+1])
            zs_pred = self.model.pred_z(zs_latent.squeeze(0))
            zs_pred = F.normalize(zs_pred.unsqueeze(0), dim=-1, p=2)
            target = F.normalize(zs_target[t:t+1], dim=-1, p=2)
            partial_pred_loss = 2.0 - 2.0 * (zs_pred * target).sum(dim=-1, keepdim=True)   # (t:t+1, b, 1)
            model_uncertainty[t][t:t+1] = partial_pred_loss.detach_()

        intrinsic_reward = torch.sum(model_uncertainty, dim=0).cpu().data.numpy()  # (t, b, 1)
        reward_mean = np.mean(intrinsic_reward)
        reward_std = np.std(intrinsic_reward)
        self.reward_rms.update_from_moments(reward_mean, reward_std ** 2, 1)
        intrinsic_reward /= np.sqrt(self.reward_rms.var)
        reward_threshold = self.reward_rms.mean / np.sqrt(self.reward_rms.var)
        intrinsic_reward = np.maximum(intrinsic_reward - reward_threshold, 0)
        intrinsic_reward = torch.from_numpy(intrinsic_reward).cuda()
        return intrinsic_reward.detach_()

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
        zs = [z.detach()]  # obs embedding for policy learning

        # calculate intrinsic reward for exploration
        explore_coef = h.linear_schedule(self.cfg.explore_schedule, step)
        intrinsic_rewards = self.intrinsic_rewards(obs, next_obses, action)
        reward_pi = explore_coef * intrinsic_rewards + reward

        similarity_loss, reward_loss, value_loss, priority_loss, consistency_loss = 0, 0, 0, 0, 0
        zs_query, zs_key = [], []

        # calculate next target q for td_lambda
        # next_q_values, outputs = [], []
        # rewards = reward_pi[:self.cfg.horizon]
        # discounts = torch.ones_like(rewards) * self.cfg.discount
        # for t, next_z in enumerate(online_next_zs[:self.cfg.horizon]):
        #     with torch.no_grad():
        #         next_q_value = torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
        #     next_q_values.append(next_q_value)
        # next_q_values = torch.stack(next_q_values, dim=0)
        # last = next_q_values[-1]
        # inputs = rewards + discounts * next_q_values * (1 - self.cfg.td_lambda)
        # for index in reversed(range(next_q_values.shape[0])):
        #     last = inputs[index] + discounts[index] * last * self.cfg.td_lambda
        #     outputs.append(last)
        # td_lambda_targets = torch.stack(list(reversed(outputs)), dim=0)

        for t in range(self.cfg.horizon):
            rho = (self.cfg.rho ** t)
            Q1, Q2 = self.model.Q(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                td_target = self._td_target(online_next_zs[t], reward_pi[t])
            # td_target = td_lambda_targets[t]
            reward_loss += (h.mse(reward_pred, reward[t])) * rho
            value_loss += (h.mse(Q1, td_target) + h.mse(Q2, td_target)) * rho
            priority_loss += (h.l1(Q1, td_target) + h.l1(Q2, td_target)) * rho

            zs_query.append(z)
            zs_key.append(next_zs[t].detach())
            zs.append(z.detach())

        similarity_loss += self.similarity_loss(zs_query, zs_key).reshape(self.cfg.horizon, self.cfg.batch_size, -1).sum(dim=0)  # (batch_size, )

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
                'intrinsic_batch_reward_mean': intrinsic_rewards.mean().item(),
                'current_explore_coef': explore_coef,
                'mixture_coef': self.mixture_coef}
