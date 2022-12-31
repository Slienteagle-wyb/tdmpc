import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.algos.utils.optim_factory import create_optimizer
from rlpyt.ul.algos.utils.scheduler_factory import create_scheduler
import algorithm.helper as h
from gym.wrappers.normalize import RunningMeanStd


class TOLD(nn.Module):
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

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
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


class TDMPCSIM():
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, 0)
        self.model = TOLD(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()
        self.reward_rms = RunningMeanStd()
        self.plan_horizon = 1

        total_epochs = int(cfg.train_steps / cfg.episode_length)
        self._optim_initialize(total_epochs)

    def _optim_initialize(self, total_epochs):
        self.optim = create_optimizer(model=self.model, optim_id=self.cfg.optim_id,
                                      lr=self.cfg.lr)
        self.pi_optim = create_optimizer(model=self.model._pi, optim_id=self.cfg.optim_id,
                                         lr=self.cfg.pi_lr)
        # self.lr_scheduler, _ = create_scheduler(optimizer=self.optim, num_epochs=total_epochs,
        #                                         sched_kwargs=self.cfg.sched_kwargs)
        # self.pi_lr_scheduler, _ = create_scheduler(optimizer=self.pi_optim, num_epochs=total_epochs,
        #                                            sched_kwargs=self.cfg.sched_kwargs)
        # self.optim.zero_grad()
        # self.optim.step()
        # self.pi_optim.zero_grad()
        # self.pi_optim.step()

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
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        """
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

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
            for t in range(self.plan_horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples + num_pi_trajs, 1)
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
            value = self.estimate_value(z, actions, self.plan_horizon).nan_to_num_(0)  # forward shoot plus q_value
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

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z, a))
            pi_loss += -Q.mean() * (self.cfg.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.cfg.discount * \
                    torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
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
        for t in range(T):
            z, _ = self.model.next(z, actions[t])
            zs_latent.append(z)
        zs_latent = torch.stack(zs_latent, dim=0)
        return zs_latent

    @torch.no_grad()
    def intrinsic_rewards(self, obs, next_obses, actions):
        zs_target = self.model_target.h(self.aug(next_obses))
        z_traj = self.model.h(self.aug(torch.cat([obs.unsqueeze(0), next_obses], dim=0)))
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

    # perform latent rollout to replace the interactive transition from environment
    @torch.no_grad()
    def dream(self, next_obses, dream_horizon):
        # feature cat (batch*traj_horizon, latent_dim)
        zs = self.model.h(next_obses.reshape((self.cfg.horizon+1) * self.cfg.batch_size, -1))
        zs_dream = [zs]
        actions, rewards = [], []
        for i in range(dream_horizon):
            action = self.model.pi(zs, self.cfg.min_std)
            zs, reward_pred = self.model.next(zs, action)
            zs_dream.append(zs)
            actions.append(action)
            rewards.append(reward_pred)
        zs_dream = torch.stack(zs_dream, dim=0)  # (dream_horizon+1, b*(h+1), latent_dim)
        actions = torch.stack(actions, dim=0)  # (dream_horizon, b*(h+1), act_dim)
        rewards = torch.stack(rewards, dim=0)  # (dream_horizon, b*(h+1), 1)

        return zs_dream, actions, rewards

    def dream_loss(self, next_obses, dream_horizon):
        zs_dream, pi_actions, rewards_pred = self.dream(next_obses, dream_horizon)
        value_loss = 0
        for t in range(dream_horizon):
            rho = self.cfg.rho ** t
            q1, q2 = self.model.Q(zs_dream[t], pi_actions[t])
            td_target = self._td_target_latent(zs_dream[t+1], rewards_pred[t])
            value_loss += rho * (h.mse(q1, td_target) + h.mse(q2, td_target))  # (b*h, 1)
        return zs_dream, value_loss

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        # obs [batch, state_dim], actions [horizon+1, batch, act_dim]
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # current_epoch = int(step // self.cfg.episode_length)
        # if step % self.cfg.episode_length == 0:
        #     self.lr_scheduler.step(current_epoch)
        #     self.pi_lr_scheduler.step(current_epoch)
        # current_lr = self.lr_scheduler.get_epoch_values(current_epoch)[0]
        self.optim.zero_grad(set_to_none=True)

        # calculate intrinsic reward for exploration
        # explore_coef = h.linear_schedule(self.cfg.explore_schedule, step)
        # intrinsic_rewards = self.intrinsic_rewards(obs, next_obses, action)
        # reward_pi = explore_coef * intrinsic_rewards + reward

        # Representation
        z = self.model.h(self.aug(obs))
        zs = [z.detach()]
        zs_query, zs_key = [], []

        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):
            # Predictions
            Q1, Q2 = self.model.Q(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_z = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t])
            zs_query.append(z)
            zs_key.append(next_z.detach())
            zs.append(z.detach())

            # Losses
            rho = (self.cfg.rho ** t)
            # consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
            reward_loss += rho * h.mse(reward_pred, reward[t])
            value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
            priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

        similarity_loss = self.similarity_loss(zs_query, zs_key)
        similarity_loss = similarity_loss.reshape(self.cfg.horizon, self.cfg.batch_size, -1).mean(dim=0)  # (batch_size, )

        # Optimize model
        total_loss = self.cfg.similarity_coef * similarity_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4)
        # total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
        #              self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
        #              self.cfg.value_coef * value_loss.clamp(max=1e4)
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
                # 'intrinsic_batch_reward_mean': intrinsic_rewards.mean().item(),
                # 'current_explore_coef': explore_coef,
                'mixture_coef': self.mixture_coef}
