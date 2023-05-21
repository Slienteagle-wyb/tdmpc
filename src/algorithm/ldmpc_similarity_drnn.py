import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.algos.utils.optim_factory import create_optimizer
import src.algorithm.helper as h
from src.models.gru_dyna import DGruDyna
from src.models.mask_generator import LowdimMaskGenerator
from gym.wrappers.normalize import RunningMeanStd
from src.models.conditional_unet1d import ConditionalUnet1D
from src.models.ema_model import EMAModel
from diffusers import DDPMScheduler
import einops


class LADSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._dynamics = DGruDyna(cfg)
        self._reward = h.mlp(cfg.hidden_dim, cfg.mlp_dim, 1)

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
            latents = self._encoder(obs)
        return latents

    def next(self, z, a, h_prev):
        z, hidden = self._dynamics(z, a, h_prev)
        reward_pred = self._reward(hidden)
        return z, hidden, reward_pred

    def init_hidden_state(self, batch_size, device):
        return self._dynamics.init_hidden_state(batch_size, device)

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)

    def pred_z(self, z):
        return self._predictor(z)


class DiffusionMpc(nn.Module):
    """Implementation of TD-MPC learning + inference."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ConditionalUnet1D(input_dim=cfg.action_dim,
                                       global_cond_dim=2*cfg.vis_horizon+cfg.latent_dim)
        self.ema_model = EMAModel(self.model, power=cfg.ema_power)
        self._noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.num_diffusion_iters,
                                              beta_schedule='squaredcos_cap_v2',
                                              clip_sample=True,
                                              prediction_type='epsilon')
        self.mask_generator = LowdimMaskGenerator(
            action_dim=cfg.action_dim, obs_dim=0,
            max_n_obs_steps=cfg.vis_horizon,
            fix_obs_steps=True,
            action_visible=False
        )

    def add_noise(self, x, noise, timesteps):
        return self._noise_scheduler.add_noise(x, noise, timesteps)

    def denoise(self, noisy_actions, timesteps, global_cond):
        return self.model(noisy_actions, timesteps, global_cond)


class TdMpcSimDssm:
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.mixture_coef = h.linear_schedule(self.cfg.regularization_schedule, 0)
        self.model = LADSSM(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.diffusion_policy = DiffusionMpc(cfg).cuda()

        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()
        self.reward_rms = RunningMeanStd()
        self.intrinsic_reward_rms = RunningMeanStd()
        self.plan_horizon = 1

        total_epochs = int(cfg.train_steps / cfg.episode_length)
        self._optim_initialize(total_epochs)

    def _optim_initialize(self, total_epochs):
        self.optim = create_optimizer(model=self.model, optim_id=self.cfg.optim_id,
                                      lr=self.cfg.lr)
        self.pi_optim = create_optimizer(model=self.model._pi, optim_id=self.cfg.optim_id,
                                         lr=self.cfg.pi_lr)
        self.diffusion_optim = create_optimizer(model=self.diffusion_policy,
                                                optim_id=self.cfg.optim_id,
                                                lr=self.cfg.diffusion_lr)

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict(),
                'diffusion_policy': self.diffusion_policy.state_dict(), }

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])
        self.diffusion_policy.load_state_dict(d['diffusion_policy'])

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
        """
        plan using the gradient of estimated value as guidance
        """
        pass

    def update_diffusion(self, zs_vis, actions):
        zs_vis = einops.rearrange(zs_vis, 't b e -> b t e')
        zs_cond = zs_vis.flatten(start_dim=1)  # current sequence of latent states as global condition
        actions = einops.rearrange(actions, 't b e -> b e t')
        # apply corruption to the sampled actions
        noise = torch.randn_like(actions, device=actions.device)
        timesteps = torch.randint(0, self.diffusion_policy.noise_scheduler.num_timesteps,
                                  (actions.shape[0],), device=actions.device).long()
        noisy_actions = self.diffusion_policy.add_noise(actions, noise, timesteps)
        pred_mask = self.diffusion_policy.mask_generator(noisy_actions.shape)
        loss_mask = ~pred_mask
        noisy_actions[pred_mask] = actions[pred_mask]
        # inverse diffusion process to predict the noise residual
        noise_pred = self.diffusion_policy.denoise(noisy_actions, timesteps, global_cond=zs_cond)

        loss = nn.functional.mse_loss(noise_pred, noise)
        loss = loss * loss_mask.type(loss.dtype)
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        self.diffusion_optim.zero_grad()
        loss.backward()
        self.diffusion_optim.step()
        self.diffusion_policy.ema_model.step(self.diffusion_policy)
        return loss.item()

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

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        # obs [batch, state_dim], actions [horizon+1, batch, act_dim]
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # current_epoch = int(step // self.cfg.episode_length)
        # if step % self.cfg.episode_length == 0:
        #     self.ensemble_lr_scheduler.step(current_epoch)
        # current_ensemble_lr = self.ensemble_lr_scheduler.get_epoch_values(current_epoch)[0]
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
        next_zs = self.model_target.h(self.aug(next_obses))
        online_next_zs = self.model.h(self.aug(next_obses))
        # input_zs = torch.cat([z.unsqueeze(0), online_next_zs], dim=0)
        zs = [z.detach()]  # obs embedding for policy learning
        beliefs = []

        similarity_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        # consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
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

        # Optimize ensemble models
        ensemble_loss = torch.zeros(1).cuda()
        if self.cfg.plan2expl:
            self.ensemble_optim.zero_grad(set_to_none=True)
            input_beliefs = torch.cat(beliefs, dim=0)  # (horizon*batch_size, hidden_dim)
            input_actions = action[:self.cfg.horizon].reshape(self.cfg.horizon * self.cfg.batch_size,
                                                              -1)  # (horizon*batch_size, act_dim)
            pred_targets = next_zs[:self.cfg.horizon].reshape(self.cfg.horizon * self.cfg.batch_size,
                                                              -1)  # (horizon*batch_size, state_dim)
            for i in range(len(self.forward_ensembles)):
                pred_means = self.forward_ensembles[i](input_beliefs, input_actions).mean
                ensemble_loss += h.mse(pred_means, pred_targets.detach()).clamp(max=1e4).mean()
            ensemble_loss.backward()
            self.ensemble_optim.step()

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
                # 'dreamed_q_loss': dreamed_q_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                'grad_norm': float(grad_norm),
                'ensemble_loss': float(ensemble_loss.mean().item()),
                # 'current_ensemble_lr': current_ensemble_lr,
                # 'intrinsic_batch_reward_mean': intrinsic_rewards.mean().item(),
                # 'current_explore_coef': explore_coef,
                'mixture_coef': self.mixture_coef}
