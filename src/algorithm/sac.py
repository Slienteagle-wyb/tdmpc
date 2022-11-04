import numpy
import torch
import torch.nn as nn
import algorithm.helper as h
import torch.nn.functional as F


# actor critic agent of Sac algorithm
class Agent(nn.Module):
    def __init__(self, cfg):
        super(Agent, self).__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._soft_q1, self._soft_q2 = h.soft_q(cfg), h.soft_q(cfg)
        self._soft_q1_target, self._soft_q2_target = h.soft_q(cfg), h.soft_q(cfg)
        self._pi = h.SoftActor(cfg)
        self.apply(h.orthogonal_init)
        self._soft_q1_target.load_state_dict(self._soft_q1.state_dict())
        self._soft_q2_target.load_state_dict(self._soft_q2.state_dict())

    def pi(self, z):
        return self._pi.get_action(z)

    def double_q(self, z, a):
        x = torch.cat([z, a], dim=1)
        return self._soft_q1(x), self._soft_q2(x)

    def double_q_target(self, z, a):
        x = torch.cat([z, a], dim=1)
        return self._soft_q1_target(x), self._soft_q2_target(x)

    def update_target(self):
        h.ema(self._soft_q1, self._soft_q1_target, self.cfg.tau)
        h.ema(self._soft_q2, self._soft_q2_target, self.cfg.tau)

    @property
    def q1_parameters(self):
        return self._soft_q1.parameters()

    @property
    def q2_parameters(self):
        return self._soft_q2.parameters()

    @property
    def pi_parameters(self):
        return self._pi.parameters()


class Sac:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        self.ac = Agent(cfg).cuda()
        self.q_optim = torch.optim.Adam(list(self.ac.q1_parameters) + list(self.ac.q2_parameters),
                                        lr=cfg.q_lr)
        self.pi_optim = torch.optim.Adam(self.ac.pi_parameters, lr=cfg.pi_lr)
        # automatic temperature tuning
        self.target_entropy = -torch.prod(torch.Tensor(cfg.action_dim).to(cfg.device)).item()
        self.log_temp = torch.zeros(1, requires_grad=True, device=cfg.device)
        self.temp = torch.exp(self.log_temp).item()
        self.temp_optim = torch.optim.Adam([self.log_temp], lr=cfg.q_lr)

        self.ac.eval()

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'ac': self.ac.state_dict()}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.ac.load_state_dict(d['ac'])

    def update(self, replay_buffer, step):
        obs, next_obs, action, reward, idx, weights = replay_buffer.sample()
        self.ac.train()
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.ac.pi(next_obs[0])
            next_target_qf1, next_target_qf2 = self.ac.double_q_target(next_obs[0], next_state_action)
            min_next_target_q = torch.min(next_target_qf1, next_target_qf2) - self.temp * next_state_log_pi
            td_target = reward[0].flatten() + self.cfg.gamma * min_next_target_q.view(-1)

        q1, q2 = self.ac.double_q(obs, action[0])
        q1_loss = F.mse_loss(q1.view(-1), td_target)
        q2_loss = F.mse_loss(q2.view(-1), td_target)
        q_loss = q1_loss + q2_loss
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # update policy
        action_pi, log_pi, _ = self.ac.pi(obs)
        q1_pi, q2_pi = self.ac.double_q(obs, action_pi)
        q_pi = torch.min(q1_pi.view(-1), q2_pi.view(-1))
        pi_loss = ((self.temp * log_pi) - q_pi).mean()
        # update policy
        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()
        # update temperature parameter
        with torch.no_grad():
            _, log_pi, _ = self.ac.pi(obs)
        temp_loss = (-self.log_temp * (log_pi + self.target_entropy)).mean()

        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        self.temp = self.log_temp.exp().item()

        # update the target network
        if step % self.cfg.update_freq == 0:
            self.ac.update_target()

        self.ac.eval()
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q_loss': q_loss.item(),
            'pi_loss': pi_loss.item(),
            'temp_loss': temp_loss.item()
        }
