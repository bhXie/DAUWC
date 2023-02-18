import copy
import torch

import os
from actor import Actor
from critic import Critic, ValueCritic, EnsembleCritic


def loss(diff, v=0.8):
    weight = torch.where(diff > 0, v, (1 - v))
    return weight * (diff**2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        v,
        discount,
        tau,
        temperature,
    ):

        self.actor1 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr)

		self.actor2 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor_lr)

        self.critic = EnsembleCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = ValueCritic(state_dim, 256, 3).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau 
        self.temperature = temperature

        self.total_it = 0
        self.v = v

    def update_v(self, states, logger=None):
        with torch.no_grad():
            a1, a2 = self.actor.get_action(states)
            q11, q12, q13, q14, std1 = self.critic_target(states, a1)
            q21, q22, q23, q24, std2 = self.critic_target(states, a2)
            all_qs1 = torch.cat([q11.unsqueeze(0), q12.unsqueeze(0),q13.unsqueeze(0), q14.unsqueeze(0)], 0)
            all_qs2 = torch.cat([q21.unsqueeze(0), q22.unsqueeze(0),q23.unsqueeze(0), q24.unsqueeze(0)], 0)
            vmax = torch.maximum(all_qs1.min(0)[0], all_qs2.min(0)[0]).detach()
            vmin = torch.minimum(all_qs1.min(0)[0], all_qs2.min(0)[0]).detach()
            vcal = (1- self.v ) * vmax + self.v * vmin

        v = self.value(states)
        value_loss = ((vcal - v)**2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        logger.log('train/value_loss', value_loss, self.total_it)
        logger.log('train/v', v.mean(), self.total_it)

    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():

            next_v = self.value(next_states)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        q1, q2, q3, q4, std = self.critic(states, actions)
        target_q = target_q - std
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q )**2 + (q3 - target_q )**2 + (q4 - target_q)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logger.log('train/critic_loss', critic_loss, self.total_it)
        logger.log('train/q1', q1.mean(), self.total_it)
        logger.log('train/q2', q2.mean(), self.total_it)

    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor(self, states, actions, logger=None):
        with torch.no_grad():
            v = self.value(states)
            q1, q2, q3, q4, std = self.critic(states, actions)
            all_qs = torch.cat([q1.unsqueeze(0), q2.unsqueeze(0),q3.unsqueeze(0), q4.unsqueeze(0)], 0)
            q = all_qs.min(0)[0]
            
            exp_a = torch.exp((q - v) * self.temperature) 
            exp_a = torch.clamp(exp_a, max=100.0).squeeze(-1).detach()

        mu1, mu2 = self.actor(states)
        actor_loss = ((exp_a.unsqueeze(-1) * ((mu1 - actions)**2)) + (exp_a.unsqueeze(-1) * ((mu2 - actions)**2))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        logger.log('train/actor_loss', actor_loss, self.total_it)
        logger.log('train/adv', (q - v).mean(), self.total_it)

    # keep actor 1
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        a1,a2 = self.actor.get_action(state)
        return a1.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Update
        self.update_v(state, logger)
        self.update_actor(state, action, logger)
        self.update_q(state, action, reward, next_state, not_done, logger)
        self.update_target()

    def save(self, model_dir):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))
        torch.save(self.actor_scheduler.state_dict(), os.path.join(
            model_dir, f"actor_scheduler_s{str(self.total_it)}.pth"))

        torch.save(self.value.state_dict(), os.path.join(model_dir, f"value_s{str(self.total_it)}.pth"))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            model_dir, f"value_optimizer_s{str(self.total_it)}.pth"))
