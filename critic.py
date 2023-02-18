import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MLP


class ValueCritic(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_layers,
        **kwargs
    ) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, 1, hidden_dim, n_layers, **kwargs)

    def forward(self, state):
        return self.mlp(state)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes=[256,256]):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		if len(state.shape) == 3:
			sa = torch.cat([state, action], 2)
		else:
			sa = torch.cat([state, action], 1)

		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)

		return 