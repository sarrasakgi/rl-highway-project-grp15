"""
ddqn_agent.py : Double DQN agent with target network and epsilon-greedy policy.

The only difference from DQNAgent is in the target computation inside update():
  DQN:  q_next = target_net(s').max()          (same net selects and evaluates)
  DDQN: q_next = target_net(s').gather(q_net(s').argmax())  (split selection/evaluation)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork
from replay_buffer import ReplayBuffer


class DDQNAgent:

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        # network
        hidden_sizes: tuple = (256, 256),
        # replay
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        # learning
        lr: float = 5e-4,
        gamma: float = 0.99,
        # exploration
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        # target network
        target_update_freq: int = 1_000,
        # misc
        min_buffer_size: int = 1_000,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size
        self.device = torch.device(device)
        self.step_count = 0

        # Networks
        self.q_net = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.buffer = ReplayBuffer(buffer_capacity)

        # Logging
        self.losses = []

    # Interaction

    def select_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    # Learning

    def update(self):

        if len(self.buffer) < self.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        # Current Q-values for taken actions
        q_pred = self.q_net(states_t).gather(1, actions_t)

        # Double DQN target: main net selects best action, target net evaluates it
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states_t).gather(1, best_actions)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # Hard-copy target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # Persistence

    def save(self, path: str):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "eps": self.eps,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.eps = checkpoint["eps"]
