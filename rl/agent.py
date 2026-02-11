import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.net(x)
    
class DQNAgent:
    def __init__(self, state_dim, num_actions, lr=1e-3):
        self.model = DQN(state_dim, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.num_actions = num_actions
        self.gamma = 0.99
        self.epsilon = 1.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        
    def train_step(self, state, action, reward, next_state):
        q_value = self.model(state)
        next_q_values = self.model(next_state)

        target = q_value.clone().detach()
        target[action] = reward + self.gamma * torch.max(next_q_values)

        loss = self.loss_fn(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.1, self.epsilon * 0.995)