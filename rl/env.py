import torch
import numpy as np

class TrafficEnv:
    def __init__(self, encoder, adj, num_nodes=206, device="cpu"):
        self.encoder = encoder
        self.adj = adj
        self.num_nodes = num_nodes
        self.device = device

        self.max_speed = 70.0
        self.min_speed = 0.0
        self.seq_len = 12

        self.alpha = 0.15
        self.beta = 1.5
        self.gamma = 2.0

    def reset(self):
        self.history = torch.rand(
            self.seq_len, 
            self.num_nodes, 
            1,
            device=self.device
        ) * 40 + 20

        return self._get_state()

    def _get_state(self):
        with torch.no_grad():
            x = self.history.unsqueeze(0)  # add batch
            embeddings = self.encoder(x, self.adj)
            return embeddings.squeeze(0)

    def step(self, action):
        """
        action:
        0 = do nothing
        1 = improve flow (green bias)
        """

        current_speed = self.history[-1].squeeze(-1)

        neighbour_effect = torch.matmul(self.adj, current_speed)
        neighbour_effect = neighbour_effect / (self.adj.sum(dim=1) + 1e-6)

        new_speed = (current_speed + self.alpha * (neighbour_effect - current_speed))

        congestion = torch.relu(50 - current_speed)
        new_speed -= self.beta * (congestion / 50)

        if action == 1:
            new_speed += self.gamma

        new_speed = torch.clamp(new_speed, self.min_speed, self.max_speed)
            
        self.history = torch.cat(
            [self.history[1:], new_speed.unsqueeze(-1).unsqueeze(0)],
            dim=0
        )

        next_state = self._get_state()

        reward = new_speed.mean().item()

        done = False

        return next_state, reward, done
