import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    metadata = {"render_modes":[]}

    def __init__(self, encoder, adj, num_nodes=206, device="cpu"):
        super().__init__()

        self.encoder = encoder
        self.adj = adj
        self.num_nodes = num_nodes
        self.device = device

        self.max_speed = 70.0
        self.min_speed = 0.0
        self.seq_len = 12
        self.max_steps = 50
        self.current_step = 0

        self.alpha = 0.15
        self.beta = 1.5
        self.gamma = 2.0

        self.action_space = spaces.Discrete(2)

        embedding_dim = 32
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes * embedding_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.history = torch.rand(
            self.seq_len, 
            self.num_nodes, 
            1,
            device=self.device
        ) * 40 + 20

        state = self._get_state()

        return state, {}

    def _get_state(self):
        with torch.no_grad():
            x = self.history.unsqueeze(0)
            embeddings = self.encoder(x, self.adj, return_embeddings=True)
            state = embeddings.squeeze(0).flatten()
            return state.cpu().numpy().astype(np.float32)

    def step(self, action):

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

        reward = new_speed.mean().item()

        terminated = False
        truncated = self.current_step >= self.max_steps

        next_state = self._get_state()

        
        return next_state, reward, terminated, truncated, {}
