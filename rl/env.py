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

        self.action_space = spaces.Discrete(self.num_nodes + 1)

        embedding_dim = 32
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes * embedding_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        initial_queue = torch.rand(self.num_nodes, device=self.device) * 20 + 5

        self.history = torch.stack(
            [initial_queue.unsqueeze(-1) for _ in range(self.seq_len)],
            dim=0
        )

        state = self._get_state()

        return state, {}

    def _get_state(self):
        with torch.no_grad():
            x = self.history.unsqueeze(0)
            embeddings = self.encoder(x, self.adj, return_embeddings=True)
            state = embeddings.squeeze(0).flatten()
            return state.cpu().numpy().astype(np.float32)

    def step(self, action):

        self.current_step += 1

        current_queue = self.history[-1].squeeze(-1)

        prev_total_queue = current_queue.sum()

        arrivals = torch.rand(self.num_nodes, device=self.device) * 1.0
        new_queue = current_queue + arrivals

        new_total_queue = new_queue.sum()
        queue_reduction = prev_total_queue - new_total_queue

        action_cost = 0.0

        if action > 0:
            node_index = action - 1

            clearance_capacity = 20.0
            cleared = torch.minimum(
                new_queue[node_index],
                torch.tensor(clearance_capacity, device=self.device)
            )

            new_queue[node_index] -= cleared
            action_cost = 0.5

        spill = torch.matmul(self.adj, new_queue) * 0.005
        new_queue = new_queue + spill

        max_queue = 100.0
        new_queue = torch.clamp(new_queue, 0.0, max_queue)

        self.history = torch.cat(
            [self.history[1:], new_queue.unsqueeze(-1).unsqueeze(0)],
            dim=0
        )

        avg_queue = new_queue.mean()

        if action > 0:
            node_index = action - 1
            reward = queue_reduction.item() - action_cost
        else:
            reward = -0.1

        terminated = avg_queue > 80.0      # network collapse
        truncated = self.current_step >= self.max_steps

        next_state = self._get_state()

        return next_state, reward, terminated, truncated, {}