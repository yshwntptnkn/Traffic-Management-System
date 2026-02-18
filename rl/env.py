import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    metadata = {"render_modes":[]}

    def __init__(self, embeddings, adj=None, device="cpu"):
        super().__init__()

        self.base_embeddings = embeddings[:10].copy()
        self.device = device

        self.num_nodes = self.base_embeddings.shape[0]
        self.embed_dim = self.base_embeddings.shape[1]

        self.max_steps = 50
        self.current_step = 0

        self.action_space = spaces.Discrete(self.num_nodes + 1)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes * self.embed_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.queue = torch.rand(self.num_nodes) * 20 + 5

        #print("Num nodes:", self.num_nodes)
        #print("Queue shape:", self.queue.shape)


        state = self._get_state()
        return state, {}

    def _get_state(self):
        emb = self.base_embeddings.copy()
        emb[:, 0] = self.queue.numpy()

        #print("Embedding shape:", emb.shape)
        #print("Queue shape:", self.queue.shape)


        return emb.flatten().astype(np.float32)

    def step(self, action):

        self.current_step += 1

        prev_total_queue = self.queue.sum().item()

        arrivals = torch.zeros(self.num_nodes)

        num_active = max(1, int(self.num_nodes * 0.1))
        active_nodes = torch.randint(0, self.num_nodes, (num_active,))
        arrivals[active_nodes] = torch.rand(num_active) * 5.0

        self.queue += arrivals

        action_cost = 0.0

        if action > 0:
            node_index = action - 1

            clearance_capacity = 60.0

            cleared = min(self.queue[node_index].item(), clearance_capacity)
            self.queue[node_index] -= cleared

            action_cost = 1.0

        if hasattr(self, "adj") and self.adj is not None:
            spill = torch.matmul(self.adj, self.queue) * 0.001
            self.queue += spill    
        
        self.queue = torch.clamp(self.queue, 0.0, 200.0)

        new_total_queue = self.queue.sum().item()
        reward = (prev_total_queue - new_total_queue) - action_cost 
        reward = max(min(reward, 10), -10)

        terminated = self.current_step >= self.max_steps
        truncated = False

        next_state = self._get_state()

        return next_state, reward, terminated, truncated, {}