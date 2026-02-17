import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

'''
    WIP: REWARD FUNC BROKEN, RUN TRAIN
'''


class TrafficEnv(gym.Env):
    metadata = {"render_modes":[]}

    def __init__(self, embeddings, adj=None, device="cpu"):
        super().__init__()

        self.base_embeddings = embeddings
        self.device = device

        self.num_nodes, self.embed_dim = embeddings.shape

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
        state = self._get_state()
        return state, {}

    def _get_state(self):
        emb = self.base_embeddings.copy()
        emb[:, 0] = self.queue.numpy()

        return emb.flatten().astype(np.float32)

    def step(self, action):

        self.current_step += 1

        prev_total_queue = self.queue.sum().item()

        arrivals = torch.rand(self.num_nodes) * 0.1
        self.queue += arrivals

        #action_cost = 0.0

        if action > 0:
            node_index = action - 1
            clearance = 60.0

            cleared = min(self.queue[node_index].item(), clearance)
            self.queue[node_index] -= cleared
            #action_cost = 0.5

        new_total_queue = self.queue.sum().item()
        reward = (prev_total_queue - new_total_queue) 

        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_state(), reward, terminated, truncated, {}