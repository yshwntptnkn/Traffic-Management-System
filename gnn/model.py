import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj, return_embeddings=False):
        x = torch.matmul(adj, x)
        h = torch.relu(self.fc1(x))
        
        if return_embeddings:
            return h
        
        out = self.fc2(h)

        return out