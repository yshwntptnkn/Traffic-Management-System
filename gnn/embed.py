import torch
import torch.nn as nn

class LSTM_GNN(nn.Module):
    def __init__(self, forecasting, gnn):
        super().__init__()
        self.forecasting = forecasting
        self.gnn = gnn

    def forward(self, x, adj):
        node_emb = self.forecasting(x, return_embeddings=True)
        out = self.gnn(node_emb, adj)

        return out