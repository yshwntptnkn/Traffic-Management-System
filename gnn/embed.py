import torch
import torch.nn as nn

class LSTM_GNN(nn.Module):
    def __init__(self, forecasting, gnn):
        super().__init__()
        self.forecasting = forecasting
        self.gnn = gnn

    def forward(self, x, adj, return_embeddings=False):
        node_emb = self.forecasting(x, return_embeddings=True)
        #print("Node embedding shape:", node_emb.shape)
        out = self.gnn(
            node_emb, 
            adj,
            return_embeddings=return_embeddings
        )

        return out