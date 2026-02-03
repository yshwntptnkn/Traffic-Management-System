import torch.nn as nn

class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_embeddings=False):
    
        B, T, N, F = x.shape  # Batch, Time, Nodes, Features
        x = x.reshape(B * N, T, F)

        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # [B*N, hidden]

        node_emb = last_hidden.view(B, N, self.hidden_size)

        if return_embeddings:
            return node_emb  # [B, N, hidden]

        preds = self.fc(node_emb)    # [B, N, output]
        return preds
