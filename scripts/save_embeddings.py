import torch
import numpy as np
from pathlib import Path

from forecasting.model import TrafficLSTM
from gnn.model import GCN
from gnn.embed import LSTM_GNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]

LSTM_CKPT = ROOT / "experiments" / "exp_001_forecasting_only" / "lstm.pt"
GNN_CKPT = ROOT / "experiments" / "exp_002_gnn_no_rl" / "gnn.pt"
ENCODER_CKPT = ROOT / "experiments" / "exp_003_embeddings_encoder" / "encoder_frozen.pt"

ADJ_PATH = ROOT / "data" / "processed" / "adj.npy"
SAVE_PATH = ROOT / "experiments" / "exp_003_embeddings_encoders" / "gnn_embeddings.npy"

def main():
    adj = torch.tensor(
        np.load(ADJ_PATH),
        dtype=torch.float32,
        device=DEVICE
    )

    lstm = TrafficLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ).to(DEVICE)

    state_dict = torch.load(LSTM_CKPT, map_location=DEVICE)

    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    lstm.load_state_dict(state_dict, strict=False)


    gnn = GCN(
        in_dim=64,
        hidden_dim=32,
        out_dim=32
    ).to(DEVICE)

    state_dict = torch.load(GNN_CKPT, map_location=DEVICE)

    # Remove final layer weights
    state_dict.pop("fc2.weight", None)
    state_dict.pop("fc2.bias", None)

    gnn.load_state_dict(state_dict, strict=False)


    encoder = LSTM_GNN(lstm, gnn).to(DEVICE)
    encoder.eval()

    seq_len = 12
    num_nodes = adj.shape[0]

    history = torch.rand(
        1, seq_len, num_nodes, 1,
        device=DEVICE
    ) * 40 + 20

    with torch.no_grad():
        embeddings = encoder(history, adj, return_embeddings=True)
    
    embeddings = embeddings.squeeze(0).cpu().numpy()

    print("Embedding shape:", embeddings.shape)

    np.save(SAVE_PATH, embeddings)
    print("✅ Embeddings saved to:", SAVE_PATH)

if __name__ == "__main__":
    main()