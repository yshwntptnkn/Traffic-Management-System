import torch
from pathlib import Path

from forecasting.model import TrafficLSTM
from gnn.model import GCN
from gnn.embed import LSTM_GNN

ROOT = Path(__file__).resolve().parents[1]

LSTM_CKPT = ROOT / "experiments" / "exp_001_forecasting_only" / "lstm.pt"
GNN_CKPT = ROOT / "experiments" / "exp_002_gnn_no_rl" / "gnn.pt"

OUT_DIR = ROOT / "experiments" / "exp_003_frozen_encoder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lstm = TrafficLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=1
).to(DEVICE)

lstm.load_state_dict(torch.load(LSTM_CKPT, map_location=DEVICE))

gnn = GCN(
    in_dim=64,
    hidden_dim=32,
    out_dim=1   # embedding dim
).to(DEVICE)

gnn.load_state_dict(torch.load(GNN_CKPT, map_location=DEVICE))

encoder = LSTM_GNN(
    forecasting=lstm,
    gnn=gnn
).to(DEVICE)

encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

num_trainable = sum(p.requires_grad for p in encoder.parameters())
print(f"Trainable parameters after freezing: {num_trainable}")

torch.save(
    {
        "lstm_state": lstm.state_dict(),
        "gnn_state": gnn.state_dict(),
    },
    OUT_DIR / "encoder_frozen.pt"
)

print("✅ Frozen encoder saved to:", OUT_DIR)
