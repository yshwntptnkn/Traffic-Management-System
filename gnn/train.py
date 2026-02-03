import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from forecasting import dataset
from forecasting.model import TrafficLSTM
from forecasting.dataset import METRLADataset
from gnn.model import GCN
from gnn.embed import LSTM_GNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

def train():

    dataset = METRLADataset(
    csv_path=ROOT / "data" / "raw" / "vel_metr_la.csv",
    seq_len=12,
    horizon=1
    )

    loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
    )

    adj = torch.tensor(
        np.load(DATA_DIR / "adj.npy"),
        dtype=torch.float32,
        device=DEVICE
    )

    forecasting = TrafficLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=3
    ).to(DEVICE)

    forecasting.load_state_dict(torch.load(ROOT / "experiments" / "exp_001_forecasting_only" / "lstm.pt", map_location=DEVICE))

    for p in forecasting.parameters():
        p.requires_grad = False

    forecasting.eval()

    gnn = GCN(
        in_dim=64,
        hidden_dim=32,
        out_dim=1
    ).to(DEVICE)

    model = LSTM_GNN(
        forecasting,
        gnn
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )

    loss_fn = torch.nn.L1Loss()

    NUM_NODES = adj.shape[0]
                    
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x = x[:, :, :NUM_NODES, :].to(DEVICE)  
            y = y[:, 0, :NUM_NODES, :].to(DEVICE)

            optimizer.zero_grad()

            preds = model(x, adj)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch:03d} | MAE {epoch_loss/len(loader):.4f}")

    EXP_DIR = ROOT / "experiments" / "exp_003_gnn_no_rl"
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.gnn.state_dict(),
        EXP_DIR / "gnn.pt"
    )


if __name__ == "__main__":
    train()