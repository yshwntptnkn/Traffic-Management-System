import torch
from torch.utils.data import DataLoader
from datasets.metr_la import METRLADataset
from forecasting.model import TrafficLSTM
from utils.preprocessing import preprocess_series
from pathlib import Path
import pandas as pd
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "vel_metr_la.csv"
EXP_DIR = ROOT / "experiments" / "exp_001_forecasting_only"
EXP_DIR.mkdir(parents=True, exist_ok=True)

vel = pd.read_csv(DATA_PATH)
vel = vel.iloc[:, 1:]              # drop timestamp
series = vel.iloc[:, 0].values     # single sensor

n = len(series)
train_end = int(0.7 * n)
train_series = series[:train_end]

mean = train_series.mean()
std = train_series.std()

# Save for eval / inference / GNN
np.save(EXP_DIR / "norm.npy", np.array([mean, std]))

train_series = preprocess_series(
    train_series,
    mean=mean,
    std=std
)

dataset = METRLADataset(
    series=train_series,
    seq_len=12,
    horizon=1
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = TrafficLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=1
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

for epoch in range(30):
    model.train()
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        x = x.unsqueeze(2)
        y = y[:, 0].to(DEVICE)

        optimizer.zero_grad()
        preds = model(x).squeeze(-1).squeeze(-1)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    avg_loss = loss_sum / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), EXP_DIR / "lstm.pt")
np.save(EXP_DIR / "mean_std.npy", np.array([mean, std]))

