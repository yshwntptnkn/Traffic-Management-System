import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import METRLADataset
from model import TrafficLSTM
import numpy as np

# Load data
vel = pd.read_csv("data/raw/vel_metr_la.csv")
vel = vel.iloc[:, 1:]          # drop timestamp
series = vel.iloc[:, 0].values # sensor 0

series = pd.Series(series)
series = series.replace(0, np.nan).ffill().bfill().values

print("Min speed:", series.min())
print("Max speed:", series.max())
print("Mean speed:", series.mean())

print(series[:10])

mean = series.mean()
std = series.std()

series = (series - mean) / std

dataset = METRLADataset(series)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TrafficLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

for epoch in range(30):
    loss_sum = 0
    for x, y in loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    avg_loss = loss_sum / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "experiments/exp_001_forecasting_only/lstm.pt")
np.save("experiments/exp_001_forecasting_only/mean_std.npy", np.array([mean, std]))
