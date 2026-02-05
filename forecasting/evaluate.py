import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datasets.metr_la import TrafficDataset
from model import TrafficLSTM


vel = pd.read_csv("data/raw/vel_metr_la.csv")
vel = vel.iloc[:, 1:]          # drop timestamp
series = vel.iloc[:, 0].values # sensor 0

series = pd.Series(series)
series = series.replace(0, np.nan).ffill().bfill().values

seq_len = 12
pred_len = 3

n = len(series)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

test_series = series[val_end:]

mean, std = np.load("experiments/exp_001_forecasting_only/mean_std.npy")
test_series = (test_series - mean) / std


test_dataset = TrafficDataset(
    test_series,
    seq_len=seq_len,
    pred_len=pred_len
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = TrafficLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=pred_len
)

model.load_state_dict(
    torch.load("experiments/exp_001_forecasting_only/lstm.pt")
)
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        preds = model(x)

        all_preds.append(preds.numpy())
        all_targets.append(y.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

mean = float(mean)
std = float(std)

all_preds_denorm = all_preds * std + mean
all_targets_denorm = all_targets * std + mean

mae = mean_absolute_error(all_targets_denorm, all_preds_denorm)
rmse = np.sqrt(mean_squared_error(all_targets_denorm, all_preds_denorm))

print(f"Test MAE  : {mae:.3f}")
print(f"Test RMSE : {rmse:.3f}")
