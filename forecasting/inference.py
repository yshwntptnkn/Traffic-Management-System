import torch
import numpy as np
import pandas as pd
from model import TrafficLSTM

seq_len = 12
pred_len = 3

model = TrafficLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=pred_len
)

model.load_state_dict(torch.load("experiments/exp_001_forecasting_only/lstm.pt"))
model.eval()

mean, std = np.load("experiments/exp_001_forecasting_only/mean_std.npy")

def forecast(recent_speeds):
    if len(recent_speeds) != seq_len:
        raise ValueError(f"Expected {seq_len} recent speeds, got {len(recent_speeds)}")
    
    series = np.array(recent_speeds)
    series = series.replace(0, np.nan).ffill().bfill().values
    series = (series - mean) / std

    x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        preds = model(x).numpy().squeeze(0)

    preds_denorm = preds * std + mean
    return preds_denorm.tolist()
