import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from utils.preprocessing import preprocess_series


class METRLADataset(Dataset):
    def __init__(self, csv_path, seq_len=12, horizon=1, mean=None, std=None):

        self.seq_len = seq_len
        self.horizon = horizon

        vel = pd.read_csv(csv_path)
        vel = vel.iloc[:, 1:]          # drop timestamp
        data = vel.values              # [T, N]

        processed = []
        for i in range(data.shape[1]):
            series_i, mean_i, std_i = preprocess_series(
                data[:, i],
                mean=mean,
                std=std
            )
            processed.append(series_i)

        data = np.stack(processed, axis=1)  # [T, N]

        # Store mean/std (shared across sensors)
        self.mean = mean_i if mean is not None else data.mean()
        self.std = std_i if std is not None else data.std()

        # Add feature dimension
        self.data = data[..., None]     # [T, N, 1]

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y
