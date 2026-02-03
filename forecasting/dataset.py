# forecasting/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class METRLADataset(Dataset):
    def __init__(self, csv_path, seq_len=12, horizon=1):
        data = pd.read_csv(csv_path, index_col=0).values  # [T, N]
        data = data[..., None]  # [T, N, 1]

        self.x = []
        self.y = []

        for i in range(len(data) - seq_len - horizon):
            self.x.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len:i+seq_len+horizon])
            
        self.x = torch.from_numpy(np.array(self.x)).float()
        self.y = torch.from_numpy(np.array(self.y)).float()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
