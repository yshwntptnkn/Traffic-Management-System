import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, series, seq_len=12, pred_len=3):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.series[idx:idx + self.seq_len]
        y = self.series[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        #print("X:", x)
        #print("Y:", y)

        return x, y