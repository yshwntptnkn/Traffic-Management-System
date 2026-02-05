import numpy as np
import pandas as pd

def preprocess_series(series, mean=None, std=None):
    series = pd.Series(series)
    series = series.replace(0, np.nan).ffill().bfill().values

    if mean is None or std is None:
        mean = series.mean()
        std = series.std()

    processed_series = (series - mean) / std
    return processed_series, mean, std
