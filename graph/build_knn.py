import numpy as np
import pandas as pd
from pathlib import Path

K = 8                 # number of neighbors
SIGMA = 0.5           # km
SELF_LOOP = True

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

dist_path = RAW_DIR / "wam_metr_la.csv"
dist = pd.read_csv(dist_path).values

N = dist.shape[0]
adj = np.zeros((N, N), dtype=np.float32)

for i in range(N):
    # exclude self
    distances = dist[i].copy()
    distances[i] = np.inf

    # find k nearest neighbors
    nn_idx = np.argsort(distances)[:K]

    for j in nn_idx:
        d = dist[i, j]
        weight = np.exp(-(d ** 2) / (SIGMA ** 2))
        adj[i, j] = weight

if SELF_LOOP:
    np.fill_diagonal(adj, 1.0)

np.save(OUT_DIR / "adj_knn_dir.npy", adj)

print("✅ Directed k-NN adjacency built")
print("Nodes:", N)
print("Edges:", np.count_nonzero(adj))
print("Density:", np.count_nonzero(adj) / (N * N))
