import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

DIST_THRESHOLD_KM = 2.0

def build_graph():
    print("=== BUILD GRAPH START ===")
    print("ROOT:", ROOT)
    print("RAW_DIR:", RAW_DIR)
    print("OUT_DIR:", OUT_DIR)

    csv_path = RAW_DIR / "wam_metr_la.csv"
    print("CSV PATH:", csv_path)
    print("CSV EXISTS:", csv_path.exists())

    if not csv_path.exists():
        print("❌ CSV FILE NOT FOUND — exiting")
        return

    print("Loading CSV...")
    df = pd.read_csv(csv_path, index_col=0)
    print("CSV loaded")
    print("CSV shape:", df.shape)

    distances = df.values
    print("Distance matrix shape:", distances.shape)

    num_nodes = distances.shape[0]
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    edge_count = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = distances[i, j]
            if not np.isnan(d) and d <= DIST_THRESHOLD_KM:
                adj[i, j] = 1.0
                edge_count += 1

    print("Edges before self-loops:", edge_count)

    np.fill_diagonal(adj, 1.0)

    deg = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(deg, 1.0, None)

    edge_index = np.array(np.nonzero(adj))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "adj.npy", adj)
    np.save(OUT_DIR / "edge_index.npy", edge_index)

    print("✅ Graph saved")
    print("Adj shape:", adj.shape)
    print("Edge index shape:", edge_index.shape)
    print("=== BUILD GRAPH END ===")

if __name__ == "__main__":
    build_graph()
