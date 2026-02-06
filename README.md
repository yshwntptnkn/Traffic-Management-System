# 🚦 Traffic Management System — Spatiotemporal Module

This repository contains the **spatiotemporal modeling component** of a larger Traffic Management System.
At this stage, the focus is on **robust representation of traffic dynamics** by combining:

- **Temporal Modeling** (LSTM-based forecasting)

- **Spatial modeling** (Graph Neural Networks over road sensors)

These learned representations are designed to serve as the **state encoder** for downstream
**reinforcement learning–based traffic signal control**.

---

## 📌 Project Status

**Current stage:**

✅ Temporal forecasting layer (LSTM) — complete & frozen

✅ Spatial modeling layer (GNN) — complete & evaluated

⏳ Reinforcement Learning for traffic signal control — next phase

The LSTM and GNN are treated as **representation-learning components**, not endlessly tuned predictors.

---

## 🎯 Objective

Learn a **spatiotemporal representation of traffic flow** from historical sensor data.

Formally:

*Given traffic speeds over the past 60 minutes across a road network, learn representations that capture both temporal dynamics and spatial dependencies, enabling short-term forecasting and downstream control.*

While short-term speed prediction is evaluated, **forecasting accuracy is not the final objective** — structured representations for decision-making are.

---

## 📊 Dataset

**METR-LA Traffic Dataset**

- Traffic speed data from highway loop detectors in Los Angeles

- Temporal resolution: 5-minute intervals

- Sensors: ~200 road sensors

- Data used:

    - ```vel_metr_la.csv``` — multivariate traffic speed time series

    - ```wam_metr_la.csv``` — inter-sensor distance matrix (used for graph construction)

---

## 🧠 Temporal Modeling - LSTM

**Model Architecture**

- Type: LSTM (Long Short-Term Memory)

- Layers: 2

- Hidden size: 64

- Dropout: 0.2

- Input: Traffic speed 

- Output: Multi-step speed forecast

**Training Setup**

- Input window: 12 timesteps (60 minutes)

- Prediction horizon: 3 timesteps (15 minutes)

- Loss function: Normalized Mean Squared Error (MSE)

- Optimizer: Adam (lr = 5e-4)

The LSTM is trained first and then **frozen**, acting as a **temporal feature extractor** for the GNN.

---

## 🌐 Spatial Modeling — GNN

A Graph Neural Network is trained **on top of LSTM embeddings** to explicitly model spatial dependencies between sensors.

**Graph Design**

Two graph constructions were explored:

1. **Dense distance-based graph** (baseline)

2. **Directed k - NN graph** (final choice)

    - A Graph Neural Network is trained on top of LSTM embeddings to explicitly model spatial dependencies between sensors.

    - Directional edges (models upstream → downstream influence)

    - Gaussian distance weighting

The directed k-NN graph was chosen for its **physical realism and suitability for traffic control**, even when forecasting gains are modest.

## GNN Role

- Refines LSTM outputs using spatial context

- Produces node **embeddings that encode network-level interactions**

- Acts as a **state encoder** for downstream RL

Importantly, the GNN is **not expected to dramatically outperform the LSTM on MAE alone** — its value lies in **structuring the system as a graph**.

---

## 🔧 Data Preprocessing

Preprocessing is centralized and shared across models.

- Zero-value handling via forward/backward fill

- Z-score normalization (training statistics only)

- No data leakage across splits

- Normalization parameters saved and reused consistently

---

## 📈 Evaluation Results

**Forecasting Metrics (denormalized, mph)**:

**LSTM-only**: MAE ≈ 2.0 mph

**LSTM + GNN**: Comparable MAE with **small but consistent improvements**

While raw forecasting gains are modest, the GNN:

- Enforces spatial inductive bias

- Produces structured node embeddings

- Improves robustness and suitability for control

These properties are **more important than marginal MAE gains** for traffic management.

---

## 📂 Repository Structure (Relevant Sections)

```bash
Traffic-Management-System/
│
├── data/
│   ├── raw/
│   │   ├── vel_metr_la.csv
│   │   └── wam_metr_la.csv
│   └── processed/
│       ├── adj.npy
│       ├── edge_index.npy
│       └── adj_knn_dir.npy
│
├── datasets/
│   └── metr_la.py      # Shared dataset & preprocessing logic
|
├── experiments/
│   ├── exp_001_forecasting_only/
│   │   ├── lstm.pt     # Trained model weights
│   │   └── mean_std.npy    # Normalization parameters
│   ├── exp_002_gnn_no_rl/
│   │   └── gnn.pt
│   └── exp_003_frozen_encoder/
│       └── freeze_encoder.py
│
├── forecasting/
│   ├── model.py        # LSTM architecture
│   ├── train.py        # Model training
│   ├── evaluate.py     # Evaluation on test data
│   └── inference.py    # Deployment-ready inference API
│
├── gnn/
│   ├── model.py        # GCN layers
│   ├── embed.py        # LSTM → GNN integration
│   └── train.py        # GNN training (LSTM frozen)
│
├── graph/
│   ├── build_graph.py
│   └── build_gknn.py
│
├── utils/
│   └── preprocessing.py 
│
├── LICENSE
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. **Set up environment**
```bash
    python -m venv .venv
    .\.venv\Scripts\activate    # or  source .venv/bin/activate
    pip install -r requirements.txt
```

2. **Train the model**
```bash
    python -m forecasting.train
```

3. **Evaluate the model**
```bash
    python -m forecasting.evaluate
```

4. **Run inference**
```bash
    from forecasting.inference import forecast

    recent_speeds = [ ... 12 recent speed values ... ]
    future_speeds = forecast(recent_speeds)
```

5. **Train GNN** (LSTM Frozen)
```bash
    python -m gnn.train
```

---

## 🔮 Next Steps — Traffic Signal Control (RL)

Planned extensions (not yet implemented):

- Use **LSTM + GNN embeddings as RL state**

- Integrate with SUMO or custom traffic simulator

- Design decentralized or coordinated signal control policies

- Evaluate system-level metrics:

    - delay

    - queue length

    - spillback

    - throughput

At this stage, **representation learning is complete** — the focus shifts from prediction to **decision-making**.

---

## 📌 Notes

- Strong temporal models alone saturate forecasting metrics

- Explicit spatial structure enables **system-level reasoning**

- GNNs are more valuable for **control and robustness** than raw MAE gains

- This repository represents the **foundation** of a **learning-based** traffic management system
