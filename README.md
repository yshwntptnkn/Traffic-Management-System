# 🚦 Traffic Management System — Forecasting Module

This repository contains the **forecasting component** of a larger Traffic Management System.
At this stage, the focus is on **short-term traffic speed prediction** using historical sensor data.

The forecasting layer is designed to later integrate with:

- Graph Neural Networks (spatial modeling)

- Reinforcement Learning (traffic signal control)

For now, this repo implements and evaluates a **robust LSTM-based time-series forecasting baseline.**

---

## 📌 Project Status

**Current stage:**

✅ Traffic forecasting layer complete

⏳ Graph (GNN) modeling — planned

⏳ RL-based control — planned

The forecasting model is considered **frozen and production-ready** for downstream integration.

---

## 🎯 Objective

Predict future traffic speeds for a road segment using recent historical observations.

Formally:
*Given traffic speeds over the past 60 minutes, predict speeds for the next 15 minutes.*

---

## 📊 Dataset

**METR-LA Traffic Dataset**

- Traffic speed data from highway loop detectors in Los Angeles

- Temporal resolution: 5-minute intervals

- Sensors: ~200 road sensors

- Data used:

    - ```vel_metr_la.csv``` — traffic speed time series

    - ```wam_metr_la.csv``` — weighted adjacency matrix (reserved for future GNN use)

At this stage, **only speed data** is used.
Graph connectivity is intentionally deferred.

---

## 🧠 Forecasting Model

**Model Architecture**

- Type: LSTM (Long Short-Term Memory)

- Layers: 2

- Hidden size: 64

- Dropout: 0.2

- Input: Traffic speed only

- Output: Multi-step speed forecast

**Forecasting Setup**

- Input window: 12 timesteps (60 minutes)

- Prediction horizon: 3 timesteps (15 minutes)

- Loss function: Mean Squared Error (MSE)

- Optimizer: Adam (lr = 5e-4)

---

## 🔧 Data Preprocessing

To ensure robustness and realism:

- **Chronological train/validation/test split**
    - 70% train / 15% val / 15% test

- **Zero-value handling**
    - Zero speeds are forward-filled and backward-filled

- **Normalization**
    - Z-score normalization using training data statistics only

- **No data leakage**
    - Test data is never used during training or normalization

Normalization parameters are saved and reused during evaluation and inference.

---

## 📈 Evaluation Results

The model was evaluated on unseen test data.

**Metrics (denormalized, mph):**
```bash
    MAE = 2.03 mph
    RMSE = 4.68 mph
``` 

These results are consistent with strong single-sensor forecasting baselines reported in traffic forecasting literature.

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
│
├── forecasting/
│   ├── dataset.py     # Sliding-window dataset
│   ├── model.py       # LSTM architecture
│   ├── train.py       # Model training
│   ├── evaluate.py    # Evaluation on test data
│   └── inference.py   # Deployment-ready inference API
│
├── experiments/
│   └── exp_001_forecasting_only/
│       ├── lstm.pt    # Trained model weights
│       └── mean_std.npy   # Normalization parameters
│
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
    python forecasting/train.py
```

3. **Evaluate the model**
```bash
    python forecasting/evaluate.py
```

4. **Run inference**
```bash
    from forecasting.inference import forecast

    recent_speeds = [ ... 12 recent speed values ... ]
    future_speeds = forecast(recent_speeds)
```

---

## 🔮 Next Steps

Planned extensions (not yet implemented):

- Graph construction using sensor adjacency (```wam_metr_la.csv```)

- Spatio-temporal modeling with Graph Neural Networks

- Reinforcement Learning for adaptive traffic signal control

- End-to-end simulation and deployment pipeline

---

## 📌 Notes

- The forecasting layer is intentionally kept **simple but robust**

- Performance gains are expected primarily from **spatial modeling**, not further LSTM tuning

- This repository represents **Phase 1** of a larger intelligent traffic system
