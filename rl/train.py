import torch
from pathlib import Path
import json
from forecasting.model import TrafficLSTM
from gnn.model import GCN
from gnn.embed import LSTM_GNN
from rl.env import TrafficEnv
from rl.agent import DQNAgent
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "experiments" / "exp_004_rl"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

adj = torch.tensor(
    np.load(ROOT / "data" / "processed" / "adj.npy"),
    dtype=torch.float32,
    device=DEVICE
)

ckpt = torch.load(
    ROOT / "experiments" / "exp_003_frozen_encoder" / "encoder_frozen.pt",
    map_location=DEVICE
)

lstm = TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
gnn = GCN(in_dim=64, hidden_dim=32, out_dim=1)

lstm.load_state_dict(ckpt["lstm_state"])
gnn.load_state_dict(ckpt["gnn_state"])

encoder = LSTM_GNN(lstm, gnn).to(DEVICE)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

env = TrafficEnv(encoder, adj, device=DEVICE)

state = env.reset()
state_dim = state.numel()

agent = DQNAgent(state_dim, num_actions=2)

NUM_EPISODES = 50
STEPS_PER_EPISODE = 50
episode_rewards = []

for episode in range(NUM_EPISODES):
    state = env.reset().flatten()

    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        action = agent.select_action(state)

        next_state, reward, done = env.step(action)
        next_state = next_state.flatten()

        agent.train_step(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode {episode} | Total Reward: {total_reward:.2f}")

torch.save(
    agent.model.state_dict(),
    SAVE_DIR / "dqn_policy.pt"
)

print("✅ RL policy saved.")

np.save(
    SAVE_DIR / "reward_curve.npy",
    np.array(episode_rewards)
)

config = {
    "episodes": NUM_EPISODES,
    "steps_per_episode": STEPS_PER_EPISODE,
    "gamma": agent.gamma,
    "epsilon_final": agent.epsilon
}

with open(SAVE_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=4)
