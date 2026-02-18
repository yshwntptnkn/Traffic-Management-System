import torch
from pathlib import Path
import json
from forecasting.model import TrafficLSTM
from gnn.model import GCN
from gnn.embed import LSTM_GNN
from rl.env import TrafficEnv
from rl.agent import DQNAgent
from rl.replay_buffer import ReplayBuffer
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

embeddings = np.load(ROOT / "experiments" / "exp_003_embeddings_encoders" / "gnn_embeddings.npy")
print(embeddings.shape)

env = TrafficEnv(embeddings)

state, _ = env.reset()
#print("State shape:", state.shape)
state_dim = state.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim, device=DEVICE)

replay_buffer = ReplayBuffer(capacity=100000)

NUM_EPISODES = 500
STEPS_PER_EPISODE = 50

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 400

episode_rewards = []
ep_count = 100

for episode in range(NUM_EPISODES):

    epsilon = max(
    EPS_END,
    EPS_START - episode / EPS_DECAY_EPISODES
    )
    agent.epsilon = epsilon

    state, _ = env.reset()
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):

        state_tensor = torch.from_numpy(state).float()
        action = agent.select_action(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        replay_buffer.push(state, action, reward, next_state, done)

        agent.train_step(replay_buffer)

        state = next_state
        total_reward += reward

    if episode % 10 == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())
        
    episode_rewards.append(total_reward)
    print(f"Episode {episode} | Total Reward: {total_reward:.2f}")

    if len(episode_rewards) == ep_count:
        recent_avg = np.mean(episode_rewards[-10:])
        print(f"Last 10 Avg Reward: {recent_avg:.2f}")
        ep_count += 100

    if episode % 20 == 0:
        print("Epsilon:", agent.epsilon)


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
