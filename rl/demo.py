import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from rl.env import TrafficEnv
from rl.agent import DQNAgent

ROOT = Path(__file__).resolve().parents[1]
embeddings = np.load(ROOT / "experiments" / "exp_003_embeddings_encoders" / "gnn_embeddings.npy")

env_fixed = TrafficEnv(embeddings)
env_rl = TrafficEnv(embeddings)

agent = DQNAgent(
    state_dim=env_rl.observation_space.shape[0],
    action_dim=env_rl.action_space.n
)

agent.model.load_state_dict(torch.load(ROOT / "experiments" / "exp_004_rl" / "dqn_policy.pt"))
agent.model.eval()
agent.epsilon = 0.0


def fixed_policy(step, num_nodes):
    return (step % num_nodes) + 1

def rl_policy(state):
    state_tensor = torch.from_numpy(state).float()
    with torch.no_grad():
        q_values = agent.model(state_tensor)
    return torch.argmax(q_values).item()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax_fixed, ax_rl = axes

bars_fixed = ax_fixed.bar(range(env_fixed.num_nodes), np.zeros(env_fixed.num_nodes))
bars_rl = ax_rl.bar(range(env_rl.num_nodes), np.zeros(env_rl.num_nodes))

ax_fixed.set_ylim(0, 150)
ax_rl.set_ylim(0, 150)


state_fixed, _ = env_fixed.reset()
state_rl, _ = env_rl.reset()


def update(frame):
    global state_fixed, state_rl

    action_fixed = fixed_policy(frame, env_fixed.num_nodes)
    next_state_fixed, _, term_r, trunc_r, _ = env_fixed.step(action_fixed)
    state_fixed = next_state_fixed

    action_rl = rl_policy(state_rl)
    next_state_rl, _, term_r, trunc_r, _ = env_rl.step(action_rl)
    state_rl = next_state_rl

    for i, bar in enumerate(bars_fixed):
        bar.set_height(env_fixed.queue[i].item())
        bar.set_color("red" if i == action_fixed - 1 else "blue")

    for i, bar in enumerate(bars_rl):
        bar.set_height(env_rl.queue[i].item())
        bar.set_color("red" if i == action_rl - 1 else "green")

    return bars_fixed + bars_rl

ani = animation.FuncAnimation(
    fig,
    update,
    frames=50,
    interval=500,
    blit=False
)

plt.tight_layout()
plt.show()