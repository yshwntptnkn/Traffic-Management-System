import torch
import numpy as np
from pathlib import Path
from rl.agent import DQNAgent
from rl.env import TrafficEnv

def random_policy(env, state):
    return env.action_space.sample()

def fixed_policy(env, state, step):
    return (step % env.num_nodes) + 1

'''def greedy_policy(env, state):
    return torch.argmax(env.queue).item() + 1
'''

def rl_policy(env, state, agent):
    state_tensor = torch.from_numpy(state).float()
    with torch.no_grad():
        q_values = agent.model(state_tensor)
    return torch.argmax(q_values).item()

def evaluate_controller(env, controller_fn, agent=None, episodes=50):

    total_rewards = []

    for ep in range(episodes):

        state, _ = env.reset()
        done= False
        total_reward = 0
        step = 0

        while not done:
            if controller_fn.__name__ == "fixed_policy":
                action = controller_fn(env, state, step)
            elif controller_fn.__name__ == "rl_policy":
                action = controller_fn(env, state, agent)
            else:
                action = controller_fn(env, state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            state = next_state
            step += 1

        total_rewards.append(total_reward)

    return np.mean(total_rewards), np.std(total_rewards)

def run_full_evaluation(env, agent):
    agent.epsilon = 0.0

    results = {}

    for policy in [
        random_policy,
        fixed_policy,
        #greedy_policy,
        rl_policy
    ]:
        
        mean_reward, std_reward = evaluate_controller(
            env,
            policy,
            agent=agent if policy == rl_policy else None,
            episodes=50
        )

        results[policy.__name__] = (mean_reward, std_reward)

    print("\n===== Controller Comparison =====")
    for name, (mean, std) in results.items():
        print(f"{name:15s} | Avg Reward: {mean:8.2f} | Std: {std:6.2f}")

def main():

    ROOT = Path(__file__).resolve().parents[1]
    model_path = ROOT / "experiments" / "exp_004_RL" / "dqn_policy.pt"
    embeddings_path = ROOT / "experiments" / "exp_003_embeddings_encoders" / "gnn_embeddings.npy"
    embeddings = np.load(embeddings_path)

    env = TrafficEnv(embeddings)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    agent.model.load_state_dict(torch.load(model_path))

    agent.model.eval()

    run_full_evaluation(env, agent)

if __name__ == "__main__":
    main()
            