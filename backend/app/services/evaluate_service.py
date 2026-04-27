"""Evaluation service for CartPole DQN agent."""

from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import torch

from app.models.dqn_model import DQN


def evaluate_dqn(evaluation_episodes: int = 20) -> Dict[str, object]:
    """Evaluate the trained DQN agent on CartPole-v1.

    Evaluation uses greedy action selection (argmax Q-value), without epsilon-greedy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).resolve().parents[1] / "outputs" / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")

    policy_net = DQN(input_dim=4, output_dim=2).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    env = gym.make("CartPole-v1")
    rewards: List[float] = []

    for _ in range(evaluation_episodes):
        state, _info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += float(reward)

        rewards.append(total_reward)

    env.close()

    average_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    best_reward = float(max(rewards)) if rewards else 0.0

    return {
        "evaluation_episodes": evaluation_episodes,
        "average_reward": average_reward,
        "best_reward": best_reward,
        "rewards": rewards,
    }
