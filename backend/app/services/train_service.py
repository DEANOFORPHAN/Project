"""First version of DQN training skeleton for CartPole-v1.

This file focuses on a clean training flow:
1) build env and networks
2) collect transitions with epsilon-greedy actions
3) record episode rewards

Gradient update details can be added later.
"""

import random
from typing import Dict, List

import gymnasium as gym
import torch
import torch.optim as optim

from app.models.dqn_model import DQN
from app.models.replay_buffer import ReplayBuffer


def select_action(policy_net: DQN, state, epsilon: float, action_dim: int, device: torch.device) -> int:
    """Choose action by epsilon-greedy policy.

    With probability epsilon: random action.
    Otherwise: action with the largest Q-value from policy network.
    """
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def train_dqn(episodes: int = 300) -> Dict[str, object]:
    """Run a basic DQN training skeleton on CartPole-v1.

    Returns a dictionary containing reward history and key objects.
    """
    env = gym.make("CartPole-v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Initialize core components.
    policy_net = DQN(input_dim=4, output_dim=2).to(device)
    target_net = DQN(input_dim=4, output_dim=2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000)

    # Basic epsilon schedule.
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 10000  # Larger means slower decay.

    episode_rewards: List[float] = []
    global_step = 0

    # 2) Main training loop.
    for _episode in range(episodes):
        state, _info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * torch.exp(
                torch.tensor(-1.0 * global_step / epsilon_decay)
            ).item()

            # Choose action -> interact with environment -> store transition.
            # 暂时只实现 1 epsilon-greedy action 2 env获取下一步状态 3 存储 transition 到 replay buffer
            # 后续再添加 gradient update step
            action = select_action(policy_net, state, epsilon, action_dim=2, device=device)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += float(reward)
            global_step += 1

            # TODO: Add gradient update step here (sample batch + compute loss + backprop).

        # 3) Record episode result.
        episode_rewards.append(total_reward)

        # Optional: periodically copy policy weights to target network.
        if (_episode + 1) % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    return {
        "episode_rewards": episode_rewards,
        "policy_net": policy_net,
        "target_net": target_net,
        "optimizer": optimizer,
        "replay_buffer": replay_buffer,
    }
