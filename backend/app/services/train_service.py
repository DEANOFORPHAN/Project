"""DQN training service for CartPole-v1."""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.core.config import DQN_TRAINING_CONFIG
from app.models.dqn_model import DQN
from app.models.replay_buffer import ReplayBuffer


def select_action(
    policy_net: DQN,
    state,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    """Choose action by epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    replay_buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> Optional[float]:
    """Run one DQN optimization step and return loss."""
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert list of numpy arrays to one ndarray first to avoid PyTorch warning.
    states_np = np.array(states, dtype=np.float32)
    next_states_np = np.array(next_states, dtype=np.float32)

    states_tensor = torch.tensor(states_np, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(next_states_np, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    current_q = policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states_tensor)
        max_next_q = next_q_values.max(1)[0]

    target_q = rewards_tensor + gamma * max_next_q * (1.0 - dones_tensor)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())


def evaluate_policy_greedy(
    policy_net: DQN,
    eval_env: gym.Env,
    device: torch.device,
    eval_episodes: int = 5,
) -> float:
    """Evaluate current policy with greedy actions and return average reward."""
    rewards: List[float] = []
    policy_net.eval()
    with torch.no_grad():
        for _ in range(eval_episodes):
            state, _info = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(policy_net(state_tensor), dim=1).item())
                next_state, reward, terminated, truncated, _info = eval_env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += float(reward)
            rewards.append(total_reward)
    policy_net.train()
    return float(np.mean(rewards)) if rewards else 0.0


def train_dqn(episodes: Optional[int] = None) -> Dict[str, object]:
    """Train DQN on CartPole-v1 and return training artifacts."""
    cfg = DQN_TRAINING_CONFIG.copy()
    if episodes is not None:
        cfg["episodes"] = episodes

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(input_dim=4, output_dim=2).to(device)
    target_net = DQN(input_dim=4, output_dim=2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    episodes = cfg["episodes"]
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["learning_rate"])
    batch_size = cfg["batch_size"]
    gamma = cfg["gamma"]
    epsilon_start = cfg["epsilon_start"]
    epsilon_end = cfg["epsilon_end"]
    epsilon_decay = cfg["epsilon_decay"]
    epsilon_warmup_steps = cfg["epsilon_warmup_steps"]
    target_update_steps = cfg["target_update_steps"]
    warmup_size = cfg["warmup_size"]
    replay_buffer_capacity = cfg["replay_buffer_capacity"]
    checkpoint_eval_episodes = cfg["checkpoint_eval_episodes"]
    early_stop_avg20_threshold = cfg["early_stop_avg20_threshold"]
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    episode_rewards: List[float] = []
    global_step = 0
    best_avg_reward = float("-inf")
    best_episode = -1
    best_single_reward = float("-inf")
    best_eval_avg_reward = float("-inf")
    stopped_early = False
    stop_episode = episodes

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = outputs_dir / "best_model.pth"

    for episode in range(episodes):
        state, _info = env.reset()
        done = False
        total_reward = 0.0
        episode_loss: Optional[float] = None
        epsilon = epsilon_start

        while not done:
            if global_step < epsilon_warmup_steps:
                epsilon = epsilon_start
            else:
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * torch.exp(
                    torch.tensor(-1.0 * global_step / epsilon_decay)
                ).item()
            # 根据当前的epsilon决定贪心选择还是随机选择
            # 贪心的情况下，选择当前状态下Q值最高的动作；随机选择则从动作空间中随机选一个动作。
            action = select_action(policy_net, state, epsilon, action_dim=2, device=device)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            # 把当前的状态、动作、奖励、下一个状态和是否结束的标志存储到经验回放缓冲区中。
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += float(reward)
            global_step += 1

            # Warm-up: collect enough transitions before training updates.
            if len(replay_buffer) >= warmup_size:
                # 对比当前状态下动作的Q值和目标Q值，计算损失，并使用优化器更新策略网络的参数。
                episode_loss = optimize_model(
                    policy_net=policy_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                )

                # Update target network by environment steps (more stable than by episodes).
                if global_step % target_update_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        best_single_reward = max(best_single_reward, total_reward)

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"reward={total_reward:.2f} | epsilon={epsilon:.4f} | "
            f"loss={episode_loss if episode_loss is not None else 'None'}"
        )
        if (episode + 1) % 10 == 0:
            last_10_avg = float(np.mean(episode_rewards[-10:]))
            print(f"Average reward (last 10 episodes): {last_10_avg:.2f}")

            # Save checkpoint when greedy evaluation score improves.
            eval_avg_reward = evaluate_policy_greedy(
                policy_net=policy_net,
                eval_env=eval_env,
                device=device,
                eval_episodes=checkpoint_eval_episodes,
            )
            print(
                f"Checkpoint eval avg ({checkpoint_eval_episodes} eps, greedy): "
                f"{eval_avg_reward:.2f}"
            )

            if eval_avg_reward > best_eval_avg_reward:
                best_eval_avg_reward = eval_avg_reward
                best_avg_reward = last_10_avg  # keep training-view metric for report
                best_episode = episode + 1
                torch.save(policy_net.state_dict(), best_model_path)
                print(
                    f"New best checkpoint saved: eval_avg={best_eval_avg_reward:.2f} "
                    f"at episode {best_episode}"
                )

        # Early stopping: solve CartPole when avg reward over last 20 episodes >= 475.
        if len(episode_rewards) >= 20:
            last_20_avg = float(np.mean(episode_rewards[-20:]))
            if last_20_avg >= early_stop_avg20_threshold:
                stopped_early = True
                stop_episode = episode + 1
                print(
                    f"Early stopping triggered at episode {stop_episode}: "
                    f"last_20_avg={last_20_avg:.2f}"
                )
                break

    env.close()
    eval_env.close()

    # Fallback: if no 10-episode checkpoint was recorded, save final model to best_model.pth.
    if best_episode == -1:
        torch.save(policy_net.state_dict(), best_model_path)
        if episode_rewards:
            best_episode = len(episode_rewards)
            best_avg_reward = float(np.mean(episode_rewards[-min(10, len(episode_rewards)) :]))
        print(f"No best-avg checkpoint found; final model saved: {best_model_path}")

    # Save rewards.csv
    rewards_csv_path = outputs_dir / "rewards.csv"
    with rewards_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for idx, reward in enumerate(episode_rewards, start=1):
            writer.writerow([idx, float(reward)])

    # Save reward_curve.png (raw rewards + moving average)
    reward_curve_path = outputs_dir / "reward_curve.png"
    plt.figure(figsize=(10, 5))
    x = np.arange(1, len(episode_rewards) + 1)
    y = np.array(episode_rewards, dtype=np.float32)
    plt.plot(x, y, label="Episode Reward", alpha=0.5)

    ma_window = 10
    if len(y) >= ma_window:
        moving_avg = np.convolve(y, np.ones(ma_window) / ma_window, mode="valid")
        x_ma = np.arange(ma_window, len(y) + 1)
        plt.plot(x_ma, moving_avg, label=f"Moving Average ({ma_window})", linewidth=2)

    plt.title("CartPole Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reward_curve_path)
    plt.close()

    final_avg_reward_last_10 = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
    final_avg_reward_last_50 = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0

    metrics = {
        "total_episodes": len(episode_rewards),
        "best_avg_reward": float(best_avg_reward) if best_episode != -1 else None,
        "best_eval_avg_reward": float(best_eval_avg_reward) if best_episode != -1 else None,
        "best_episode": best_episode if best_episode != -1 else None,
        "best_single_reward": float(best_single_reward) if episode_rewards else None,
        "final_avg_reward_last_10": final_avg_reward_last_10,
        "final_avg_reward_last_50": final_avg_reward_last_50,
        "stopped_early": stopped_early,
        "stop_episode": stop_episode,
        "hyperparameters": {
            "episodes": cfg["episodes"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "gamma": cfg["gamma"],
            "epsilon_start": cfg["epsilon_start"],
            "epsilon_end": cfg["epsilon_end"],
            "epsilon_decay": cfg["epsilon_decay"],
            "epsilon_warmup_steps": cfg["epsilon_warmup_steps"],
            "target_update_steps": cfg["target_update_steps"],
            "warmup_size": cfg["warmup_size"],
            "replay_buffer_capacity": cfg["replay_buffer_capacity"],
            "checkpoint_eval_episodes": cfg["checkpoint_eval_episodes"],
            "early_stop_avg20_threshold": cfg["early_stop_avg20_threshold"],
        },
    }

    # Save metrics.json
    metrics_json_path = outputs_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "episode_rewards": episode_rewards,
        "policy_net": policy_net,
        "target_net": target_net,
        "optimizer": optimizer,
        "replay_buffer": replay_buffer,
        "best_model_path": str(best_model_path),
        "rewards_csv_path": str(rewards_csv_path),
        "reward_curve_path": str(reward_curve_path),
        "metrics_json_path": str(metrics_json_path),
        "metrics": metrics,
    }
