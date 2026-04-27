"""Play service: run one greedy CartPole episode and export animation GIF."""

from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import torch

from app.models.dqn_model import DQN


def play_once_and_save_gif(max_steps: int = 500, fps: int = 30) -> Dict[str, object]:
    """Run one greedy episode using best_model and save frames as GIF."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'imageio'. Please install it in your environment."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model_path = outputs_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")

    gif_path = outputs_dir / "play_latest.gif"

    policy_net = DQN(input_dim=4, output_dim=2).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _info = env.reset()

    done = False
    total_reward = 0.0
    steps = 0
    frames: List[object] = []

    while not done and steps < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        next_state, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += float(reward)
        steps += 1

    # Add one final frame for a cleaner ending.
    final_frame = env.render()
    if final_frame is not None:
        frames.append(final_frame)

    env.close()

    if not frames:
        raise RuntimeError("No frames generated for play animation.")

    imageio.mimsave(gif_path, frames, fps=fps)

    return {
        "reward": total_reward,
        "steps": steps,
        "fps": fps,
        "frames": len(frames),
        "gif_path": str(gif_path),
    }
