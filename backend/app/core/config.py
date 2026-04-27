"""Project-level runtime/training configuration."""

from typing import Dict, Any


# Stable CartPole DQN config used by train_service.py.
DQN_TRAINING_CONFIG: Dict[str, Any] = {
    "episodes": 2000,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay": 20000,
    "epsilon_warmup_steps": 2000,
    "target_update_steps": 1000,
    "warmup_size": 1000,
    "replay_buffer_capacity": 20000,
    "checkpoint_eval_episodes": 5,
    "early_stop_avg20_threshold": 475.0,
}
