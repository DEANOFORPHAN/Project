# CartPole DQN Hyperparameters (Current Stable Set)

This document records the current training hyperparameters used in:

- `backend/app/core/config.py` (`DQN_TRAINING_CONFIG`)
- `backend/app/services/train_service.py`

## Current Values

```text
episodes: 2000
learning_rate: 3e-4
batch_size: 64
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.02
epsilon_decay: 20000
epsilon_warmup_steps: 2000
target_update_steps: 1000
warmup_size: 1000
replay_buffer_capacity: 20000
checkpoint_eval_episodes: 5
early_stop_avg20_threshold: 475.0
```

## Notes

- Checkpoint selection is based on greedy evaluation average (`checkpoint_eval_episodes`), not single training-episode reward.
- Best model is saved to `backend/app/outputs/best_model.pth`.
- During training, artifacts are exported to `backend/app/outputs/`:
  - `rewards.csv`
  - `reward_curve.png`
  - `metrics.json`

## How To Change

Edit only:

- `backend/app/core/config.py`

No need to modify `train_service.py` for routine hyperparameter tuning.
