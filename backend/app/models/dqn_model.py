"""DQN network definition for CartPole-v1.

This module only contains the model architecture.
Training logic should stay in service/training modules.
"""

import torch.nn as nn


class DQN(nn.Module):
    """Simple fully connected Q-network for CartPole.

    Input:  state of size 4
    Output: Q-values for 2 actions (left, right)
    """

    def __init__(self, input_dim: int = 4, output_dim: int = 2) -> None:
        super().__init__()

        # 4 -> 128 -> 128 -> 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        """Run a forward pass and return Q-values for each action."""
        return self.net(x)
