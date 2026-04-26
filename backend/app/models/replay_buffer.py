"""Replay buffer for DQN training.

This module stores experience tuples and provides random mini-batch sampling.
"""

import random
from collections import deque
from typing import Deque, List, Tuple


Transition = Tuple[object, int, float, object, bool]


class ReplayBuffer:
    """A simple replay buffer using deque.

    Each item is a transition:
    (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """Add one transition into the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample a mini-batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)
