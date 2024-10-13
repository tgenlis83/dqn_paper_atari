from collections import deque
from typing import Tuple


class NStepTransitionBuffer:
    """Buffer to store N-step transitions."""

    def __init__(self, n_step: int, gamma: float):
        """
        Initialize the N-step buffer.

        Args:
            n_step (int): Number of steps to look ahead.
            gamma (float): Discount factor.
        """
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def store(self, transition: Tuple):
        """
        Store a transition and compute N-step returns if possible.

        Args:
            transition (Tuple): A tuple of (state, action, reward, next_state, done).

        Returns:
            Tuple or None: N-step transition or None if not enough transitions have been stored.
        """
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return None

        return self._get_n_step_transition()

    def _get_n_step_transition(self):
        """
        Get the N-step transition.

        Returns:
            Tuple: N-step transition (state, action, n_step_reward, next_state, done).
        """
        state, action = self.buffer[0][:2]
        reward, next_state, done = self._calculate_n_step_return()
        return (state, action, reward, next_state, done)

    def _calculate_n_step_return(self):
        """
        Calculate the N-step return.

        Returns:
            Tuple[float, Any, float]: N-step reward, next state, and done flag.
        """
        reward, next_state, done = self.buffer[-1][2:]
        for transition in reversed(list(self.buffer)[:-1]):
            r, n_s, d = transition[2:]
            reward = transition[2] + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def reset(self):
        """Reset the buffer."""
        self.buffer.clear()
