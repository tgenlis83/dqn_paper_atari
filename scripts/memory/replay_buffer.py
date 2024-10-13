from collections import deque
from typing import Tuple, List, Union

import numpy as np
import torch


class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences.
    """

    def __init__(self, size: int):
        """
        Initialize the ReplayBuffer.

        Args:
            size (int): Maximum number of experiences the buffer can hold.
        """
        self.size = size
        self.obs_buf: List[Union[np.ndarray, None]] = [None] * size
        self.next_obs_buf: List[Union[np.ndarray, None]] = [None] * size
        self.actions = np.empty(size, dtype=np.int64)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.empty(size, dtype=np.bool_)
        self.idx = 0
        self.current_size = 0

    def append(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ):
        """
        Append a new experience to the buffer.

        Args:
            obs (np.ndarray): Current observation.
            next_obs (np.ndarray): Next observation.
            action (int): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.obs_buf[self.idx] = obs
        self.next_obs_buf[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.current_size = min(self.current_size + 1, self.size)
        self.idx = (self.idx + 1) % self.size

    def get_minibatch(
        self, batch_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a minibatch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.
            device (str): Device to load the tensors on ("cpu" or "cuda").

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Batch of experiences.
        """
        ids = np.random.choice(self.current_size, batch_size, replace=False)
        obs_batch = [np.array(self.obs_buf[i], copy=False) for i in ids]
        next_obs_batch = [np.array(self.next_obs_buf[i], copy=False) for i in ids]

        # Stack the observations into a NumPy array
        obs_batch = np.stack(obs_batch, axis=0)
        next_obs_batch = np.stack(next_obs_batch, axis=0)

        # Convert to PyTorch tensors
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        next_obs_batch = torch.tensor(
            next_obs_batch, dtype=torch.float32, device=device
        )
        actions = torch.tensor(self.actions[ids], dtype=torch.long, device=device)
        rewards = torch.tensor(self.rewards[ids], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones[ids], dtype=torch.float32, device=device)

        return obs_batch, next_obs_batch, actions, rewards, dones

    def __len__(self) -> int:
        """
        Return the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return self.current_size


class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer without internal N-step buffer.
    """

    def __init__(self, size: int, alpha: float, batch_size: int):
        """
        Initialize the Prioritized Replay Buffer.

        Args:
            size (int): Maximum number of experiences the buffer can hold.
            alpha (float): Priority exponent.
            batch_size (int): Number of experiences to sample in each batch.
        """
        assert alpha >= 0
        self.alpha = alpha
        self.capacity = size
        self.batch_size = batch_size

        # Use NumPy arrays for storage
        self.obs_buf: List[Union[np.ndarray, None]] = [None] * size
        self.next_obs_buf: List[Union[np.ndarray, None]] = [None] * size
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.priorities = np.zeros((size,), dtype=np.float32)
        self.max_priority = 1.0
        self.ptr = 0
        self.size = 0

    def store(
        self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool
    ):
        """
        Store a new experience in the buffer.

        Args:
            obs (np.ndarray): Current observation.
            act (int): Action taken.
            rew (float): Reward received.
            next_obs (np.ndarray): Next observation.
            done (bool): Whether the episode is done.
        """
        idx = self.ptr

        # Store the experience
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.done_buf[idx] = done

        # Update priorities
        self.priorities[idx] = self.max_priority**self.alpha

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_n_step_info(
        self, n_step_buffer: deque, gamma: float
    ) -> Tuple[float, np.ndarray, float]:
        """
        Calculate N-step reward and next observation.

        Args:
            n_step_buffer (deque): Buffer containing N-step transitions.
            gamma (float): Discount factor.

        Returns:
            Tuple[float, np.ndarray, float]: N-step reward, next observation, and done flag.
        """
        rew, next_obs, done = n_step_buffer[-1][2:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[2:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        return rew, next_obs, done

    def sample_batch(self, beta: float, device: str) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
    ]:
        """
        Sample a batch of experiences.

        Args:
            beta (float): Importance sampling exponent.
            device (str): Device to load the tensors on ("cpu" or "cuda").

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]: Batch of experiences and their indices.
        """
        assert self.size >= self.batch_size

        # Calculate probabilities
        priorities = self.priorities[: self.size]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, self.batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Convert observations to tensors
        obs = np.array([np.array(self.obs_buf[idx], copy=False) for idx in indices])
        next_obs = np.array(
            [np.array(self.next_obs_buf[idx], copy=False) for idx in indices]
        )

        obs = torch.tensor(obs, dtype=torch.uint8, device=device)
        next_obs = torch.tensor(next_obs, dtype=torch.uint8, device=device)
        acts = torch.tensor(self.acts_buf[indices], dtype=torch.int64, device=device)
        rews = torch.tensor(self.rews_buf[indices], dtype=torch.float32, device=device)
        done = torch.tensor(self.done_buf[indices], dtype=torch.float32, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        return obs, next_obs, acts, rews, done, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities of sampled transitions.

        Args:
            indices (np.ndarray): Indices of the sampled transitions.
            priorities (np.ndarray): New priorities of the sampled transitions.
        """
        assert len(indices) == len(priorities)
        self.priorities[indices] = priorities**self.alpha
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        """
        Return the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return self.size
