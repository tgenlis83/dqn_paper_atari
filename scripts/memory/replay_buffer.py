from collections import deque
from typing import Tuple, List
from utils.segment_tree import SumSegmentTree, MinSegmentTree
import numpy as np
import random
import torch

from utils.utils import convert_tuple_to_tensors

class ReplayBuffer:
    """
    Replay Buffer using preallocated NumPy arrays.

    This buffer stores experiences and samples them for training reinforcement learning agents.
    """

    def __init__(self, size: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
        """
        Initialize the Replay Buffer.

        Args:
            size (int): Maximum number of experiences the buffer can hold.
            obs_shape (Tuple[int, ...]): Shape of the observation space.
            action_shape (Tuple[int, ...]): Shape of the action space.
        """
        self.size = size
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Preallocate memory for the buffer
        self.t_obs = np.empty((size, *obs_shape), dtype=np.uint8)
        self.t1_obs = np.empty((size, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((size, *action_shape), dtype=np.uint8)
        self.rewards = np.empty(size, dtype=np.float16)
        self.dones = np.empty(size, dtype=np.bool_)

        self.idx = 0
        self.current_size = 0

    def append(self, t_obs: np.ndarray, t1_obs: np.ndarray, actions: np.ndarray, reward: float, done: bool):
        """
        Append a new transition to the replay buffer.

        Args:
            t_obs (np.ndarray): Current observation.
            t1_obs (np.ndarray): Next observation.
            actions (np.ndarray): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.t_obs[self.idx] = t_obs
        self.t1_obs[self.idx] = t1_obs
        self.actions[self.idx] = actions
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        # Update the current size and index
        self.current_size = min(self.current_size + 1, self.size)
        self.idx = (self.idx + 1) % self.size

    def get_minibatch(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a minibatch from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample in each batch.
            device (str): The device to which tensors should be moved.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            A tuple containing:
            - observations (torch.Tensor)
            - next observations (torch.Tensor)
            - actions (torch.Tensor)
            - rewards (torch.Tensor)
            - done flags (torch.Tensor)
        """
        ids = np.random.choice(self.current_size, batch_size, replace=False)
        batch = (
            self.t_obs[ids],
            self.t1_obs[ids],
            self.actions[ids],
            self.rewards[ids],
            self.dones[ids],
        )
        return convert_tuple_to_tensors(batch, device)

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer using preallocated NumPy arrays.

    This buffer stores experiences and samples them based on priority, which is useful for 
    training reinforcement learning agents. The implementation is based on the Rainbow DQN paper.
    """

    def __init__(
        self, size: int, alpha: float, obs_shape: Tuple[int, ...],
        batch_size: int, n_step: int, gamma: float
    ):
        """
        Initialize the Prioritized Replay Buffer.

        Args:
            size (int): Maximum number of experiences the buffer can hold.
            alpha (float): Priority exponent. Determines how much prioritization is used.
            obs_shape (Tuple[int, ...]): Shape of the observation space.
            batch_size (int): Number of experiences to sample in each batch.
            n_step (int): Number of steps to look ahead for N-step returns.
            gamma (float): Discount factor for future rewards.
        """
        assert alpha >= 0
        self.alpha = alpha
        self.capacity = size
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma

        # Use NumPy arrays for storage
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_obs_buf = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.n_step_buffer = deque(maxlen=n_step)

        self.max_priority = 1.0
        self.ptr = 0
        self.size = 0

        # Initialize segment trees
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: float):
        """Store experience with N-step returns.

        Args:
            obs (np.ndarray): Current observation.
            act (int): Action taken.
            rew (float): Reward received.
            next_obs (np.ndarray): Next observation.
            done (float): Whether the episode is done.
        """
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute N-step returns
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        idx = self.ptr

        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.done_buf[idx] = done

        # Update priorities in the segment trees
        self.sum_tree[idx] = self.max_priority ** self.alpha
        self.min_tree[idx] = self.max_priority ** self.alpha

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_n_step_info(self, n_step_buffer: deque, gamma: float) -> Tuple[float, np.ndarray, float]:
        """Calculate N-step reward and next observation.

        Args:
            n_step_buffer (deque): Buffer containing the N-step transitions.
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

    def sample_batch(self, beta: float, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample a batch of experiences.

        device (str): The device to which tensors should be moved.

        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]: 
        A tuple containing:
        - observations (torch.Tensor)
        - next observations (torch.Tensor)
        - actions (torch.Tensor)
        - rewards (torch.Tensor)
        - done flags (torch.Tensor)
        - importance sampling weights (torch.Tensor)
        - indices of sampled experiences (List[int])
        """
        assert self.size >= self.batch_size
        idxs = self._sample_proportional()
        weights = self._calculate_weights(idxs, beta)

        obs = self.obs_buf[idxs]
        next_obs = self.next_obs_buf[idxs]
        acts = self.acts_buf[idxs]
        rews = self.rews_buf[idxs]
        done = self.done_buf[idxs]

        return convert_tuple_to_tensors((obs, next_obs, acts, rews, done, weights, idxs), device)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities of sampled transitions.

        Args:
            indices (List[int]): Indices of the sampled transitions.
            priorities (List[float]): New priorities for the sampled transitions.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on priority proportions.

        Returns:
            List[int]: List of sampled indices.
        """
        indices = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weights(self, idxs: List[int], beta: float) -> np.ndarray:
        """Calculate the importance sampling weights.

        Args:
            idxs (List[int]): Indices of the sampled transitions.
            beta (float): Importance sampling exponent.

        Returns:
            np.ndarray: Importance sampling weights.
        """
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)
        samples_p = np.array([self.sum_tree[idx] / self.sum_tree.sum() for idx in idxs], dtype=np.float32)
        weights = (samples_p * self.size) ** (-beta)
        weights /= max_weight
        return weights

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size