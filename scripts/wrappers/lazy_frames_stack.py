import gymnasium as gym
from collections import deque
from memory.lazy_frames import LazyFrames
import numpy as np
from typing import Any, Tuple, Dict


class LazyFramesStack(gym.Wrapper):
    """
    A gymnasium wrapper that stacks a specified number of frames and returns them as LazyFrames.

    Attributes:
        env (gym.Env): The environment to wrap.
        num_stack (int): The number of frames to stack.
        frames (deque): A deque to store the stacked frames.
        observation_space (gym.spaces.Box): The observation space of the wrapped environment.
    """

    def __init__(self, env: gym.Env, num_stack: int):
        """
        Initialize the LazyFramesStack wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            num_stack (int): The number of frames to stack.
        """
        super(LazyFramesStack, self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(num_stack, *shp), dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs: Any) -> Tuple[LazyFrames, Dict[str, Any]]:
        """
        Reset the environment and stack the initial frames.

        Args:
            **kwargs: Additional arguments for the environment's reset method.

        Returns:
            Tuple[LazyFrames, Dict[str, Any]]: The stacked frames and additional info.
        """
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs)  # Ensure it's a NumPy array
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return LazyFrames(list(self.frames)), info

    def step(self, action: int) -> Tuple[LazyFrames, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment and stack the resulting frame.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[LazyFrames, float, bool, bool, Dict[str, Any]]: The stacked frames, reward, done flag, truncated flag, and additional info.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        obs = np.array(obs)
        self.frames.append(obs)
        return LazyFrames(list(self.frames)), reward, done, truncated, info
