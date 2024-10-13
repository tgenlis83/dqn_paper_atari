import gymnasium as gym
from typing import Any, Tuple


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Make end-of-life equals end-of-episode, BUT only reset on TRUE game over.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super().__init__(env)
        self.lives: int = 0
        self.was_real_done: bool = True

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        """
        Step the environment with the given action.

        Args:
            action (int): The action to take.

        Returns:
            Tuple containing:
                - obs (Any): The observation after taking the action.
                - reward (float): The reward received after taking the action.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode was truncated.
                - info (dict): Additional information about the step.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        # Check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # For Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        """
        Reset the environment. Only reset when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        Args:
            **kwargs: Additional arguments for the environment reset.

        Returns:
            Tuple containing:
                - obs (Any): The initial observation.
                - info (dict): Additional information about the reset.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
