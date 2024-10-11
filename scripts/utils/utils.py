import random

import gymnasium as gym
import numpy as np
import torch


def select_device() -> str:
    """
    Select the best available device (GPU or CPU).
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def make_env(env_id: str, render_mode: str = None, frame_skip: int = 4) -> gym.Env:
    """
    Create an Atari environment with preprocessing wrappers.

    Args:
        env_id (str): The environment ID to create.
        render_mode (str, optional): The mode to render the environment. Defaults to None.
        frame_skip (int, optional): The number of frames to skip. Defaults to 4.

    Returns:
        gym.Env: The wrapped Atari environment.
    """
    # Create the base environment
    env = gym.make(
        env_id, render_mode=render_mode, frameskip=1, max_episode_steps=108_000
    )

    # Apply Atari preprocessing wrapper with frame skip and terminal on life loss as advised in the Rainbow paper
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=frame_skip, terminal_on_life_loss=True
    )

    # Record episode statistics like rewards, lives, steps, etc.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Stack frames to create a single observation and improve temporal resolution
    env = gym.wrappers.FrameStack(env, 4)

    return env

def convert_tuple_to_tensors(batch: tuple, device: str) -> tuple:
    """
    Convert a tuple of data to tensors and move them to the specified device.

    Args:
        batch (tuple): A tuple containing data to be converted to tensors.
        device (str): The device to move the tensors to (e.g., 'cpu', 'cuda', 'mps').

    Returns:
        tuple: A tuple containing the data as tensors on the specified device.
    """
    return tuple(
        torch.as_tensor(item, dtype=torch.float32).to(device) for item in batch
    )