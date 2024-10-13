import random

import gymnasium as gym
import numpy as np
import torch

from wrappers.lazy_frames_stack import LazyFramesStack
from wrappers.episodic_life_env import EpisodicLifeEnv

import json
import argparse


def train_parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN on Atari games given a JSON experiment file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the .json config file."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["double", "rainbow"],
        required=True,
        help="Type of model to test.",
    )
    return parser.parse_args()


def test_parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN on Atari games given a JSON experiment file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the .json config file."
    )
    parser.add_argument(
        "--num_envs", type=int, default=32, help="Number of environments."
    )
    parser.add_argument(
        "--warmup_episodes", type=int, default=100, help="Number of warmup episodes."
    )
    parser.add_argument(
        "--testing_episodes", type=int, default=500, help="Number of testing episodes."
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        required=True,
        help="Regex path to the checkpoints (eg. ./ckpt*.pt).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["double", "rainbow"],
        required=True,
        help="Type of model to test.",
    )

    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


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
    env = EpisodicLifeEnv(env)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=frame_skip, terminal_on_life_loss=True
    )

    # Record episode statistics like rewards, lives, steps, etc.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Stack frames to create a single observation and improve temporal resolution
    env = LazyFramesStack(env, 4)

    return env
