import os
from typing import Dict, List

import pandas as pd


class TrainingLog:
    """
    A class for logging training data and saving it to a CSV file.
    """

    def __init__(self, experiment_name: str):
        """
        Initializes a new TrainingLog instance.

        Args:
            experiment_name (str): The name of the experiment.
        """
        self.experiment_name = experiment_name
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_loss = 0
        self.episode_q_values = 0
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "mean_q_value": [],
            "episode_rewards": [],
            "steps": [],
        }
        self.last_save_index: int = 0

    def log_episode(self, t: int) -> None:
        """
        Logs the details of an episode.

        Args:
            t (int): The current timestep.
        """
        self.history["steps"].append(t)
        self.history["episode_rewards"].append(self.episode_reward)
        self.history["mean_q_value"].append(
            self.episode_q_values / self.episode_steps if self.episode_steps > 0 else 0
        )
        self.history["loss"].append(
            self.episode_loss / self.episode_steps if self.episode_steps > 0 else 0
        )
    
    def reset_episode(self) -> None:
        """
        Resets the episode-specific statistics.
        """
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_loss = 0
        self.episode_q_values = 0

    def save(self) -> None:
        """
        Saves the logged data to a CSV file.
        """
        df = pd.DataFrame(
            {
                "loss": self.history["loss"][self.last_save_index :],
                "mean_q_value": self.history["mean_q_value"][self.last_save_index :],
                "episode_rewards": self.history["episode_rewards"][
                    self.last_save_index :
                ],
                "steps": self.history["steps"][self.last_save_index :],
            }
        )
        df.to_csv(
            f"{self.experiment_name}_training_history.csv",
            mode="a",
            header=not os.path.exists(f"{self.experiment_name}_training_history.csv"),
            index=False,
        )
        self.last_save_index = len(self.history["loss"])

    def get_recent_rewards(self, n: int) -> List[float]:
        """
        Retrieves the most recent episode rewards.

        Args:
            n (int): The number of recent rewards to retrieve.

        Returns:
            List[float]: A list of the most recent episode rewards.
        """
        return (
            self.history["episode_rewards"][-n:]
            if len(self.history["episode_rewards"]) >= n
            else self.history["episode_rewards"]
        )

    def get_last(self, key: str) -> float:
        """
        Retrieves the last logged value for a given key.

        Args:
            key (str): The key to retrieve the last value for.

        Returns:
            float: The last logged value for the specified key, or 0 if no values are logged.
        """
        return self.history[key][-1] if self.history[key] else 0
