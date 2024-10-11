from tqdm import tqdm
import numpy as np
from utils.training_log import TrainingLog


class ProgressBar:
    """
    A class to represent a TQDM progress bar for training.
    """

    def __init__(self, total_steps: int):
        """
        Constructs all the necessary attributes for the ProgressBar object.

        Parameters:
        -----------
        total_steps : int
            Total number of steps for the progress bar.
        """
        self.progress_bar = tqdm(range(total_steps), desc="Training Progress")

    def update(self, training_log: TrainingLog, buffer: list) -> None:
        """
        Updates the progress bar description with the latest training statistics.

        Parameters:
        -----------
        training_log : TrainingLog
            An object that logs training statistics.
        buffer : list
            A list representing the replay buffer.
        """
        desc = (
            f"R: {training_log.get_last('episode_rewards'):.2f}, "
            f"l: {training_log.get_last('loss'):.2f}, "
            f"Mean Q: {training_log.get_last('mean_q_value'):.2f}, "
            f"m50R: {np.mean(training_log.get_recent_rewards(50)):.2f}, "
            f"RBsize: {len(buffer)}"
        )
        self.progress_bar.set_description(desc)

    def __iter__(self) -> iter:
        """
        Returns an iterator for the progress bar.

        Returns:
        --------
        iter
            An iterator for the progress bar.
        """
        return iter(self.progress_bar)
