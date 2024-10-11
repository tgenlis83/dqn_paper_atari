import torch
from torch import nn
from torch import Tensor


class DeepQNetwork(nn.Module):
    def __init__(self, n_actions: int):
        """
        Initialize the original Deep Q-Network. Inspired by the DQN paper.

        Args:
            n_actions (int): Number of actions in the action space.
        """
        super(DeepQNetwork, self).__init__()

        # Convolutional layers for feature extraction, same as DQN paper
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Fully connected layers for Q-value estimation
        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor representing the state.

        Returns:
            Tensor: Output tensor representing estimated Q-values for each action.
        """
        # Normalize input and pass through convolutional layers
        x = self.conv(x / 255.0)
        # Pass through fully connected layers
        return self.linear(x)
