import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.noisylinear import NoisyLinear


class RainbowDeepQNetwork(nn.Module):
    """
    A PyTorch implementation of the Rainbow Deep Q-Network (DQN) for reinforcement learning.
    This network combines several improvements over the traditional DQN, including:
    - Distributional Q-learning with categorical distribution of Q-values.
    - Dueling network architecture with separate value and advantage streams.
    - Noisy linear layers for exploration.
    """

    def __init__(
        self, v_min: float, v_max: float, n_atoms: int, n_actions: int, device: str
    ):
        """
        Initializes the RainbowDeepQNetwork.

        Args:
            v_min (float): Minimum value of the support for the distributional Q-learning.
            v_max (float): Maximum value of the support for the distributional Q-learning.
            n_atoms (int): Number of atoms in the distributional representation of Q-values.
            n_actions (int): Number of possible actions.
            device (str): Device to run the network on.
        """
        self.v_min = v_min
        self.v_max = v_max

        super(RainbowDeepQNetwork, self).__init__()

        # Convolutional layers for processing the input frames
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.device = device
        self.support = torch.linspace(v_min, v_max, n_atoms, device=device)

        # Dimensions after the convolutional layers
        self.fc_input_dim = 7 * 7 * 64
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        # Dueling network architecture
        # Value stream (one for all actions)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, self.n_atoms),  # Output N_ATOMS for the value
        )

        # Advantage stream (one for each action)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions * self.n_atoms),  # N_ATOMS per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing the Q-value distributions for each action.
        """
        # Normalize input and pass through convolutional layers
        x = self.conv(x.float() / 255.0)

        # Get value and advantage streams
        value = self.value_stream(x).unsqueeze(1)  # Shape: (batch_size, 1, N_ATOMS)
        advantage = self.advantage_stream(x).view(
            -1, self.n_actions, self.n_atoms
        )  # Shape: (batch_size, n_actions, N_ATOMS)

        # Combine streams into Q-values
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Apply softmax to the Q-value distributions to get probabilities over atoms
        q_atoms = F.softmax(q_atoms, dim=2)  # Apply softmax once over atoms dimension
        q_atoms = q_atoms.clamp(min=1e-3)  # for avoiding nans

        return q_atoms  # Shape: (batch_size, n_actions, N_ATOMS)

    def reset_noise(self) -> None:
        """
        Resets the noise for all NoisyLinear layers in the network.
        """
        # Reset noise for all NoisyLinear layers
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """
        Projects the distribution of the next state value distribution onto the current state.

        Args:
            next_dist (torch.Tensor): Distribution of the next state Q-values.
            rewards (torch.Tensor): Rewards received after taking the action.
            dones (torch.Tensor): Binary tensor indicating if the episode has ended.
            discount (float): Discount factor for future rewards.

        Returns:
            torch.Tensor: Projected distribution for the current state.
        """
        batch_size = rewards.size(0)
        delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        support = self.support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + discount * support * (1 - dones)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, self.n_atoms - 1)
        u = u.clamp(0, self.n_atoms - 1)

        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.n_atoms, batch_size, device=self.device
            )
            .long()
            .unsqueeze(1)
        )
        proj_dist = torch.zeros_like(next_dist)

        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist
