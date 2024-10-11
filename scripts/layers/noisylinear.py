import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize the NoisyLinear layer. This linear layer adds noise to the weights for exploration.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            std_init (float): Initial standard deviation for the noise.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Mean and sigma parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """
        Initialize the parameters of the layer.
        """
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self) -> None:
        """
        Reset the noise for the layer.
        """
        epsilon_in = self._scale_noise(self.in_features).to(self.weight_epsilon.device)
        epsilon_out = self._scale_noise(self.out_features).to(
            self.weight_epsilon.device
        )

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def _scale_noise(self, size: int) -> Tensor:
        """
        Generate scaled noise.

        Args:
            size (int): Size of the noise tensor.

        Returns:
            Tensor: Scaled noise tensor.
        """
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
