import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Noisy Networks in Reinforcement Learning.

    This layer adds parameter noise to the weights and biases during training to encourage exploration.
    """

    def __init__(self, in_features: int, out_features: int, sigma_zero: float = 0.5):
        """
        Initialize the NoisyLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            sigma_zero (float): Initial value for sigma.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.zeros(out_features))

        self.sigma_zero = sigma_zero / np.sqrt(in_features)
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the layer.
        """
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_zero)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_zero)

    def reset_noise(self) -> None:
        """
        Reset the noise for the layer.
        """
        if self.training:
            # Use in-place normal distribution
            self.weight_eps.normal_()
            self.bias_eps.normal_()
        else:
            self.weight_eps.zero_()
            self.bias_eps.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

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
