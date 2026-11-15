"""Lightweight 1D convolutional vibration encoder and source classifier."""
from __future__ import annotations

import torch
from torch import nn


class WaveEncoder(nn.Module):
    """Three-stage 1D Conv + GroupNorm(8) + GELU encoder producing 64-D embeddings.

    Deployment parameters (classification head removed):
    - 1-channel input (JNU, CWRU, SelfA): 68,800 parameters
    - 3-channel input (HUST):             69,760 parameters
    """

    def __init__(self, channels: int = 1, feature_dim: int = 64):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(1, channels, 1))
        self.register_buffer("input_scale", torch.ones(1, channels, 1))
        layers: list[nn.Module] = []
        in_channels = channels
        for width, kernel in ((32, 15), (64, 9), (128, 5)):
            layers += [
                nn.Conv1d(in_channels, width, kernel, stride=2, padding=kernel // 2),
                nn.GroupNorm(8, width),
                nn.GELU(),
            ]
            in_channels = width
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.input_mean.shape[1]:
            raise ValueError("Expected x of shape [batch, channels, length].")
        x = (x - self.input_mean) / self.input_scale
        return self.projection(self.pool(self.blocks(x)).squeeze(-1))


class SourceClassifier(nn.Module):
    """Source-trained classifier wrapping ``WaveEncoder`` and a linear classification head."""

    def __init__(self, channels: int, feature_dim: int, num_classes: int):
        super().__init__()
        self.encoder = WaveEncoder(channels, feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
