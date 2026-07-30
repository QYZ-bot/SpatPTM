from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1.0 - pt) ** self.gamma * bce).mean()


class TransformerBlock(nn.Module):
    def __init__(self, dimension: int = 256, heads: int = 8, feedforward: int = 1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.ff1 = nn.Linear(dimension, feedforward)
        self.ff2 = nn.Linear(feedforward, dimension)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        forwarded = self.ff2(F.relu(self.ff1(x)))
        return self.norm2(x + forwarded)


class SpatPTM(nn.Module):
    """Audited main architecture for 43 x 128 GATE windows."""

    def __init__(self, sequence_length: int = 43, input_dimension: int = 128):
        super().__init__()
        branch_channels = 128
        self.branch3 = nn.Sequential(nn.Conv1d(input_dimension, branch_channels, 3, padding=1), nn.ReLU())
        self.branch5 = nn.Sequential(nn.Conv1d(input_dimension, branch_channels, 5, padding=2), nn.ReLU())
        self.project = nn.Conv1d(branch_channels * 2, 256, 1)
        self.position = nn.Parameter(torch.randn(1, sequence_length, 256))
        self.transformer1 = TransformerBlock()
        self.transformer2 = TransformerBlock()
        self.attention_pool = nn.Linear(256, 1)
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(torch.cat([self.branch3(x), self.branch5(x)], dim=1))
        x = x.permute(0, 2, 1) + self.position
        x = self.transformer1(x)
        x = self.transformer2(x)
        weights = torch.softmax(self.attention_pool(x), dim=1)
        return self.classifier(torch.sum(weights * x, dim=1)).squeeze(1)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    import numpy as np

    coefficient = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    permutation = torch.randperm(x.size(0), device=device)
    return coefficient * x + (1.0 - coefficient) * x[permutation], y, y[permutation], coefficient

