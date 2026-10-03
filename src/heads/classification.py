"""
BioFusionFormer - Classification Head & Loss
============================================
Binary classification head mapping fused patient representation to logit,
paired with label-smoothed binary cross-entropy loss with positive class weighting.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    MLP head: (B, d_model) -> (B, 1) logit.
    """

    def __init__(self, in_features: int = 32, hidden_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LabelSmoothedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing (targets smoothed: 0 -> eps/2, 1 -> 1 - eps/2)
    and positive class weight for class imbalance.
    """

    def __init__(self, eps: float = 0.05, pos_weight: Optional[float] = None):
        super().__init__()
        self.eps = eps
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed_targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets, pos_weight=pw)
