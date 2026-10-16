"""
src/encoders/clinical_encoder.py
================================
Clinical feature encoder for BioFusionFormer (Stage 5.2).

Maps baseline clinical features (age, sex, stage) to ONE clinical token
of shape (B, 1, d_model) via a 2-layer MLP with LayerNorm and Dropout (>= 0.3).
"""

from typing import Optional
import torch
import torch.nn as nn


class ClinicalFeatureEncoder(nn.Module):
    """
    2-layer MLP with LayerNorm and Dropout >= 0.3 projecting raw clinical features to token dimension d_model.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 32,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2

        if dropout < 0.3:
            dropout = 0.3

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)
        Returns: (B, 1, d_model)
        """
        token = self.net(x)  # (B, d_model)
        return token.unsqueeze(1)  # (B, 1, d_model)
