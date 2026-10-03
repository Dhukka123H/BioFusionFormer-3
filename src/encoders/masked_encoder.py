"""
BioFusionFormer - Modality Encoders
===================================
Sparse, biology-constrained projections from omics features (genes, miRNAs, CpGs)
to pathway token representations.

Enforces that a weight W[pathway_i, feature_j] is only trainable if feature_j
is an annotated member of pathway_i according to the Reactome knowledgebase.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """
    Linear projection layer constrained by a binary biological prior mask.
    Only weights where mask[i, j] == 1 are trainable; all other weights are strictly 0.0.
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True):
        super().__init__()
        assert mask.shape == (out_features, in_features), (
            f"Mask shape {mask.shape} does not match (out={out_features}, in={in_features})"
        )
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", mask.float())

        # Initialize weights small inside the mask (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        with torch.no_grad():
            self.weight.mul_(self.mask)

        # Hook ensures gradient is zero for masked-out entries
        self.weight.register_hook(lambda grad: grad * self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        active = int(self.mask.sum().item())
        total = self.in_features * self.out_features
        return f"in={self.in_features}, out={self.out_features}, active={active}/{total} ({100*active/total:.1f}%)"


class PathwayModalityEncoder(nn.Module):
    """
    Encodes one omics modality into pathway token vectors.
    Architecture:
      x (N, F_mod)
        -> MaskedLinear(F_mod, P)
        -> LayerNorm(P)
        -> GELU
        -> Dropout
        -> Unfold to (N, P, 1) -> Linear(1, d_model) -> (N, P, d_model)
    """

    def __init__(self, in_features: int, n_pathways: int, mask: torch.Tensor, d_model: int = 32, dropout: float = 0.3):
        super().__init__()
        self.masked_linear = MaskedLinear(in_features, n_pathways, mask=mask)
        self.norm = nn.LayerNorm(n_pathways)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.token_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, F_mod) -> (N, P)
        p_act = self.drop(self.act(self.norm(self.masked_linear(x))))
        # (N, P) -> (N, P, 1) -> (N, P, d_model)
        tokens = self.token_proj(p_act.unsqueeze(-1))
        return tokens
