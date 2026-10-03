"""
BioFusionFormer - Masked Pathway Projection Layer
==================================================
Maps raw feature vectors (mRNA / miRNA / methylation) onto a fixed set of
Reactome pathways using a *masked* linear projection.

The mask M[i, j] = 1 iff feature j is annotated to pathway i.
Only non-zero weights can receive gradient, enforcing biological sparsity.
The mask is registered as a buffer (not a parameter) so it stays on device
and is saved with the model state.

Design constraints
------------------
* No information leakage: the mask is derived purely from prior biological
  annotation (Reactome gene-set membership), NOT from the data labels.
* Modular: each modality has its own MaskedProjection that maps its features
  to the shared pathway embedding space (dim = n_pathways).
* Parameter budget: weights are n_features × n_pathways but only
  sum(mask) weights are effectively trainable via SGD; the zero-masked
  weights are multiplied out after every gradient step (hook).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedLinear(nn.Module):
    """
    Linear layer whose weight matrix is element-wise masked.

    Parameters
    ----------
    in_features  : int
        Number of input features (e.g. mRNA genes).
    out_features : int
        Number of output dimensions (= number of pathways).
    mask         : torch.BoolTensor, shape (out_features, in_features)
        True wherever a weight is *allowed* to be non-zero.
    bias         : bool, optional (default True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.BoolTensor,
        bias: bool = True,
    ):
        super().__init__()
        assert mask.shape == (out_features, in_features), (
            f"Mask shape {mask.shape} must be (out={out_features}, in={in_features})"
        )
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight — same storage as a plain Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Non-trainable binary mask
        self.register_buffer("mask", mask.float())

        # Initialise only the unmasked entries with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.weight.mul_(self.mask)   # zero-out masked weights at init

        # Register a post-gradient hook so masked weights stay zero
        self.weight.register_hook(self._grad_mask_hook)

    def _grad_mask_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero out gradients for masked (disallowed) weights."""
        return grad * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask before every forward pass to prevent drift
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        n_active = int(self.mask.sum().item())
        n_total = self.in_features * self.out_features
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"active_weights={n_active}/{n_total} "
            f"({100*n_active/n_total:.1f}%)"
        )


class ModalityProjection(nn.Module):
    """
    Projects a single omics modality into pathway space.

    Architecture
    ------------
    x (N, F_mod)
      -> MaskedLinear(F_mod, P)      # biology-constrained pathway scores
      -> LayerNorm(P)
      -> GELU
      -> Dropout
    out (N, P)                        # P = n_pathways
    """

    def __init__(
        self,
        in_features: int,
        n_pathways: int,
        mask: torch.BoolTensor,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.masked_linear = MaskedLinear(in_features, n_pathways, mask)
        self.norm = nn.LayerNorm(n_pathways)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.masked_linear(x))))
