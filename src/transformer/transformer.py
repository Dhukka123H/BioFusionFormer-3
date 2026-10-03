"""
BioFusionFormer - Pathway-Aware Cross-Modal Transformer
=======================================================
Architecture:
  - Token sequence: [CLS] + M modalities x P pathways (Total length = 1 + M * P)
  - Learned modality embeddings added per modality token
  - 2 Transformer encoder layers with pre-LayerNorm
  - Multi-head attention with additive biological attention bias and learnable scale
  - Soft biological prior:
      (i) cross-modal same-pathway attention
      (ii) hierarchical parent-child DAG relations from Reactome
      (iii) bi-directional attention to/from [CLS] token
  - Flag `use_bias=False` allows unconstrained full self-attention (Ablation E4)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasedCrossModalAttention(nn.Module):
    """
    Multi-head self-attention with pre-computed additive biological bias and learnable scale.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, attn_dropout: float = 0.2, dropout: float = 0.3):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

        # Learnable scale for biological attention bias
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        bio_bias: Optional[torch.Tensor] = None,
        use_bias: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, S, d_model) where S = 1 + M * P
        bio_bias: (S, S) additive biological affinity matrix
        """
        B, S, _ = x.shape
        H, d_h = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, S, H, d_h).transpose(1, 2)  # (B, H, S, d_h)
        K = self.k_proj(x).view(B, S, H, d_h).transpose(1, 2)  # (B, H, S, d_h)
        V = self.v_proj(x).view(B, S, H, d_h).transpose(1, 2)  # (B, H, S, d_h)

        logits = (Q @ K.transpose(-2, -1)) / self.scale  # (B, H, S, S)

        if use_bias and bio_bias is not None:
            # Broadcast bio_bias across batch and heads with learnable scale
            scaled_bias = self.bias_scale * bio_bias.unsqueeze(0).unsqueeze(0)
            logits = logits + scaled_bias

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)

        out = (attn_weights_dropped @ V).transpose(1, 2).contiguous().view(B, S, -1)
        out = self.resid_drop(self.out_proj(out))
        return out, attn_weights.detach()


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block (Norm -> Attention -> Resid -> Norm -> FFN -> Resid).
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 128, attn_dropout: float = 0.2, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BiasedCrossModalAttention(d_model, n_heads, attn_dropout=attn_dropout, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, bio_bias: Optional[torch.Tensor] = None, use_bias: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.norm1(x), bio_bias=bio_bias, use_bias=use_bias)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class PathwayCrossModalTransformer(nn.Module):
    """
    Stack of 2 TransformerBlocks operating over [CLS] and modality-pathway tokens.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        attn_dropout: float = 0.2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, attn_dropout, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        bio_bias: Optional[torch.Tensor] = None,
        use_bias: bool = True,
    ) -> Tuple[torch.Tensor, list]:
        attns = []
        for layer in self.layers:
            x, w = layer(x, bio_bias=bio_bias, use_bias=use_bias)
            attns.append(w)
        x = self.final_norm(x)
        return x, attns
