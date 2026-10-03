"""
BioFusionFormer - Pathway-Aware Cross-Modal Transformer
=======================================================
Architecture
------------
                    Input tensors (N, F_mod) for each modality
                           |
             ┌─────────────────────────────┐
             │  ModalityProjection × 3     │  masked linear -> PathwayEmbed (N, P, d)
             └──────────────┬──────────────┘
                            │ Stack -> (N, M, P, d_model)
                            │ + modality embeddings
                            │
             ┌──────────────▼──────────────┐
             │  GatedModalityFusion         │  softmax gate over M modalities -> (N, P, d)
             └──────────────┬──────────────┘
                            │
             ┌──────────────▼──────────────┐
             │  PathwayTransformerLayer × L │  multi-head self-attention + FFN
             │  (with pathway topology bias)│  (N, P, d_model)
             └──────────────┬──────────────┘
                            │ mean-pool over P pathways
                            │
             ┌──────────────▼──────────────┐
             │  Classifier MLP             │  (N, d_model) -> (N, 1)
             └─────────────────────────────┘

Parameter budget (default config for TARGET-WT N=121)
-----------------------------------------------------
  d_model = 32, n_heads = 4, n_layers = 2, n_pathways ≤ 3368
  Projection weights are sparse (masked); only sum(mask) are trainable.
  Target: < 500k total parameters.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_projection import MaskedLinear, ModalityProjection
from .pathway_bias import PathwayBiasBuilder


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with additive topology bias
# ---------------------------------------------------------------------------

class BiasedMultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention with an additive pre-computed bias.

    The bias is added to the raw attention logits before softmax, encoding
    known pathway relationships (parent-child, modality co-annotation).

    Parameters
    ----------
    d_model   : int     embedding dimension
    n_heads   : int     number of attention heads
    dropout   : float   attention dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self._scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x    : (N, P, d_model)
        bias : (P, P) additive topology bias, or None

        Returns
        -------
        out  : (N, P, d_model)
        attn : (N, n_heads, P, P) softmax attention weights (detached, for explainability)
        """
        N, P, _ = x.shape
        H, d_h  = self.n_heads, self.d_head

        # Project and reshape to (N, H, P, d_h)
        Q = self.q_proj(x).view(N, P, H, d_h).transpose(1, 2)
        K = self.k_proj(x).view(N, P, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(N, P, H, d_h).transpose(1, 2)

        # Scaled dot-product attention logits (N, H, P, P)
        logits = (Q @ K.transpose(-2, -1)) / self._scale

        # Add topology bias (broadcast across batch and heads)
        if bias is not None:
            logits = logits + bias.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # (N, H, P, d_h) -> (N, P, d_model)
        out = (attn_weights @ V).transpose(1, 2).contiguous().view(N, P, -1)
        out = self.resid_drop(self.o_proj(out))

        return out, attn_weights.detach()


# ---------------------------------------------------------------------------
# Position-wise FFN
# ---------------------------------------------------------------------------

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Pathway Transformer Layer = Attention + FFN with pre-LayerNorm
# ---------------------------------------------------------------------------

class PathwayTransformerLayer(nn.Module):
    """
    Pre-LayerNorm transformer layer (more stable training than post-LN).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = BiasedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm attention (residual connection)
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, bias)
        x = x + attn_out

        # Pre-norm FFN (residual connection)
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights


# ---------------------------------------------------------------------------
# Gated Modality Fusion
# ---------------------------------------------------------------------------

class GatedModalityFusion(nn.Module):
    """
    Soft gating mechanism over M modalities.
    Gate scores are patient-specific (learned function of the modality embeddings).

    Input  : (N, M, P, d_model)
    Output : (N, P, d_model)
    Gate   : (N, M) returned for downstream explainability
    """

    def __init__(self, n_modalities: int, d_model: int):
        super().__init__()
        # Global-pooled modality embeddings -> gate logit
        self.gate_proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (N, M, P, d_model)
        """
        N, M, P, D = x.shape
        # Pathway-mean pooling -> (N, M, d_model)
        pooled = x.mean(dim=2)   # (N, M, d_model)
        # Gate logit per modality -> (N, M, 1) -> (N, M)
        gate_logits = self.gate_proj(pooled).squeeze(-1)   # (N, M)
        gate = F.softmax(gate_logits, dim=-1)              # (N, M)

        # Weighted sum over modalities: gate (N, M) x (N, M, P, D)
        gate_expanded = gate.unsqueeze(-1).unsqueeze(-1)   # (N, M, 1, 1)
        fused = (x * gate_expanded).sum(dim=1)             # (N, P, D)

        return fused, gate


# ---------------------------------------------------------------------------
# Full BioFusionFormer Model
# ---------------------------------------------------------------------------

class BioFusionFormer(nn.Module):
    """
    Pathway-Aware Hierarchical Cross-Modal Transformer.

    Parameters
    ----------
    modality_features : dict[str, int]
        Keys are modality names (e.g. "mrna", "mirna", "meth"),
        values are the number of features for that modality.
    modality_masks    : dict[str, torch.BoolTensor]
        One mask per modality, shape (n_pathways, n_features).
        Mask[i, j] = True if feature j belongs to pathway i.
    n_pathways        : int
        Number of Reactome pathways (dimension of pathway space).
    d_model           : int (default 32)
        Transformer embedding dimension.
    n_heads           : int (default 4)
        Number of attention heads.
    n_layers          : int (default 2)
        Number of transformer encoder layers.
    d_ff              : int (default 128)
        Hidden dimension in position-wise FFN.
    dropout           : float (default 0.1)
    topology_bias     : torch.Tensor | None
        Precomputed (n_pathways, n_pathways) additive bias matrix.
    """

    MODALITY_ORDER = ["mrna", "mirna", "meth"]

    def __init__(
        self,
        modality_features: Dict[str, int],
        modality_masks: Dict[str, torch.Tensor],
        n_pathways: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        topology_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.d_model    = d_model
        self.modality_names = [m for m in self.MODALITY_ORDER if m in modality_features]
        M = len(self.modality_names)

        # ---- Modality projections (masked) ----
        self.projections = nn.ModuleDict()
        for mod in self.modality_names:
            self.projections[mod] = ModalityProjection(
                in_features=modality_features[mod],
                n_pathways=n_pathways,
                mask=modality_masks[mod].bool(),
                dropout=dropout,
            )

        # ---- Pathway-level linear embedding (pathway -> d_model) ----
        # After projection each modality gives (N, P) pathway activation;
        # we add a per-pathway linear map to d_model dimensions.
        self.pathway_embed = nn.Linear(1, d_model, bias=True)

        # ---- Learnable modality type embeddings ----
        self.modality_embed = nn.Embedding(M, d_model)

        # ---- Gated modality fusion ----
        self.fusion = GatedModalityFusion(M, d_model)

        # ---- Transformer encoder ----
        self.layers = nn.ModuleList([
            PathwayTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ---- Topology bias (non-trainable buffer) ----
        if topology_bias is not None:
            self.register_buffer("topology_bias", topology_bias.float())
        else:
            self.register_buffer("topology_bias", None)

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, zero bias for classifier output."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and not isinstance(module, MaskedLinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Encode multi-modal inputs to a single fused representation.

        Parameters
        ----------
        inputs : dict[str, Tensor(N, F_mod)]

        Returns
        -------
        x      : (N, P, d_model) after transformer
        gate   : (N, M) modality gate weights
        attns  : list of (N, H, P, P) per layer
        """
        N = next(iter(inputs.values())).shape[0]
        M = len(self.modality_names)

        # ---- Project each modality to pathway space ----
        # Each projection: (N, P) -> unsqueeze(-1) -> (N, P, 1) -> linear -> (N, P, d_model)
        modality_tensors = []
        for i, mod in enumerate(self.modality_names):
            proj = self.projections[mod](inputs[mod])      # (N, P)
            emb  = self.pathway_embed(proj.unsqueeze(-1))  # (N, P, d_model)
            emb  = emb + self.modality_embed(
                torch.tensor(i, device=proj.device)
            )                                              # add modality type embedding
            modality_tensors.append(emb)                   # (N, P, d_model)

        # Stack -> (N, M, P, d_model)
        stacked = torch.stack(modality_tensors, dim=1)

        # ---- Gated fusion -> (N, P, d_model) ----
        x, gate = self.fusion(stacked)

        # ---- Transformer encoder ----
        attns = []
        for layer in self.layers:
            x, attn = layer(x, self.topology_bias)
            attns.append(attn)

        x = self.final_norm(x)
        return x, gate, attns

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : dict[str, Tensor(N, F_mod)]

        Returns
        -------
        dict with:
          'logit' : (N,) raw binary logit
          'gate'  : (N, M) modality gate weights
          'attns' : list[(N, H, P, P)] per-layer attention maps
        """
        x, gate, attns = self.encode(inputs)

        # Mean-pool over pathway tokens -> (N, d_model)
        pooled = x.mean(dim=1)

        # Classification
        logit = self.classifier(pooled).squeeze(-1)  # (N,)

        return {"logit": logit, "gate": gate, "attns": attns}

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per component."""
        groups = {
            "projections": sum(p.numel() for p in self.projections.parameters()),
            "pathway_embed": sum(p.numel() for p in self.pathway_embed.parameters()),
            "modality_embed": sum(p.numel() for p in self.modality_embed.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "transformer": sum(p.numel() for layer in self.layers for p in layer.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
        }
        groups["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return groups
