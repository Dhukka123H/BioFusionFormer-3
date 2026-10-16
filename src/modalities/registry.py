"""
src/modalities/registry.py
==========================
Modality Registry & Missing Modality Wrapper (Stage 6.5).

Provides:
  1. ModalitySpec and ModalityRegistry: Extensible registry for multi-modal architectures
     (mRNA, miRNA, methylation, clinical EHR, future CT/MRI imaging).
  2. ImagingFeatureEncoder: Reference implementation for pre-extracted imaging features/patches -> (B, T, d_model).
  3. RobustCrossModalTransformerWrapper: Opt-in key-padding mask transformer wrapper
     that handles missing modalities without altering the frozen E1 code.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.transformer.transformer import PathwayCrossModalTransformer


@dataclass
class ModalitySpec:
    """Specification of a modality registered in the pipeline."""
    name: str
    token_count: int
    modality_type: str  # "omics", "clinical", "imaging"
    encoder_factory: Optional[Callable[..., nn.Module]] = None
    description: str = ""


class ModalityRegistry:
    """Registry for discovering and managing modalities."""

    def __init__(self):
        self._specs: Dict[str, ModalitySpec] = {}

    def register(self, spec: ModalitySpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Modality '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModalitySpec:
        if name not in self._specs:
            raise KeyError(f"Modality '{name}' not found in registry. Registered: {list(self._specs.keys())}")
        return self._specs[name]

    def list_modalities(self) -> List[str]:
        return list(self._specs.keys())


class ImagingFeatureEncoder(nn.Module):
    """
    Reference imaging encoder for pre-extracted image feature vectors or patch embeddings (e.g. from CT/MRI).
    Returns tokens (B, T, d_model) and an availability mask (B,) where 1 = available, 0 = missing.
    """

    def __init__(
        self,
        in_dim: int = 512,
        n_patches: int = 16,
        d_model: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_patches = n_patches
        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, in_dim) tensor if present, or None if absent across the batch.
        Returns:
            tokens: (B, T, d_model)
            availability_mask: (B,) boolean tensor (True = present, False = missing)
        """
        if x is not None:
            B, T, _ = x.shape
            dev = x.device
            tokens = self.proj(x)
            mask = torch.ones(B, dtype=torch.bool, device=dev)
            return tokens, mask

        if batch_size is None or device is None:
            raise ValueError("batch_size and device must be provided when x is None.")

        # Zero tokens for missing modality
        tokens = torch.zeros(batch_size, self.n_patches, self.d_model, device=device)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return tokens, mask


class RobustCrossModalTransformerWrapper(nn.Module):
    """
    Opt-in wrapper around PathwayCrossModalTransformer adding key-padding mask support
    for missing modalities without touching the frozen E1 architecture.
    """

    def __init__(
        self,
        transformer: PathwayCrossModalTransformer,
    ):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        bio_bias: Optional[torch.Tensor] = None,
        use_bias: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        seq: (B, S, d_model)
        key_padding_mask: (B, S) boolean tensor where True indicates a missing/padded token that should be masked out.
        """
        if key_padding_mask is None:
            # Standard frozen E1 path: no key-padding mask
            return self.transformer(seq, bio_bias=bio_bias, use_bias=use_bias)

        # Opt-in path: adjust bio_bias or attention mask with key_padding_mask
        B, S, d_model = seq.shape
        # Create an additive attention bias combining biological bias and key padding mask
        attn_bias = torch.zeros(B, 1, S, S, device=seq.device, dtype=seq.dtype)
        if use_bias and bio_bias is not None:
            attn_bias = attn_bias + bio_bias.unsqueeze(0).unsqueeze(0)

        # Mask out padded keys by adding -1e4 to columns where key_padding_mask is True
        # key_padding_mask: (B, S) -> (B, 1, 1, S)
        mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        attn_bias = attn_bias.masked_fill(mask_expanded, -1e4)

        x = seq
        all_attns = []
        for layer in self.transformer.layers:
            x_norm = layer.norm1(x)
            attn = layer.attn
            H, d_h = attn.n_heads, attn.d_head
            Q = attn.q_proj(x_norm).view(B, S, H, d_h).transpose(1, 2)
            K = attn.k_proj(x_norm).view(B, S, H, d_h).transpose(1, 2)
            V = attn.v_proj(x_norm).view(B, S, H, d_h).transpose(1, 2)

            logits = (Q @ K.transpose(-2, -1)) / attn.scale
            if use_bias and bio_bias is not None:
                logits = logits + attn.bias_scale * bio_bias.unsqueeze(0).unsqueeze(0)
            # Apply key_padding_mask
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            logits = logits.masked_fill(mask_expanded, -1e4)

            attn_weights = torch.softmax(logits, dim=-1)
            all_attns.append(attn_weights.detach())
            attn_dropped = attn.attn_drop(attn_weights)
            out = (attn_dropped @ V).transpose(1, 2).contiguous().view(B, S, -1)
            out = attn.resid_drop(attn.out_proj(out))

            x = x + out
            x = x + layer.ffn(layer.norm2(x))

        x = self.transformer.final_norm(x)
        return x, all_attns
