"""
BioFusionFormer - Adaptive Gating & Modality Fusion
===================================================
Produces patient-specific softmax gate weights across modalities from pooled representations,
incorporating Modality Dropout (p=0.2) during training to ensure robustness against noisy modalities
(e.g., methylation).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveModalityGate(nn.Module):
    """
    Computes modality gating weights and applies optional modality dropout.
    """

    def __init__(self, n_modalities: int, d_model: int = 32, modality_dropout: float = 0.2):
        super().__init__()
        self.n_modalities = n_modalities
        self.d_model = d_model
        self.modality_dropout = modality_dropout
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        modality_tokens: torch.Tensor,
        training: bool = True,
        use_gating: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        modality_tokens: (B, M, P, d_model)
        Returns:
          gated_pooled: (B, d_model)
          gate_weights: (B, M)
        """
        B, M, P, D = modality_tokens.shape
        # Pool across pathways per modality: (B, M, D)
        mod_pooled = modality_tokens.mean(dim=2)

        if not use_gating:
            # Uniform mean fusion across modalities (Ablation E7)
            gate_weights = torch.full((B, M), 1.0 / M, device=modality_tokens.device)
            fused = mod_pooled.mean(dim=1)
            return fused, gate_weights

        # Compute gate logits per modality: (B, M, 1) -> (B, M)
        gate_logits = self.gate_mlp(mod_pooled).squeeze(-1)

        # Modality dropout during training: randomly mask out modalities (ensure at least one remains)
        if training and self.modality_dropout > 0:
            drop_mask = torch.rand(B, M, device=modality_tokens.device) > self.modality_dropout
            # Ensure at least one modality is kept per sample
            all_dropped = (~drop_mask).all(dim=-1)
            if all_dropped.any():
                keep_idx = torch.randint(0, M, (all_dropped.sum(),), device=modality_tokens.device)
                drop_mask[all_dropped, keep_idx] = True
            gate_logits = gate_logits.masked_fill(~drop_mask, -1e9)

        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, M)
        # Weighted sum: (B, M, 1) * (B, M, D) -> sum over M -> (B, D)
        fused = (gate_weights.unsqueeze(-1) * mod_pooled).sum(dim=1)
        return fused, gate_weights
