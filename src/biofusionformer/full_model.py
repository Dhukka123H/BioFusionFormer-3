"""
BioFusionFormer - Full Integrated Model
========================================
Combines:
  1. PathwayModalityEncoders (biology-masked projection from genes/miRNAs/CpGs to pathway tokens)
  2. Modality Embeddings + [CLS] Token
  3. PathwayCrossModalTransformer (pre-LN, biased multi-head attention with soft Reactome DAG prior)
  4. AdaptiveModalityGate (patient-specific modality weights + modality dropout)
  5. ClassificationHead
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from src.encoders.masked_encoder import PathwayModalityEncoder
from src.transformer.transformer import PathwayCrossModalTransformer
from src.fusion.adaptive_fusion import AdaptiveModalityGate
from src.heads.classification import ClassificationHead


class BioFusionFormerModel(nn.Module):
    """
    BioFusionFormer: Pathway-Aware Hierarchical Cross-Modal Transformer.
    """

    def __init__(
        self,
        modality_features: Dict[str, int],
        modality_masks: Dict[str, torch.Tensor],
        n_pathways: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.3,
        attn_dropout: float = 0.2,
        modality_dropout: float = 0.2,
        use_pathway_bias: bool = True,
        use_gating: bool = True,
        bio_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.modality_names = list(modality_features.keys())
        self.n_modalities = len(self.modality_names)
        self.n_pathways = n_pathways
        self.d_model = d_model
        self.use_pathway_bias = use_pathway_bias
        self.use_gating = use_gating

        # 1. Modality Encoders with Masked Linear layers
        self.encoders = nn.ModuleDict()
        for mod in self.modality_names:
            self.encoders[mod] = PathwayModalityEncoder(
                in_features=modality_features[mod],
                n_pathways=n_pathways,
                mask=modality_masks[mod],
                d_model=d_model,
                dropout=dropout,
            )

        # 2. Embeddings & [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.modality_embed = nn.Embedding(self.n_modalities, d_model)
        nn.init.normal_(self.modality_embed.weight, std=0.02)

        # 3. Cross-Modal Transformer
        self.transformer = PathwayCrossModalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )

        # Biological attention bias buffer
        if bio_bias is not None:
            self.register_buffer("bio_bias", bio_bias.float())
        else:
            self.register_buffer("bio_bias", None)

        # 4. Adaptive Fusion Gating
        self.fusion_gate = AdaptiveModalityGate(
            n_modalities=self.n_modalities,
            d_model=d_model,
            modality_dropout=modality_dropout,
        )

        # 5. Classification Head
        self.head = ClassificationHead(in_features=d_model, hidden_dim=d_model // 2, dropout=dropout)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        inputs: dict mapping modality name to tensor (B, F_mod)
        """
        B = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device

        # Encode each modality into pathway tokens
        token_lists = []
        for i, mod in enumerate(self.modality_names):
            mod_x = inputs[mod]
            tokens = self.encoders[mod](mod_x)  # (B, P, d_model)
            # Add modality embedding
            mod_emb = self.modality_embed(torch.tensor(i, device=device))  # (d_model,)
            tokens = tokens + mod_emb.unsqueeze(0).unsqueeze(0)
            token_lists.append(tokens)

        # Stack modality tokens: (B, M, P, d_model)
        stacked_mod_tokens = torch.stack(token_lists, dim=1)

        # Flatten modality-pathway sequence: (B, M * P, d_model)
        flat_tokens = stacked_mod_tokens.view(B, self.n_modalities * self.n_pathways, self.d_model)

        # Prepend [CLS] token: (B, 1 + M * P, d_model)
        cls_expanded = self.cls_token.expand(B, -1, -1)
        full_seq = torch.cat([cls_expanded, flat_tokens], dim=1)

        # Transformer forward pass
        trans_out, attns = self.transformer(
            full_seq,
            bio_bias=self.bio_bias,
            use_bias=self.use_pathway_bias,
        )

        # Split output into [CLS] and contextualized modality tokens
        cls_out = trans_out[:, 0, :]  # (B, d_model)
        contextualized_flat = trans_out[:, 1:, :]  # (B, M * P, d_model)
        contextualized_mod = contextualized_flat.view(B, self.n_modalities, self.n_pathways, self.d_model)

        # Adaptive Modality Gating
        gated_rep, gate_weights = self.fusion_gate(
            contextualized_mod,
            training=self.training,
            use_gating=self.use_gating,
        )

        # Combined patient representation: CLS token + gated representation
        patient_rep = cls_out + gated_rep
        logit = self.head(patient_rep)

        return {
            "logit": logit,
            "gate": gate_weights,
            "attns": attns,
            "cls_rep": cls_out,
            "gated_rep": gated_rep,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoders = sum(p.numel() for p in self.encoders.parameters() if p.requires_grad)
        transformer = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        fusion = sum(p.numel() for p in self.fusion_gate.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "total_trainable": total,
            "encoders": encoders,
            "transformer": transformer,
            "fusion": fusion,
            "head": head,
        }
