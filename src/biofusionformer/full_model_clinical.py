"""
src/biofusionformer/full_model_clinical.py
==========================================
BioFusionFormerClinicalModel: Extends BioFusionFormer with an optional 4th clinical modality (Stage 5.2).

Design rules per AGENT_PROMPT Stage 5.2:
  1. Additive 4th modality made of ONE clinical token (2-layer MLP, LayerNorm, dropout >= 0.3 -> d_model).
  2. The existing AdaptiveModalityGate expects (B, M, P, d_model); handle the clinical token
     by repeating/pooling so that the existing gate code is not modified.
  3. Regression guarantee: With use_clinical=False, parameter names, shapes, and eval outputs
     are IDENTICAL to BioFusionFormerModel for the same seed.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.encoders.masked_encoder import PathwayModalityEncoder
from src.encoders.clinical_encoder import ClinicalFeatureEncoder
from src.transformer.transformer import PathwayCrossModalTransformer
from src.fusion.adaptive_fusion import AdaptiveModalityGate
from src.heads.classification import ClassificationHead


class BioFusionFormerClinicalModel(nn.Module):
    """
    BioFusionFormer with optional clinical feature token integration.
    """

    def __init__(
        self,
        modality_features: Dict[str, int],
        modality_masks: Dict[str, torch.Tensor],
        n_pathways: int,
        clinical_features: int = 0,
        use_clinical: bool = False,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.3,
        attn_dropout: float = 0.2,
        modality_dropout: float = 0.2,
        clinical_dropout: float = 0.3,
        use_pathway_bias: bool = True,
        use_gating: bool = True,
        bio_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.modality_names = list(modality_features.keys())
        self.use_clinical = use_clinical and (clinical_features > 0)
        self.n_pathways = n_pathways
        self.d_model = d_model
        self.use_pathway_bias = use_pathway_bias
        self.use_gating = use_gating

        # Total modalities (3 omics + optional 1 clinical)
        total_modalities = len(self.modality_names) + (1 if self.use_clinical else 0)
        self.n_modalities = total_modalities

        # 1. Omics Modality Encoders with Masked Linear layers
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
        # NOTE: when use_clinical=False, total_modalities == len(self.modality_names) == 3,
        # so cls_token and modality_embed match BioFusionFormerModel exactly.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.modality_embed = nn.Embedding(total_modalities, d_model)
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
            if self.use_clinical and bio_bias.shape[0] != (1 + total_modalities * n_pathways):
                # Extend bio_bias with neutral (0.0) prior for clinical tokens
                target_len = 1 + total_modalities * n_pathways
                extended_bias = torch.zeros(target_len, target_len, device=bio_bias.device, dtype=bio_bias.dtype)
                cur_len = bio_bias.shape[0]
                extended_bias[:cur_len, :cur_len] = bio_bias
                self.register_buffer("bio_bias", extended_bias.float())
            else:
                self.register_buffer("bio_bias", bio_bias.float())
        else:
            self.register_buffer("bio_bias", None)

        # 4. Adaptive Fusion Gating
        self.fusion_gate = AdaptiveModalityGate(
            n_modalities=total_modalities,
            d_model=d_model,
            modality_dropout=modality_dropout,
        )

        # 5. Classification Head
        self.head = ClassificationHead(in_features=d_model, hidden_dim=d_model // 2, dropout=dropout)

        # 6. Optional Clinical Feature Encoder (only created when use_clinical=True to preserve exact param names)
        if self.use_clinical:
            self.clinical_encoder = ClinicalFeatureEncoder(
                in_features=clinical_features,
                d_model=d_model,
                dropout=clinical_dropout,
            )
        else:
            self.clinical_encoder = None

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        inputs: dict mapping modality names ("mrna", "mirna", "meth", and optionally "clinical") to tensors.
        """
        first_tensor = next(iter(inputs.values()))
        B = first_tensor.shape[0]
        device = first_tensor.device

        # Encode omics modalities into pathway tokens (B, P, d_model)
        token_lists = []
        for i, mod in enumerate(self.modality_names):
            mod_x = inputs[mod]
            tokens = self.encoders[mod](mod_x)  # (B, P, d_model)
            mod_emb = self.modality_embed(torch.tensor(i, device=device))  # (d_model,)
            tokens = tokens + mod_emb.unsqueeze(0).unsqueeze(0)
            token_lists.append(tokens)

        # Encode clinical modality if enabled
        if self.use_clinical:
            clin_idx = len(self.modality_names)
            clin_x = inputs["clinical"]  # (B, F_clin)
            # ClinicalFeatureEncoder produces ONE token: (B, 1, d_model)
            one_clin_token = self.clinical_encoder(clin_x)  # (B, 1, d_model)
            clin_emb = self.modality_embed(torch.tensor(clin_idx, device=device))  # (d_model,)
            one_clin_token = one_clin_token + clin_emb.unsqueeze(0).unsqueeze(0)

            # Handle clinical token for AdaptiveModalityGate by repeating along the P dimension:
            # (B, 1, d_model) -> (B, P, d_model)
            clin_tokens = one_clin_token.repeat(1, self.n_pathways, 1)  # (B, P, d_model)
            token_lists.append(clin_tokens)

        # Stack modality tokens: (B, M, P, d_model)
        stacked_mod_tokens = torch.stack(token_lists, dim=1)

        # Flatten sequence: (B, M * P, d_model)
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

        # Combined patient representation: CLS + gated representation
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
        clinical = sum(p.numel() for p in self.clinical_encoder.parameters() if p.requires_grad) if self.use_clinical else 0
        return {
            "total_trainable": total,
            "encoders": encoders,
            "transformer": transformer,
            "fusion": fusion,
            "head": head,
            "clinical": clinical,
        }

    def count_effective_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dead = sum(int((enc.masked_linear.mask == 0).sum().item()) for enc in self.encoders.values())
        return total - dead
