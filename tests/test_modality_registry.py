"""
tests/test_modality_registry.py
===============================
Unit tests for Stage 6.5:
1. Modality registry registration and retrieval.
2. ImagingFeatureEncoder produces correct shapes and availability masks.
3. RobustCrossModalTransformerWrapper handles 5th modality (imaging) when present and when absent.
"""

import pytest
import torch

from src.modalities.registry import (
    ImagingFeatureEncoder,
    ModalityRegistry,
    ModalitySpec,
    RobustCrossModalTransformerWrapper,
)
from src.transformer.transformer import PathwayCrossModalTransformer


def test_registry_registration():
    reg = ModalityRegistry()
    spec_mrna = ModalitySpec(name="mrna", token_count=192, modality_type="omics")
    spec_img = ModalitySpec(name="imaging", token_count=16, modality_type="imaging")

    reg.register(spec_mrna)
    reg.register(spec_img)

    assert reg.list_modalities() == ["mrna", "imaging"]
    assert reg.get("mrna").token_count == 192

    with pytest.raises(ValueError, match="already registered"):
        reg.register(spec_mrna)

    with pytest.raises(KeyError):
        reg.get("unknown_modality")


def test_imaging_encoder_present_and_absent():
    enc = ImagingFeatureEncoder(in_dim=128, n_patches=8, d_model=32, dropout=0.3)

    # 1. Present modality
    x_present = torch.randn(4, 8, 128)
    tokens, mask = enc(x_present)
    assert tokens.shape == (4, 8, 32)
    assert mask.shape == (4,)
    assert mask.all().item() is True

    # 2. Absent modality
    tokens_abs, mask_abs = enc(x=None, batch_size=4, device=torch.device("cpu"))
    assert tokens_abs.shape == (4, 8, 32)
    assert (tokens_abs == 0).all().item() is True
    assert (mask_abs == False).all().item() is True


def test_robust_transformer_with_imaging_modality_present_and_absent():
    """Verify that a 5th modality (imaging) plugs in and wrapper runs when present and absent."""
    d_model = 32
    n_heads = 4
    base_trans = PathwayCrossModalTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=2, d_ff=64
    )
    wrapper = RobustCrossModalTransformerWrapper(base_trans)

    B = 3
    S_omics = 1 + 3 * 6  # CLS + 3 omics * 6 pathways = 19
    S_img = 4             # 4 imaging patch tokens
    S_total = S_omics + S_img  # 23 tokens total

    # Case A: Imaging present
    seq_present = torch.randn(B, S_total, d_model)
    key_mask_none = torch.zeros(B, S_total, dtype=torch.bool)  # all tokens present
    out_present, attns_present = wrapper(seq_present, key_padding_mask=key_mask_none)
    assert out_present.shape == (B, S_total, d_model)
    assert len(attns_present) == 2

    # Case B: Imaging absent for patient 1 and 2
    seq_with_missing = seq_present.clone()
    key_padding_mask = torch.zeros(B, S_total, dtype=torch.bool)
    # Mask out imaging tokens (last 4 tokens) for patients 1 and 2
    key_padding_mask[1:, -S_img:] = True
    seq_with_missing[1:, -S_img:] = 0.0

    out_missing, attns_missing = wrapper(seq_with_missing, key_padding_mask=key_padding_mask)
    assert out_missing.shape == (B, S_total, d_model)
    assert len(attns_missing) == 2

    # Verify that attention to padded keys for patients 1 and 2 is practically zero (~0)
    for attn in attns_missing:
        # attn: (B, n_heads, S, S)
        # For patients 1 and 2, attention to keys -S_img: should be 0 (< 1e-4)
        masked_attn_weights = attn[1:, :, :, -S_img:]
        assert (masked_attn_weights < 1e-4).all().item() is True
