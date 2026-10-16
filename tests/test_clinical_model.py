"""
tests/test_clinical_model.py
============================
Regression and unit tests for Stage 5.2 BioFusionFormerClinicalModel:
1. With use_clinical=False: parameter names, shapes, and eval output are identical
   to BioFusionFormerModel for the exact same seed.
2. With use_clinical=True: 4th modality produces valid logits, gate weights (B, 4)
   summing to 1.0, and runs without errors.
"""

import pytest
import torch

from src.biofusionformer.full_model import BioFusionFormerModel
from src.biofusionformer.full_model_clinical import BioFusionFormerClinicalModel
from src.encoders.clinical_encoder import ClinicalFeatureEncoder


def _make_dummy_masks(P=6, feat_dims=None):
    if feat_dims is None:
        feat_dims = {"mrna": 20, "mirna": 8, "meth": 12}
    masks = {}
    for i, (m, f) in enumerate(feat_dims.items()):
        g = torch.Generator().manual_seed(i)
        mask = torch.rand(P, f, generator=g) < 0.3
        mask[:, 0] = True
        masks[m] = mask
    return masks


def test_clinical_encoder_output_shape():
    encoder = ClinicalFeatureEncoder(in_features=3, d_model=16, dropout=0.3)
    x = torch.randn(5, 3)
    out = encoder(x)
    assert out.shape == (5, 1, 16)


def test_clinical_model_regression_with_use_clinical_false():
    """Verify that with use_clinical=False, parameter names, shapes, and output match BioFusionFormerModel."""
    seed = 42
    feat_dims = {"mrna": 15, "mirna": 10, "meth": 12}
    P = 4
    d_model = 16
    masks = _make_dummy_masks(P=P, feat_dims=feat_dims)

    # 1. Base model
    torch.manual_seed(seed)
    base_model = BioFusionFormerModel(
        modality_features=feat_dims,
        modality_masks=masks,
        n_pathways=P,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        d_ff=32,
    )

    # 2. Clinical model with use_clinical=False
    torch.manual_seed(seed)
    clin_model = BioFusionFormerClinicalModel(
        modality_features=feat_dims,
        modality_masks=masks,
        n_pathways=P,
        clinical_features=3,
        use_clinical=False,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        d_ff=32,
    )

    # Verify parameter names and shapes
    base_params = dict(base_model.named_parameters())
    clin_params = dict(clin_model.named_parameters())

    assert list(base_params.keys()) == list(clin_params.keys()), (
        f"Param names differ:\nBase: {list(base_params.keys())}\nClin: {list(clin_params.keys())}"
    )

    for k in base_params:
        assert base_params[k].shape == clin_params[k].shape, f"Shape mismatch at {k}"
        torch.testing.assert_close(base_params[k], clin_params[k], msg=f"Value mismatch at {k}")

    # Verify forward eval outputs
    base_model.eval()
    clin_model.eval()

    inputs = {m: torch.randn(4, f) for m, f in feat_dims.items()}
    with torch.no_grad():
        out_base = base_model(inputs)
        out_clin = clin_model(inputs)

    torch.testing.assert_close(out_base["logit"], out_clin["logit"])
    torch.testing.assert_close(out_base["gate"], out_clin["gate"])
    torch.testing.assert_close(out_base["cls_rep"], out_clin["cls_rep"])
    torch.testing.assert_close(out_base["gated_rep"], out_clin["gated_rep"])


def test_clinical_model_with_use_clinical_true():
    """Verify that with use_clinical=True, 4 modalities are processed and gate weights sum to 1."""
    feat_dims = {"mrna": 15, "mirna": 10, "meth": 12}
    P = 4
    d_model = 16
    masks = _make_dummy_masks(P=P, feat_dims=feat_dims)

    clin_model = BioFusionFormerClinicalModel(
        modality_features=feat_dims,
        modality_masks=masks,
        n_pathways=P,
        clinical_features=3,
        use_clinical=True,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        d_ff=32,
    )

    inputs = {m: torch.randn(4, f) for m, f in feat_dims.items()}
    inputs["clinical"] = torch.randn(4, 3)

    clin_model.eval()
    with torch.no_grad():
        out = clin_model(inputs)

    assert out["logit"].shape == (4,)
    assert out["gate"].shape == (4, 4)  # 4 modalities: mrna, mirna, meth, clinical
    # Gate weights must sum to 1.0 across modalities
    torch.testing.assert_close(out["gate"].sum(dim=1), torch.ones(4))
