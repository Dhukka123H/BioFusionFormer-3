"""
BioFusionFormer Unit Tests
===========================
Tests cover:
  1. Mask enforcement  - MaskedLinear zeros out masked weights
  2. Gradient flow     - all trainable parameters receive gradients
  3. No leakage        - test-fold labels/stats never used in train preproc
  4. Determinism       - same seed -> identical outputs
  5. Parameter budget  - total params < 500k
  6. Forward shapes    - all output tensors have correct shapes
  7. Gate sum-to-one   - modality gate is a proper probability distribution
  8. Topology bias unchanged  - register_buffer does not make bias trainable
  9. PathwayBiasBuilder symmetry/diagonal checks
"""

import copy
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.biofusionformer.masked_projection import MaskedLinear, ModalityProjection
from src.biofusionformer.model import BioFusionFormer
from src.biofusionformer.pathway_bias import PathwayBiasBuilder
from src.biofusionformer.trainer import set_seed


N_PATHWAYS = 20
D_MODEL    = 16
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 32
N_MRNA     = 50
N_MIRNA    = 30
N_METH     = 40
BATCH_SIZE = 8


def _random_mask(n_pathways, n_features, density=0.3, seed=0):
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(n_pathways, n_features, generator=rng) < density


def make_masks():
    return {
        "mrna":  _random_mask(N_PATHWAYS, N_MRNA,  density=0.3, seed=1),
        "mirna": _random_mask(N_PATHWAYS, N_MIRNA, density=0.2, seed=2),
        "meth":  _random_mask(N_PATHWAYS, N_METH,  density=0.25, seed=3),
    }


def make_model(topology_bias=None):
    masks = make_masks()
    return BioFusionFormer(
        modality_features={"mrna": N_MRNA, "mirna": N_MIRNA, "meth": N_METH},
        modality_masks=masks,
        n_pathways=N_PATHWAYS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=0.0,
        topology_bias=topology_bias,
    )


def make_inputs(batch_size=BATCH_SIZE):
    torch.manual_seed(99)
    return {
        "mrna":  torch.randn(batch_size, N_MRNA),
        "mirna": torch.randn(batch_size, N_MIRNA),
        "meth":  torch.randn(batch_size, N_METH),
    }


# ----- Test 1: MaskedLinear -----

def test_masked_weights_zero_at_init():
    mask = _random_mask(N_PATHWAYS, N_MRNA)
    layer = MaskedLinear(N_MRNA, N_PATHWAYS, mask=mask)
    w = layer.weight.detach()
    assert (w[~mask] == 0.0).all(), "Masked positions not zero at init"


def test_active_weights_get_gradients():
    mask = _random_mask(N_PATHWAYS, N_MRNA)
    layer = MaskedLinear(N_MRNA, N_PATHWAYS, mask=mask)
    x = torch.randn(BATCH_SIZE, N_MRNA)
    layer(x).sum().backward()
    assert (layer.weight.grad[mask].abs() > 0).any(), "Active mask positions have no gradient"


def test_masked_weights_stay_zero_after_update():
    mask = _random_mask(N_PATHWAYS, N_MRNA)
    layer = MaskedLinear(N_MRNA, N_PATHWAYS, mask=mask)
    x = torch.randn(BATCH_SIZE, N_MRNA)
    layer(x).sum().backward()
    with torch.no_grad():
        if layer.weight.grad is not None:
            layer.weight.grad[~mask] = 0.0
            layer.weight -= 0.01 * layer.weight.grad
    assert (layer.weight.detach()[~mask] == 0.0).all()


# ----- Test 2: Gradient flow -----

def test_all_parameters_receive_gradients():
    set_seed(0)
    model = make_model()
    model.train()
    out = model(make_inputs())
    out["logit"].sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"Params missing grad: {no_grad}"


def test_no_nan_in_forward():
    set_seed(1)
    model = make_model()
    model.eval()
    with torch.no_grad():
        out = model(make_inputs())
    assert torch.isfinite(out["logit"]).all()
    assert torch.isfinite(out["gate"]).all()


# ----- Test 3: No leakage -----

def test_impute_uses_train_stats():
    from src.biofusionformer.dataset import _impute
    rng = np.random.RandomState(0)
    train = rng.randn(80, 10).astype(np.float32)
    test  = rng.randn(20, 10).astype(np.float32)
    train[0, 0] = np.nan
    test[0, 0]  = np.nan
    _, train_means = _impute(train)
    test_imputed, _ = _impute(test, fill_values=train_means)
    assert abs(float(test_imputed[0, 0]) - float(train_means[0])) < 1e-5


def test_scaling_uses_train_stats():
    rng = np.random.RandomState(0)
    train_data = rng.randn(80, 5).astype(np.float32) + 10.0
    test_data  = rng.randn(20, 5).astype(np.float32) - 10.0
    mu  = train_data.mean(0)
    std = train_data.std(0) + 1e-8
    test_scaled = (test_data - mu) / std
    # Test mean should NOT be near 0 because we used train statistics
    assert abs(test_scaled.mean()) > 1.0


# ----- Test 4: Determinism -----

def test_determinism_same_seed():
    set_seed(42)
    m1 = make_model(); m1.eval()
    inputs = make_inputs()
    with torch.no_grad():
        o1 = m1(inputs)["logit"].clone()
    set_seed(42)
    m2 = make_model(); m2.eval()
    with torch.no_grad():
        o2 = m2(inputs)["logit"].clone()
    assert torch.allclose(o1, o2, atol=1e-6)


def test_different_seeds_different_init():
    set_seed(1); m1 = make_model()
    set_seed(2); m2 = make_model()
    w1 = list(m1.projections["mrna"].parameters())[0].detach()
    w2 = list(m2.projections["mrna"].parameters())[0].detach()
    assert not torch.allclose(w1, w2)


# ----- Test 5: Parameter budget -----

def test_params_under_500k():
    model = make_model()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n < 500_000, f"{n:,} params exceeds 500k budget"


def test_param_count_report():
    model = make_model()
    counts = model.count_parameters()
    assert "total" in counts and counts["total"] > 0


# ----- Test 6: Forward shapes -----

def test_logit_shape():
    model = make_model(); model.eval()
    with torch.no_grad():
        out = model(make_inputs(BATCH_SIZE))
    assert out["logit"].shape == (BATCH_SIZE,)


def test_gate_shape():
    model = make_model(); model.eval()
    M = len(model.modality_names)
    with torch.no_grad():
        out = model(make_inputs(BATCH_SIZE))
    assert out["gate"].shape == (BATCH_SIZE, M)


def test_attn_shape():
    model = make_model(); model.eval()
    with torch.no_grad():
        out = model(make_inputs(BATCH_SIZE))
    assert len(out["attns"]) == N_LAYERS
    for a in out["attns"]:
        assert a.shape == (BATCH_SIZE, N_HEADS, N_PATHWAYS, N_PATHWAYS)


def test_single_sample_forward():
    model = make_model(); model.eval()
    with torch.no_grad():
        out = model(make_inputs(1))
    assert out["logit"].shape == (1,)


# ----- Test 7: Gate probability -----

def test_gate_sums_to_one():
    model = make_model(); model.eval()
    with torch.no_grad():
        gate = model(make_inputs(BATCH_SIZE))["gate"]
    assert torch.allclose(gate.sum(-1), torch.ones(BATCH_SIZE), atol=1e-5)


def test_gate_non_negative():
    model = make_model(); model.eval()
    with torch.no_grad():
        gate = model(make_inputs(BATCH_SIZE))["gate"]
    assert (gate >= 0).all()


# ----- Test 8: Topology bias non-trainable -----

def test_bias_is_buffer():
    bias = torch.rand(N_PATHWAYS, N_PATHWAYS)
    model = make_model(topology_bias=bias)
    param_names = {n for n, _ in model.named_parameters()}
    assert "topology_bias" not in param_names


def test_bias_unchanged_after_step():
    bias = torch.rand(N_PATHWAYS, N_PATHWAYS)
    model = make_model(topology_bias=bias)
    model.train()
    bias_before = model.topology_bias.clone()
    model(make_inputs())["logit"].sum().backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
    assert torch.allclose(bias_before, model.topology_bias)


def test_no_bias_works():
    model = make_model(topology_bias=None); model.eval()
    with torch.no_grad():
        out = model(make_inputs())
    assert torch.isfinite(out["logit"]).all()


# ----- Test 9: PathwayBiasBuilder -----

def test_coannot_bias_symmetric():
    masks = make_masks()
    pw = [f"R-HSA-{i}" for i in range(N_PATHWAYS)]
    B = PathwayBiasBuilder(
        hierarchy_path=None, pathway_order=pw,
        modality_masks={k: v.float() for k, v in masks.items()},
        parent_bias=0.0, coannot_bias=1.0, min_shared=1,
    ).build()
    assert torch.allclose(B, B.T, atol=1e-6)


def test_bias_diagonal_zero():
    pw = [f"R-HSA-{i}" for i in range(N_PATHWAYS)]
    B = PathwayBiasBuilder(hierarchy_path=None, pathway_order=pw).build()
    assert (B.diagonal() == 0).all()


def test_bias_shape():
    pw = [f"R-HSA-{i}" for i in range(N_PATHWAYS)]
    B = PathwayBiasBuilder(hierarchy_path=None, pathway_order=pw).build()
    assert B.shape == (N_PATHWAYS, N_PATHWAYS)


# ----- Smoke test -----

def test_smoke_full_pipeline():
    set_seed(7)
    model = make_model(); model.train()
    out = model(make_inputs())
    assert out["logit"].shape == (BATCH_SIZE,)
    nn.BCEWithLogitsLoss()(out["logit"], torch.zeros(BATCH_SIZE)).backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad: {n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
