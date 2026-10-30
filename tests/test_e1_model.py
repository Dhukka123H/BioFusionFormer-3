"""Unit tests for the modules that ACTUALLY produced E1 (src.encoders, src.fusion, src.transformer,
src.biofusionformer.full_model). They replace the old tests that only covered the discarded pilot model.
Run: python -m pytest tests -q
"""
import pytest
import torch
import torch.nn.functional as F

from src.biofusionformer.full_model import BioFusionFormerModel
from src.encoders.masked_encoder import MaskedLinear, PathwayModalityEncoder
from src.fusion.adaptive_fusion import AdaptiveModalityGate


def _mask(out_f, in_f, seed=0, p=0.3):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(out_f, in_f, generator=g) < p
    m[:, 0] = True
    return m


FEATS = {"mrna": 20, "mirna": 8, "meth": 12}
P = 6


def _build(seed=0, **kw):
    torch.manual_seed(seed)
    masks = {m: _mask(P, f, seed=i) for i, (m, f) in enumerate(FEATS.items())}
    S = 1 + len(FEATS) * P
    bias = torch.zeros(S, S)
    bias[0, :] = 0.5
    bias[:, 0] = 0.5
    bias[0, 0] = 0.0
    return BioFusionFormerModel(
        modality_features=FEATS, modality_masks=masks, n_pathways=P,
        d_model=8, n_heads=2, n_layers=2, d_ff=16, bio_bias=bias, **kw,
    ), masks


def _inputs(B=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(B, f, generator=g) for m, f in FEATS.items()}


def test_masked_linear_zero_outside_mask_at_init():
    mask = _mask(4, 10)
    ml = MaskedLinear(10, 4, mask)
    assert (ml.weight[~mask] == 0).all()


def test_masked_linear_no_grad_and_stays_zero_after_adamw_step():
    mask = _mask(4, 10)
    ml = MaskedLinear(10, 4, mask)
    opt = torch.optim.AdamW(ml.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(3):
        opt.zero_grad()
        ml(torch.randn(6, 10)).pow(2).sum().backward()
        assert ml.weight.grad[~mask].abs().sum() == 0
        opt.step()
    assert (ml.weight[~mask] == 0).all()
    assert ml.weight[mask].abs().sum() > 0


def test_masked_linear_forward_matches_masked_weight():
    mask = _mask(4, 10)
    ml = MaskedLinear(10, 4, mask)
    x = torch.randn(3, 10)
    ref = F.linear(x, ml.weight * mask.float(), ml.bias)
    assert torch.allclose(ml(x), ref, atol=1e-6)


def test_encoder_output_shape():
    enc = PathwayModalityEncoder(10, 4, _mask(4, 10), d_model=8)
    assert enc(torch.randn(5, 10)).shape == (5, 4, 8)


def test_gate_weights_sum_to_one_and_modality_dropout_keeps_one():
    gate = AdaptiveModalityGate(n_modalities=3, d_model=8, modality_dropout=0.9)
    tokens = torch.randn(64, 3, 4, 8)
    for _ in range(50):
        _, w = gate(tokens, training=True, use_gating=True)
        assert (w.sum(-1) - 1).abs().max() < 1e-5
        assert ((w < 1e-6).sum(-1) <= 2).all()  # at most M-1 modalities are dropped per sample


def test_gate_uniform_when_gating_off():
    gate = AdaptiveModalityGate(n_modalities=3, d_model=8)
    _, w = gate(torch.randn(4, 3, 4, 8), training=False, use_gating=False)
    assert torch.allclose(w, torch.full((4, 3), 1 / 3))


def test_full_model_shapes_and_attention_rows_sum_to_one():
    model, _ = _build()
    model.eval()
    out = model(_inputs())
    assert out["logit"].shape == (5,)
    assert out["gate"].shape == (5, 3) and torch.allclose(out["gate"].sum(-1), torch.ones(5), atol=1e-5)
    assert len(out["attns"]) == 2
    S = 1 + 3 * P
    for a in out["attns"]:
        assert a.shape == (5, 2, S, S)
        assert torch.allclose(a.sum(-1), torch.ones(5, 2, S), atol=1e-5)


def test_masks_survive_a_full_model_training_step():
    model, masks = _build()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(3):
        opt.zero_grad()
        model(_inputs())["logit"].sum().backward()
        opt.step()
    for name, m in masks.items():
        assert (model.encoders[name].masked_linear.weight[~m] == 0).all()


def test_same_seed_same_model_and_output():
    m1, _ = _build(seed=7)
    m2, _ = _build(seed=7)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    m1.eval(); m2.eval()
    x = _inputs()
    assert torch.equal(m1(x)["logit"], m2(x)["logit"])


def test_effective_parameters_exclude_masked_out_weights():
    model, masks = _build()
    dense = model.count_parameters()["total_trainable"]
    dead = sum(int((~m).sum()) for m in masks.values())
    assert model.count_effective_parameters() == dense - dead < dense


def test_missing_modality_is_not_supported_documented_limitation():
    """Modality dropout in this model only affects the GATE during training; the transformer and the [CLS]
    token always see every modality. Inference therefore needs all three inputs. If this test starts to
    fail, token-level masking was added and the README/GUI documentation must be updated."""
    model, _ = _build()
    model.eval()
    x = _inputs()
    x.pop("meth")
    with pytest.raises(KeyError):
        model(x)


def test_build_bio_bias_matrix_structure_and_symmetry():
    import scipy.sparse as sp
    from src.pathways.bio_bias import build_bio_bias_matrix

    P_toy = 4
    M_toy = 3
    S_toy = 1 + M_toy * P_toy  # 13

    # Toy symmetric adjacency
    adj = sp.csr_matrix([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    bias = build_bio_bias_matrix(list(range(P_toy)), n_modalities=M_toy, adj_sparse=adj)

    assert bias.shape == (S_toy, S_toy)
    # [CLS] token bidirectional attention is 0.5; [CLS, CLS] is 0.0
    assert (bias[0, 1:] == 0.5).all()
    assert (bias[1:, 0] == 0.5).all()
    assert bias[0, 0] == 0.0

    # Self-attention diagonal is strictly zero
    assert torch.equal(torch.diag(bias), torch.zeros(S_toy))

    # Attention bias is symmetric
    assert torch.equal(bias, bias.T)

    # Cross-modality identical pathway receives at least 1.0
    assert bias[1, 1 + P_toy] >= 1.0

def test_build_bio_bias_matrix_equivalence_to_prerefactor():
    """
    Extracts the pre-refactor build_bio_bias_matrix directly from
    git show cbaaffc^:scripts/run_e1.py, executes it with the same arguments,
    and asserts torch.equal against src.pathways.bio_bias.build_bio_bias_matrix.
    Includes golden checks on the canonical bio_bias:
    nonzero=3330, breakdown 1152/1152/1026, and SHA-256 of the tensor bytes.
    Skips if git show is unavailable.
    """
    import hashlib
    import json
    import subprocess
    from pathlib import Path
    import scipy.sparse as sp
    from src.pathways.bio_bias import build_bio_bias_matrix as new_build

    root = Path(__file__).resolve().parent.parent

    # 1. Golden check on canonical bio_bias
    bio_bias_path = root / "models" / "final" / "bio_bias.pt"
    if bio_bias_path.exists():
        bias = torch.load(bio_bias_path, map_location="cpu", weights_only=True)
    else:
        from src.pathways.selection import select_unsupervised_pathways
        data_dir = root / "data"
        pri = data_dir / "interim" / "priors"
        selected_pways, _ = select_unsupervised_pathways(data_dir, top_k=192)
        prior_pathways = json.load(open(pri / "gene_pathway_mask_pathways.json"))
        pway_map = {p: i for i, p in enumerate(prior_pathways)}
        pway_idx = [pway_map[p] for p in selected_pways]
        adj_full = sp.load_npz(pri / "pathway_adjacency.npz")
        bias = new_build(pway_idx, 3, adj_full)

    # Golden check: nonzero count = 3330
    nonzero = int((bias != 0).sum().item())
    assert nonzero == 3330, f"Expected 3330 nonzeros, got {nonzero}"

    # Golden check: breakdown 1152 / 1152 / 1026
    P = (bias.shape[0] - 1) // 3
    cls_mask = torch.zeros_like(bias, dtype=torch.bool)
    cls_mask[0, 1:] = True
    cls_mask[1:, 0] = True
    cls_nz = int(((bias != 0) & cls_mask).sum().item())
    assert cls_nz == 1152, f"Expected 1152 CLS entries, got {cls_nz}"

    same_pway_mask = torch.zeros_like(bias, dtype=torch.bool)
    for m1 in range(3):
        for m2 in range(3):
            if m1 != m2:
                off1 = 1 + m1 * P
                off2 = 1 + m2 * P
                for p in range(P):
                    same_pway_mask[off1 + p, off2 + p] = True
    same_nz = int(((bias != 0) & same_pway_mask).sum().item())
    assert same_nz == 1152, f"Expected 1152 same-pathway entries, got {same_nz}"

    hier_mask = torch.zeros_like(bias, dtype=torch.bool)
    for m1 in range(3):
        for m2 in range(3):
            off1 = 1 + m1 * P
            off2 = 1 + m2 * P
            for p1 in range(P):
                for p2 in range(P):
                    if p1 != p2:
                        hier_mask[off1 + p1, off2 + p2] = True
    hier_nz = int(((bias != 0) & hier_mask).sum().item())
    assert hier_nz == 1026, f"Expected 1026 hierarchy entries, got {hier_nz}"

    # Golden check: SHA-256 of the bias tensor bytes
    tensor_bytes_sha = hashlib.sha256(bias.numpy().tobytes()).hexdigest()
    assert tensor_bytes_sha == "07286ee1c34400a66336e3684fab0cedac82c1ac5f32ba76ec11f3e46254e8d4"

    # 2. Pre-refactor equivalence test via git show (skips if unavailable)
    try:
        res = subprocess.run(
            ["git", "show", "cbaaffc^:scripts/run_e1.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception as e:
        pytest.skip(f"git show unavailable or commit cbaaffc not found: {e}")

    lines = res.stdout.splitlines()
    func_lines = []
    capturing = False
    for line in lines:
        if line.startswith("def build_bio_bias_matrix("):
            capturing = True
        if capturing:
            func_lines.append(line)
            if len(func_lines) > 5 and line.startswith("def ") and not line.startswith("def build_bio_bias_matrix("):
                func_lines.pop()
                break

    if not func_lines:
        pytest.skip("Could not extract pre-refactor build_bio_bias_matrix from git show")

    scope = {"sp": sp, "np": __import__("numpy"), "torch": torch, "List": list}
    exec("\n".join(func_lines), scope)
    old_build = scope["build_bio_bias_matrix"]

    # Test equivalence on toy graph
    adj = sp.csr_matrix([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    p_indices = [0, 1, 2, 3]

    b_old = old_build(p_indices, 3, adj)
    b_new = new_build(p_indices, 3, adj)

    assert torch.equal(b_old, b_new)
