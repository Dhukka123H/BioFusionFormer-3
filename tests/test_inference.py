"""
tests/test_inference.py
=======================
Tests for Stage 6.2 and 6.3:
1. Validator tests:
   - Rejects if any modality is missing.
   - Rejects Ensembl IDs with version suffixes (.XX).
   - Rejects negative values in mRNA/miRNA.
   - Rejects out-of-range beta values (> 1.0 or < 0).
   - Rejects < 90% feature coverage.
2. End-to-end test on 3 patients of the cohort.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.inference.predict import predict, validate_omics_inputs

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"


def test_validator_rejects_missing_modality():
    df = pd.DataFrame([[1.0, 2.0]], columns=["ENSG00000000419", "ENSG00000000460"])
    with pytest.raises(ValueError, match="Missing required modality"):
        validate_omics_inputs(mrna_df=None, mirna_df=df, meth_df=df, expected_features={})

    with pytest.raises(ValueError, match="Missing required modality"):
        validate_omics_inputs(mrna_df=df, mirna_df=None, meth_df=df, expected_features={})

    with pytest.raises(ValueError, match="Missing required modality"):
        validate_omics_inputs(mrna_df=df, mirna_df=df, meth_df=None, expected_features={})


def test_validator_rejects_ensembl_version_suffix():
    mrna_bad = pd.DataFrame([[1.0]], columns=["ENSG00000000419.15"])
    mirna_ok = pd.DataFrame([[1.0]], columns=["hsa-let-7a-5p"])
    meth_ok = pd.DataFrame([[0.5]], columns=["ENSG00000010810"])

    expected = {
        "mrna": ["ENSG00000000419"],
        "mirna": ["hsa-let-7a-5p"],
        "meth": ["ENSG00000010810"],
    }
    with pytest.raises(ValueError, match="version suffixes"):
        validate_omics_inputs(mrna_bad, mirna_ok, meth_ok, expected)


def test_validator_rejects_negative_mrna_or_mirna():
    mrna_bad = pd.DataFrame([[-2.0]], columns=["ENSG00000000419"])
    mirna_ok = pd.DataFrame([[1.0]], columns=["hsa-let-7a-5p"])
    meth_ok = pd.DataFrame([[0.5]], columns=["ENSG00000010810"])

    expected = {
        "mrna": ["ENSG00000000419"],
        "mirna": ["hsa-let-7a-5p"],
        "meth": ["ENSG00000010810"],
    }
    with pytest.raises(ValueError, match="negative values"):
        validate_omics_inputs(mrna_bad, mirna_ok, meth_ok, expected)


def test_validator_rejects_out_of_range_methylation():
    mrna_ok = pd.DataFrame([[2.0]], columns=["ENSG00000000419"])
    mirna_ok = pd.DataFrame([[1.0]], columns=["hsa-let-7a-5p"])
    meth_bad = pd.DataFrame([[1.8]], columns=["ENSG00000010810"])  # > 1.0

    expected = {
        "mrna": ["ENSG00000000419"],
        "mirna": ["hsa-let-7a-5p"],
        "meth": ["ENSG00000010810"],
    }
    with pytest.raises(ValueError, match="Methylation beta values out of range"):
        validate_omics_inputs(mrna_ok, mirna_ok, meth_bad, expected)


def test_validator_rejects_low_feature_coverage():
    # Only 1 feature present out of 10 expected (10% coverage < 90%)
    mrna_sparse = pd.DataFrame([[2.0]], columns=["ENSG00000000419"])
    mirna_ok = pd.DataFrame([[1.0]], columns=["hsa-let-7a-5p"])
    meth_ok = pd.DataFrame([[0.5]], columns=["ENSG00000010810"])

    expected = {
        "mrna": [f"ENSG{i:011d}" for i in range(10)],
        "mirna": ["hsa-let-7a-5p"],
        "meth": ["ENSG00000010810"],
    }
    with pytest.raises(ValueError, match="Insufficient feature coverage"):
        validate_omics_inputs(mrna_sparse, mirna_ok, meth_ok, expected)


def test_end_to_end_prediction_on_cohort_subset():
    """Test full predict() pipeline on 3 real patients from the cohort."""
    if not (DATA_DIR / "mrna_gene.parquet").exists():
        pytest.skip("Processed omics data not available")

    mrna = pd.read_parquet(DATA_DIR / "mrna_gene.parquet").iloc[:3]
    mirna = pd.read_parquet(DATA_DIR / "mirna_unique.parquet").iloc[:3]
    meth = pd.read_parquet(DATA_DIR / "meth_gene.parquet").iloc[:3]

    res = predict(mrna, mirna, meth)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert "patient_id" in res.columns
    assert "probability_unfavorable" in res.columns
    assert "gate_mrna" in res.columns
    assert "gate_mirna" in res.columns
    assert "gate_meth" in res.columns

    # Probabilities must be valid in [0, 1]
    probs = res["probability_unfavorable"].values
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Gate weights must sum to approximately 1.0
    gates = res[["gate_mrna", "gate_mirna", "gate_meth"]].values
    np.testing.assert_allclose(gates.sum(axis=1), np.ones(3), atol=1e-3)


def test_batch_invariance_single_vs_batch_of_3():
    """Test (i): Same patient alone vs inside a batch of 3 gives identical probability (atol 1e-5)."""
    demo_dir = ROOT / "data" / "demo"
    if not (demo_dir / "demo_mrna.csv").exists():
        pytest.skip("Demo omics data not available")

    mrna = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0)
    mirna = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0)
    meth = pd.read_csv(demo_dir / "demo_meth.csv", index_col=0)

    # Batch of 3
    res_batch = predict(mrna, mirna, meth)
    p_batch_first = float(res_batch["probability_unfavorable"].iloc[0])

    # Patient alone
    mrna_single = mrna.iloc[[0]]
    mirna_single = mirna.iloc[[0]]
    meth_single = meth.iloc[[0]]

    res_single = predict(mrna_single, mirna_single, meth_single)
    p_single = float(res_single["probability_unfavorable"].iloc[0])

    assert np.isclose(p_batch_first, p_single, atol=1e-5), (
        f"Batch probability {p_batch_first} differs from single patient {p_single}"
    )


def test_missing_checkpoint_raises_error(tmp_path):
    """Test (ii): Missing checkpoint raises FileNotFoundError."""
    demo_dir = ROOT / "data" / "demo"
    if not (demo_dir / "demo_mrna.csv").exists():
        pytest.skip("Demo omics data not available")

    mrna = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0).iloc[[0]]
    mirna = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0).iloc[[0]]
    meth = pd.read_csv(demo_dir / "demo_meth.csv", index_col=0).iloc[[0]]

    non_existent = tmp_path / "non_existent_model.pt"
    with pytest.raises(FileNotFoundError, match="not found"):
        predict(mrna, mirna, meth, weights_path=non_existent)


def test_bio_bias_non_none_after_loading():
    """Test (iii): bio_bias buffer is non-None after model loading."""
    from src.inference.predict import load_inference_artifacts

    _, _, model = load_inference_artifacts()
    assert hasattr(model, "bio_bias"), "Model has no bio_bias attribute"
    assert model.bio_bias is not None, "model.bio_bias is None after loading"
    assert isinstance(model.bio_bias, torch.Tensor), "model.bio_bias is not a Tensor"
    assert model.bio_bias.shape == (577, 577), f"Expected shape (577, 577), got {model.bio_bias.shape}"

