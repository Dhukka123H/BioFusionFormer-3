"""
tests/test_demo_inference.py
============================
Tests end-to-end inference on the exported demo CSV files.
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
from src.inference.predict import predict, validate_omics_inputs

has_model_weights = bool(list((ROOT / "models" / "final").glob("*.pt")))
skip_no_weights = pytest.mark.skipif(
    not has_model_weights,
    reason="weights distributed separately",
)


@skip_no_weights
def test_demo_csv_inference():
    demo_dir = ROOT / "data" / "demo"
    assert (demo_dir / "demo_mrna.csv").exists()
    assert (demo_dir / "demo_mirna.csv").exists()
    assert (demo_dir / "demo_meth.csv").exists()

    mrna = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0)
    mirna = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0)
    meth = pd.read_csv(demo_dir / "demo_meth.csv", index_col=0)

    assert len(mrna) == 3
    assert len(mirna) == 3
    assert len(meth) == 3

    res = predict(mrna, mirna, meth)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert "patient_id" in res.columns
    assert "probability_unfavorable" in res.columns
    assert "gate_mrna" in res.columns
    assert "gate_mirna" in res.columns
    assert "gate_meth" in res.columns

    # Verify probability range
    for p in res["probability_unfavorable"]:
        assert 0.0 <= p <= 1.0


@skip_no_weights
def test_missing_modality_safeguard():
    demo_dir = ROOT / "data" / "demo"
    mrna = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0)
    mirna = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0)

    with pytest.raises(ValueError, match="Missing required modality"):
        predict(mrna, mirna, None)


@skip_no_weights
def test_low_coverage_safeguard():
    demo_dir = ROOT / "data" / "demo"
    mrna = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0)
    mirna = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0)
    meth = pd.read_csv(demo_dir / "demo_meth.csv", index_col=0)

    # Subsample mRNA columns to only 50%
    half_mrna = mrna.iloc[:, : len(mrna.columns) // 2]
    with pytest.raises(ValueError, match="Insufficient feature coverage"):
        predict(half_mrna, mirna, meth)
