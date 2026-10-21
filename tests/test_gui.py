"""
tests/test_gui.py
=================
Tests for Stage D: GUI validators, end-to-end predict() on demo data, and Streamlit AppTest smoke test.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
from src.inference.predict import predict, validate_omics_inputs

DEMO_DIR = ROOT / "app" / "demo_data"


@pytest.fixture
def demo_data():
    assert (DEMO_DIR / "demo_mrna.csv").exists(), "Missing demo_mrna.csv"
    assert (DEMO_DIR / "demo_mirna.csv").exists(), "Missing demo_mirna.csv"
    assert (DEMO_DIR / "demo_meth.csv").exists(), "Missing demo_meth.csv"

    mrna = pd.read_csv(DEMO_DIR / "demo_mrna.csv", index_col=0)
    mirna = pd.read_csv(DEMO_DIR / "demo_mirna.csv", index_col=0)
    meth = pd.read_csv(DEMO_DIR / "demo_meth.csv", index_col=0)
    return mrna, mirna, meth


def test_validator_success(demo_data):
    mrna, mirna, meth = demo_data
    expected_feats = {
        "mrna": list(mrna.columns),
        "mirna": list(mirna.columns),
        "meth": list(meth.columns),
    }
    aligned, warnings = validate_omics_inputs(mrna, mirna, meth, expected_feats)
    assert len(aligned) == 3
    assert len(aligned["mrna"]) == 3


def test_validator_missing_modality(demo_data):
    mrna, mirna, _ = demo_data
    with pytest.raises(ValueError, match="Missing required modality"):
        validate_omics_inputs(mrna, mirna, None, {"mrna": [], "mirna": [], "meth": []})


def test_validator_negative_values(demo_data):
    mrna, mirna, meth = demo_data
    bad_mrna = mrna.copy()
    bad_mrna.iloc[0, 0] = -5.0
    expected_feats = {
        "mrna": list(mrna.columns),
        "mirna": list(mirna.columns),
        "meth": list(meth.columns),
    }
    with pytest.raises(ValueError, match="negative values"):
        validate_omics_inputs(bad_mrna, mirna, meth, expected_feats)


def test_validator_invalid_methylation_range(demo_data):
    mrna, mirna, meth = demo_data
    bad_meth = meth.copy()
    bad_meth.iloc[0, 0] = 2.5
    expected_feats = {
        "mrna": list(mrna.columns),
        "mirna": list(mirna.columns),
        "meth": list(meth.columns),
    }
    with pytest.raises(ValueError, match="out of range"):
        validate_omics_inputs(mrna, mirna, bad_meth, expected_feats)


def test_validator_versioned_ensembl(demo_data):
    mrna, mirna, meth = demo_data
    bad_mrna = mrna.copy()
    bad_mrna.columns = [f"{c}.1" if str(c).startswith("ENSG") else c for c in bad_mrna.columns]
    expected_feats = {
        "mrna": list(mrna.columns),
        "mirna": list(mirna.columns),
        "meth": list(meth.columns),
    }
    with pytest.raises(ValueError, match="version suffixes"):
        validate_omics_inputs(bad_mrna, mirna, meth, expected_feats)


def test_validator_coverage_refusal(demo_data):
    mrna, mirna, meth = demo_data
    # Drop half the columns
    dropped_mrna = mrna.iloc[:, :len(mrna.columns) // 2]
    expected_feats = {
        "mrna": list(mrna.columns),
        "mirna": list(mirna.columns),
        "meth": list(meth.columns),
    }
    with pytest.raises(ValueError, match="Insufficient feature coverage"):
        validate_omics_inputs(dropped_mrna, mirna, meth, expected_feats)


def test_predict_end_to_end(demo_data):
    mrna, mirna, meth = demo_data
    df_out = predict(mrna, mirna, meth)

    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) == 3
    for col in ["patient_id", "probability_unfavorable", "gate_mrna", "gate_mirna", "gate_meth"]:
        assert col in df_out.columns

    # Check probabilities are in [0, 1]
    probs = df_out["probability_unfavorable"].values
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Check gates sum to ~1
    gates_sum = df_out["gate_mrna"] + df_out["gate_mirna"] + df_out["gate_meth"]
    np.testing.assert_allclose(gates_sum.values, 1.0, atol=1e-3)


def test_streamlit_app_smoke():
    app_path = str(ROOT / "app" / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Streamlit app raised an unhandled exception: {at.exception}"
