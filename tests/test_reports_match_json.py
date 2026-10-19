"""
tests/test_reports_match_json.py
================================
Validates that results/FINAL_STATUS.md, docs/MODEL_CARD.md, and README.md
accurately match results/model/E1_metrics.json and exact parameter counts.
"""

import json
import re
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
from src.evaluation.cv_stats import nb_interval


@pytest.fixture(scope="module")
def e1_metrics():
    path = ROOT / "results" / "model" / "E1_metrics.json"
    assert path.exists(), f"{path} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    n_tr = data.get("n_train", 97)
    n_te = data.get("n_test", 24)

    metrics = {}
    for k in ["auroc", "auprc", "bal_acc", "f1", "brier"]:
        m, lo, hi, _ = nb_interval(folds[k], n_train=n_tr, n_test=n_te)
        metrics[k] = {
            "mean": f"{m:.4f}",
            "ci": f"[{lo:.4f}, {hi:.4f}]",
        }
    metrics["brier_pooled"] = f"{data['brier_pooled']:.4f}"
    return metrics


def test_readme_matches_json(e1_metrics):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert e1_metrics["auroc"]["mean"] in readme
    assert e1_metrics["auroc"]["ci"] in readme
    assert e1_metrics["auprc"]["mean"] in readme
    assert e1_metrics["auprc"]["ci"] in readme
    assert e1_metrics["bal_acc"]["mean"] in readme
    assert e1_metrics["bal_acc"]["ci"] in readme
    assert e1_metrics["f1"]["mean"] in readme
    assert e1_metrics["f1"]["ci"] in readme
    assert e1_metrics["brier"]["mean"] in readme
    assert e1_metrics["brier"]["ci"] in readme
    assert e1_metrics["brier_pooled"] in readme
    # Verify pooled is labelled pooled and has no interval
    assert "pooled (no CI)" in readme or "pooled; no CI" in readme


def test_model_card_matches_json(e1_metrics):
    card = (ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert e1_metrics["auroc"]["mean"] in card
    assert e1_metrics["auroc"]["ci"] in card
    assert e1_metrics["auprc"]["mean"] in card
    assert e1_metrics["auprc"]["ci"] in card
    assert e1_metrics["bal_acc"]["mean"] in card
    assert e1_metrics["bal_acc"]["ci"] in card
    assert e1_metrics["f1"]["mean"] in card
    assert e1_metrics["f1"]["ci"] in card
    assert e1_metrics["brier"]["mean"] in card
    assert e1_metrics["brier"]["ci"] in card
    assert e1_metrics["brier_pooled"] in card
    assert "47,221" in card
    assert "4,249,284" in card


def test_final_status_matches_json(e1_metrics):
    status = (ROOT / "results" / "FINAL_STATUS.md").read_text(encoding="utf-8")
    assert e1_metrics["auroc"]["mean"] in status
    assert e1_metrics["auroc"]["ci"] in status
    assert e1_metrics["auprc"]["mean"] in status
    assert e1_metrics["auprc"]["ci"] in status
    assert e1_metrics["bal_acc"]["mean"] in status
    assert e1_metrics["bal_acc"]["ci"] in status
    assert e1_metrics["f1"]["mean"] in status
    assert e1_metrics["f1"]["ci"] in status
    assert e1_metrics["brier"]["mean"] in status
    assert e1_metrics["brier"]["ci"] in status
    assert e1_metrics["brier_pooled"] in status
    assert "47,221" in status
    assert "4,249,284" in status
    assert "could not be reproduced" in status


def test_parameters_everywhere():
    # Verify that 47,221 is documented as free parameter count across all target files
    param_doc = (ROOT / "results" / "model" / "parameter_count.md").read_text(encoding="utf-8")
    assert "47,221" in param_doc
    assert "4,249,284" in param_doc
    assert "could not be reproduced" in param_doc

    data_card = (ROOT / "data" / "DATA_CARD.json").read_text(encoding="utf-8")
    assert '"parameter_count": 47221' in data_card
    assert "could not be reproduced" in data_card

    config = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    assert "effective_free_parameters: 47221" in config
    assert "could not be reproduced" in config
