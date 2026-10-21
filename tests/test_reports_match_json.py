"""
tests/test_reports_match_json.py
================================
Tests that results/FINAL_STATUS.md, docs/MODEL_CARD.md, and README.md
numerically agree with results/model/E1_metrics.json and that outdated figures
(62,450 free parameters, Bit-identical) are properly expunged or clarified.
"""

import json
import re
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def e1_metrics():
    path = ROOT / "results" / "model" / "E1_metrics.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_final_status_matches_json(e1_metrics):
    final_status = (ROOT / "results" / "FINAL_STATUS.md").read_text(encoding="utf-8")

    # 1. AUROC 0.8716
    assert "0.8716" in final_status, "FINAL_STATUS.md missing verified AUROC 0.8716"

    # 2. AUPRC ~0.8332 (fold mean from json)
    folds = e1_metrics["folds"]
    auprc_mean = f"{np.mean(folds['auprc']):.4f}"
    assert auprc_mean in final_status, f"FINAL_STATUS.md missing verified AUPRC {auprc_mean}"

    # 3. Balanced accuracy ~0.7653
    bal_mean = f"{np.mean(folds['bal_acc']):.4f}"
    assert bal_mean in final_status, f"FINAL_STATUS.md missing verified Balanced Accuracy {bal_mean}"

    # 4. Macro-F1 ~0.7697
    f1_mean = f"{np.mean(folds['f1']):.4f}"
    assert f1_mean in final_status, f"FINAL_STATUS.md missing verified Macro-F1 {f1_mean}"

    # 5. Brier fold mean ~0.1436 and pooled ~0.1365
    brier_fold = f"{e1_metrics['brier_foldmean']:.4f}"
    brier_pool = f"{e1_metrics['brier_pooled']:.4f}"
    assert brier_fold in final_status, f"FINAL_STATUS.md missing fold-mean Brier {brier_fold}"
    assert brier_pool in final_status, f"FINAL_STATUS.md missing pooled Brier {brier_pool}"

    # Pooled metric must not carry an NB interval
    for line in final_status.splitlines():
        if "Brier Score (pooled)" in line:
            assert "—" in line or "-" in line or "N/A" in line or "no CI" in line.lower()
            assert "[" not in line, "Pooled Brier must not carry a confidence interval"

    # 6. Parameter counts: 47,221 free and 4,249,284 dense
    assert "47,221" in final_status
    assert "4,249,284" in final_status

    # 7. No claims that 62,450 is free parameter count
    if "62,450" in final_status:
        assert "could not be reproduced" in final_status

    # 8. No "Bit-identical" claims
    assert "Bit-identical" not in final_status


def test_model_card_matches_json(e1_metrics):
    model_card = (ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")

    assert "0.8716" in model_card
    folds = e1_metrics["folds"]
    assert f"{np.mean(folds['auprc']):.4f}" in model_card
    assert f"{np.mean(folds['bal_acc']):.4f}" in model_card
    assert f"{np.mean(folds['f1']):.4f}" in model_card
    assert f"{e1_metrics['brier_foldmean']:.4f}" in model_card
    assert f"{e1_metrics['brier_pooled']:.4f}" in model_card

    assert "47,221" in model_card
    assert "4,249,284" in model_card


def test_readme_matches_json(e1_metrics):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "0.8716" in readme
    assert "47,221" in readme
    assert "4,249,284" in readme
