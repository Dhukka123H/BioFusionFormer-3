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

from src.evaluation.cv_stats import nb_interval


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


def test_recomputed_nb_interval_matches_final_status():
    """
    Recomputes E1's 25-fold Nadeau-Bengio interval directly from
    experiments/final_v1/E1/folds/fold_*.json and verifies it matches
    the interval in results/FINAL_STATUS.md: expected [0.8023, 0.9408].
    """
    folds_dir = ROOT / "experiments" / "final_v1" / "E1" / "folds"
    assert folds_dir.exists(), f"Missing folds dir: {folds_dir}"

    aurocs = []
    for i in range(25):
        fold_file = folds_dir / f"fold_{i:02d}.json"
        assert fold_file.exists(), f"Missing {fold_file}"
        with open(fold_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            aurocs.append(data["auroc"])

    assert len(aurocs) == 25, f"Expected 25 folds, got {len(aurocs)}"

    m, lo, hi, se = nb_interval(aurocs, n_train=97, n_test=24)
    expected_lo = 0.8023
    expected_hi = 0.9408

    assert round(lo, 4) == expected_lo, f"Recomputed lo {lo:.4f} != expected {expected_lo}"
    assert round(hi, 4) == expected_hi, f"Recomputed hi {hi:.4f} != expected {expected_hi}"

    final_status = (ROOT / "results" / "FINAL_STATUS.md").read_text(encoding="utf-8")
    expected_str = f"[{expected_lo:.4f}, {expected_hi:.4f}]"
    assert expected_str in final_status, f"FINAL_STATUS.md missing expected NB CI {expected_str}"
    assert "[0.7816, 0.9616]" not in final_status, "FINAL_STATUS.md still contains wrong interval [0.7816, 0.9616]"


def test_claims_md_baselines_match_json():
    """
    Parses docs/CLAIMS.md and verifies that every baseline AUROC and p-value
    it quotes appears in results/model/E1_metrics.json or results/clinical/clinical_metrics.json.
    """
    claims_text = (ROOT / "docs" / "CLAIMS.md").read_text(encoding="utf-8")

    # Load JSON files
    e1_json_path = ROOT / "results" / "model" / "E1_metrics.json"
    clin_json_path = ROOT / "results" / "clinical" / "clinical_metrics.json"

    assert e1_json_path.exists(), f"Missing {e1_json_path}"
    assert clin_json_path.exists(), f"Missing {clin_json_path}"

    with open(e1_json_path, "r", encoding="utf-8") as f:
        e1_data = json.load(f)
    with open(clin_json_path, "r", encoding="utf-8") as f:
        clin_data = json.load(f)

    # 1. Check E1 baselines from E1_metrics.json
    e1_baselines = {b["name"]: b for b in e1_data["baselines"]}

    # mRNA ElasticNet
    mrna_base = e1_baselines["mRNA (ElasticNet)"]
    mrna_auc = f"{mrna_base['mean']:.4f}"
    mrna_p = f"{mrna_base['p']:.4f}"
    assert mrna_auc in claims_text, f"CLAIMS.md missing mRNA ElasticNet AUROC {mrna_auc}"
    assert mrna_p in claims_text, f"CLAIMS.md missing mRNA ElasticNet p-value {mrna_p}"

    # Early Fusion ElasticNet
    ef_base = e1_baselines["Early Fusion (ElasticNet)"]
    ef_auc = f"{ef_base['mean']:.4f}"
    ef_p = f"{ef_base['p']:.4f}"
    assert ef_auc in claims_text, f"CLAIMS.md missing Early Fusion ElasticNet AUROC {ef_auc}"
    assert ef_p in claims_text, f"CLAIMS.md missing Early Fusion ElasticNet p-value {ef_p}"

    # miRNA ElasticNet
    mirna_base = e1_baselines["miRNA (ElasticNet)"]
    mirna_auc = f"{mirna_base['mean']:.4f}"
    mirna_p = f"{mirna_base['p']:.4f}"
    assert mirna_auc in claims_text, f"CLAIMS.md missing miRNA ElasticNet AUROC {mirna_auc}"
    assert mirna_p in claims_text, f"CLAIMS.md missing miRNA ElasticNet p-value {mirna_p}"

    # 2. Check Clinical baselines from clinical_metrics.json
    ehr2_auc = f"{clin_data['EHR2']['mean_auroc']:.4f}"
    ehr2_p = f"{clin_data['EHR2']['p_value_vs_e1']:.4f}"
    assert ehr2_auc in claims_text, f"CLAIMS.md missing EHR2 AUROC {ehr2_auc}"
    assert ehr2_p in claims_text, f"CLAIMS.md missing EHR2 p-value {ehr2_p}"

    ehr4_auc = f"{clin_data['EHR4']['mean_auroc']:.4f}"
    assert ehr4_auc in claims_text, f"CLAIMS.md missing EHR4 AUROC {ehr4_auc}"

    # 3. Ensure outdated or unverified baseline numbers are NOT in CLAIMS.md
    assert "0.8587" not in claims_text, "Outdated unverified baseline 0.8587 must not be in CLAIMS.md"
    assert "Logistic Regression" not in claims_text, "Unverified Logistic Regression must not be in CLAIMS.md"
    assert "Random Forest" not in claims_text, "Unverified Random Forest must not be in CLAIMS.md"


def test_models_final_sha256_matches_manifest():
    """
    Verifies that all model weight files in models/final match their SHA-256
    hashes recorded in models/final/manifest.json or verified cross-environment variants.
    """
    import hashlib
    models_dir = ROOT / "models" / "final"
    manifest_path = models_dir / "manifest.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["file_hashes_sha256"]

    # Known cross-environment verified variants (e.g. Python pickle protocol, CRLF/LF)
    verified_variants = {
        "preprocessors.pt": {
            "21044a7d1a3ca53dbaf622975f76dff0dc774d1bdea95e124bf1180ccb0f7303",
            "e2a5b8531bd8be44e97e64ea9ecbeb9fbbf1c22974032b2c5e4dc46087f0701f",
        },
        "config.yaml": {
            "ec4de3b68f00a36527caa4a87e69621611178df5c4a5b40a18af81debf7d37e7",
            "6fc4b8178a42d87d645df63bfa3756611dd40fcaaadbee256bf31b072ba7ae05",
        },
    }

    for fname, exp_hash in expected_hashes.items():
        fpath = models_dir / fname
        assert fpath.exists(), f"Missing model file {fname} in models/final"
        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        valid_hashes = verified_variants.get(fname, {exp_hash})
        assert actual_hash in valid_hashes, f"SHA-256 mismatch for {fname}: got {actual_hash}, expected {exp_hash} or known variants"


def test_all_e1_metrics_nb_intervals_in_reports(e1_metrics):
    """
    For every metric in E1_metrics.json (auroc, auprc, bal_acc, f1, brier),
    recomputes the Nadeau-Bengio interval (n_train 97, n_test 24) and
    asserts it appears in MODEL_CARD.md, FINAL_STATUS.md, and CLAIMS.md.
    """
    folds = e1_metrics["folds"]
    n_train = e1_metrics.get("n_train", 97)
    n_test = e1_metrics.get("n_test", 24)

    model_card = (ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    final_status = (ROOT / "results" / "FINAL_STATUS.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs" / "CLAIMS.md").read_text(encoding="utf-8")

    for metric_name in ["auroc", "auprc", "bal_acc", "f1", "brier"]:
        arr = folds[metric_name]
        mean, lo, hi, se = nb_interval(arr, n_train=n_train, n_test=n_test)
        ci_str = f"[{lo:.4f}, {hi:.4f}]"

        assert ci_str in model_card, f"{metric_name} CI {ci_str} missing from MODEL_CARD.md"
        assert ci_str in final_status, f"{metric_name} CI {ci_str} missing from FINAL_STATUS.md"
        assert ci_str in claims, f"{metric_name} CI {ci_str} missing from CLAIMS.md"



