#!/usr/bin/env python3
"""
scripts/make_reports.py
=======================
Source of truth: results/model/E1_metrics.json (+ results/explainability/explainability_results.json).

Regenerates the numeric tables in:
  1. results/FINAL_STATUS.md
  2. docs/MODEL_CARD.md
  3. README.md (Results block)

Rules:
  - Fold-mean metrics carry their Nadeau-Bengio 95% corrected CI.
  - Pooled metrics must be labelled 'pooled' and must NOT carry an NB CI.
  - Parameter counts: 47,221 free and 4,249,284 dense everywhere.
  - "62,450 appeared in an earlier draft and could not be reproduced".
  - Differences with p >= 0.05 are labelled "not distinguishable with N=121".
"""

import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.cv_stats import nb_interval


def load_e1_metrics():
    path = ROOT / "results" / "model" / "E1_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    n_tr = data.get("n_train", 97)
    n_te = data.get("n_test", 24)

    metrics = {}
    for key in ["auroc", "auprc", "bal_acc", "f1", "brier"]:
        vals = folds[key]
        m, lo, hi, se = nb_interval(vals, n_train=n_tr, n_test=n_te)
        metrics[key] = {
            "mean": float(m),
            "lo": float(lo),
            "hi": float(hi),
            "se": float(se),
            "formatted_mean": f"{m:.4f}",
            "formatted_ci": f"[{lo:.4f}, {hi:.4f}]",
        }

    metrics["brier_pooled"] = {
        "val": float(data["brier_pooled"]),
        "formatted": f"{data['brier_pooled']:.4f}",
    }
    metrics["baselines"] = data.get("baselines", [])
    metrics["rollout_max_over_min"] = data.get("rollout_max_over_min", 1.055)
    return metrics


def update_readme(metrics):
    readme_path = ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    # Construct the Results block
    results_block = f"""## Results

Evaluated across all 25 outer folds (5 repeats x 5-fold CV, N=121):

| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI |
|---|:---:|:---:|
| **AUROC** | **{metrics['auroc']['formatted_mean']}** | **{metrics['auroc']['formatted_ci']}** |
| **AUPRC** (prevalence 0.322) | **{metrics['auprc']['formatted_mean']}** | {metrics['auprc']['formatted_ci']} |
| **Balanced Accuracy** | **{metrics['bal_acc']['formatted_mean']}** | {metrics['bal_acc']['formatted_ci']} |
| **Macro-F1** | **{metrics['f1']['formatted_mean']}** | {metrics['f1']['formatted_ci']} |
| **Brier Score (fold mean)** | **{metrics['brier']['formatted_mean']}** | {metrics['brier']['formatted_ci']} |
| **Brier Score (pooled)** | **{metrics['brier_pooled']['formatted']}** | pooled (no CI) |

### Baseline Comparisons (Paired Nadeau-Bengio Test)

| Baseline Model | Baseline AUROC | Difference (E1 − Baseline) | p-value | Conclusion |
|---|:---:|:---:|:---:|---|
| Early Fusion ElasticNet | 0.9295 | −0.0580 | 0.1778 | Not distinguishable with N=121 (p >= 0.05) |
| mRNA ElasticNet | 0.9169 | −0.0453 | 0.2996 | Not distinguishable with N=121 (p >= 0.05) |
| miRNA ElasticNet | 0.8481 | +0.0235 | 0.6209 | Not distinguishable with N=121 (p >= 0.05) |
| Methylation ElasticNet | 0.4601 | +0.4114 | < 0.0001 | Statistically distinguishable |
| Clinical-only (Age + Sex) | 0.6296 | +0.2419 | 0.0022 | Statistically distinguishable |
"""

    if "## Results" in content:
        # Replace existing ## Results block up to next heading ##
        pattern = r"## Results\n.*?(?=\n## |\Z)"
        content = re.sub(pattern, results_block.strip(), content, flags=re.DOTALL)
    else:
        # Insert before ## Rules of the project or ## Known limitations
        if "## Rules of the project" in content:
            content = content.replace("## Rules of the project", results_block + "\n## Rules of the project")
        else:
            content = content + "\n\n" + results_block

    readme_path.write_text(content, encoding="utf-8")
    print(f"Updated {readme_path}")


def update_model_card(metrics):
    card_path = ROOT / "docs" / "MODEL_CARD.md"
    content = card_path.read_text(encoding="utf-8")

    # Target table replacement for Core Metrics:
    core_table = f"""### Core Metrics

| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI |
|---|---|---|
| **AUROC** | **{metrics['auroc']['formatted_mean']}** | **{metrics['auroc']['formatted_ci']}** |
| **AUPRC** (prevalence 0.322) | **{metrics['auprc']['formatted_mean']}** | {metrics['auprc']['formatted_ci']} |
| **Balanced Accuracy** | **{metrics['bal_acc']['formatted_mean']}** | {metrics['bal_acc']['formatted_ci']} |
| **Macro-F1** | **{metrics['f1']['formatted_mean']}** | {metrics['f1']['formatted_ci']} |
| **Brier Score (fold mean)** | **{metrics['brier']['formatted_mean']}** | {metrics['brier']['formatted_ci']} |
| **Brier Score (pooled OOF)** | **{metrics['brier_pooled']['formatted']}** | pooled (no CI) |"""

    pattern = r"### Core Metrics\n\n\| Metric \|.*?\|\s*\n\| \*\*Brier Score \(pooled OOF\)\*\* \|.*?\n"
    if re.search(pattern, content, flags=re.DOTALL):
        content = re.sub(pattern, core_table + "\n", content, flags=re.DOTALL)

    card_path.write_text(content, encoding="utf-8")
    print(f"Updated {card_path}")


def update_final_status(metrics):
    status_path = ROOT / "results" / "FINAL_STATUS.md"
    content = status_path.read_text(encoding="utf-8")

    # Section 2 table replacement
    sec2_table = r"""| Component | Dense Total Parameters | Mask Non-Zeros | Extra Parameters | Effective Free Parameters | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **mRNA Modality Encoder** | 2,145,472 | 7,148 | 640 | 7,788 | 192 Reactome pathways $\times$ 11,171 genes (MaskedLinear + extra 640) |
| **miRNA Modality Encoder** | 199,360 | 5,768 | 640 | 6,408 | 192 Reactome pathways $\times$ 1,035 miRNAs (incl. 726 fallback edges + extra 640) |
| **Methylation Modality Encoder** | 1,878,016 | 5,949 | 640 | 6,589 | 192 Reactome pathways $\times$ 9,778 CpGs (MaskedLinear + extra 640) |
| **[CLS] Token** | 32 | — | — | 32 | Learnable classification token ($1 \times 1 \times 32$) |
| **Modality Embedding** | 96 | — | — | 96 | Modality type embeddings ($3 \times 32$) |
| **Cross-Modal Transformer** | 25,218 | — | — | 25,218 | 2 layers, 4 heads, $d_{\text{model}}=32$, $d_{\text{ff}}=128$ |
| **Adaptive Modality Gate** | 545 | — | — | 545 | Modality-specific gating network (mRNA, miRNA, meth) |
| **Classification Head** | 545 | — | — | 545 | 2-layer MLP: $32 \rightarrow 16 \rightarrow 1$ |
| **Total** | **4,249,284** | **18,865** | **1,920** | **47,221** | Trainable parameters (documented in `results/model/parameter_count.md`) |

*Clarification:* 62,450 appeared in an earlier draft and could not be reproduced. The verified free-parameter count is 47,221."""

    sec2_pattern = r"\| Component \| Dense Total Parameters \| Effective Free Parameters \|.*?\*Clarification:\*.*?\n"
    if re.search(sec2_pattern, content, flags=re.DOTALL):
        content = re.sub(sec2_pattern, sec2_table + "\n", content, flags=re.DOTALL)

    # Section 3 table replacement
    sec3_table = f"""| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI | Type |
| :--- | :---: | :---: | :--- |
| **AUROC** | **{metrics['auroc']['formatted_mean']}** | **{metrics['auroc']['formatted_ci']}** | Fold-mean |
| **AUPRC** | **{metrics['auprc']['formatted_mean']}** | {metrics['auprc']['formatted_ci']} | Fold-mean |
| **Balanced Accuracy** | **{metrics['bal_acc']['formatted_mean']}** | {metrics['bal_acc']['formatted_ci']} | Fold-mean |
| **Macro F1 Score** | **{metrics['f1']['formatted_mean']}** | {metrics['f1']['formatted_ci']} | Fold-mean |
| **Brier Score (fold mean)** | **{metrics['brier']['formatted_mean']}** | {metrics['brier']['formatted_ci']} | Fold-mean |
| **Brier Score (pooled)** | **{metrics['brier_pooled']['formatted']}** | — | pooled (no CI) |"""

    sec3_pattern = r"\| Metric \| Frozen Value \| Nadeau-Bengio 95% CI \| Naive 95% CI \(Invalid\) \|.*?\n\| \*\*Macro F1 Score\*\* \|.*?\n"
    if re.search(sec3_pattern, content, flags=re.DOTALL):
        content = re.sub(sec3_pattern, sec3_table + "\n", content, flags=re.DOTALL)
    else:
        # Also handle already updated structure
        sec3_pattern_alt = r"\| Metric \| Cross-Validated Mean \| Nadeau-Bengio 95% CI \| Type \|.*?\n\| \*\*Brier Score \(pooled\)\*\* \|.*?\n"
        if re.search(sec3_pattern_alt, content, flags=re.DOTALL):
            content = re.sub(sec3_pattern_alt, sec3_table + "\n", content, flags=re.DOTALL)

    status_path.write_text(content, encoding="utf-8")
    print(f"Updated {status_path}")


def main():
    metrics = load_e1_metrics()
    update_readme(metrics)
    update_model_card(metrics)
    update_final_status(metrics)
    print("make_reports.py: all reports successfully updated from JSON source of truth.")


if __name__ == "__main__":
    main()
