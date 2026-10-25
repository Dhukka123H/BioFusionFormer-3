#!/usr/bin/env python3
"""
scripts/make_reports.py
=======================
Stage B1: Generate verified reports and documentation from JSON source-of-truth.

Source of truth:
  - results/model/E1_metrics.json
  - results/explainability/explainability_results.json (if available)

Regenerates the numeric tables of:
  - results/FINAL_STATUS.md
  - docs/MODEL_CARD.md
  - README.md ("Results" block)

Rules:
  - Fold-mean metrics keep their Nadeau-Bengio CI.
  - Pooled metrics must be labelled "pooled" and must NOT carry an NB CI.
  - Parameter counts: 47,221 (free), 4,249,284 (dense).
  - 62,450: "appeared in an earlier draft and could not be reproduced".
  - Status labels: "numerically identical (max |dp| 1.8e-5)" instead of "Bit-identical".
"""

import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.cv_stats import nb_interval

E1_METRICS_PATH = ROOT / "results" / "model" / "E1_metrics.json"
EXPLAIN_JSON_PATH = ROOT / "results" / "explainability" / "explainability_results.json"
FINAL_STATUS_PATH = ROOT / "results" / "FINAL_STATUS.md"
MODEL_CARD_PATH = ROOT / "docs" / "MODEL_CARD.md"
README_PATH = ROOT / "README.md"


def load_e1_metrics():
    if not E1_METRICS_PATH.exists():
        sys.exit(f"Missing {E1_METRICS_PATH}. Run scripts/run_e1.py --report first.")
    with open(E1_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_explain_metrics():
    if EXPLAIN_JSON_PATH.exists():
        with open(EXPLAIN_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def format_ci(ci):
    if len(ci) == 4:
        return f"[{ci[1]:.4f}, {ci[2]:.4f}]"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def make_final_status(e1, explain):
    folds = e1["folds"]
    n_train = e1.get("n_train", 97)
    n_test = e1.get("n_test", 24)

    auroc_mean = float(e1["auroc_mean"])
    auroc_ci = e1["auroc_nb_ci"]

    auprc_arr = np.array(folds["auprc"])
    auprc_mean = float(np.mean(auprc_arr))
    auprc_ci = nb_interval(auprc_arr, n_train=n_train, n_test=n_test)

    bal_arr = np.array(folds["bal_acc"])
    bal_mean = float(np.mean(bal_arr))
    bal_ci = nb_interval(bal_arr, n_train=n_train, n_test=n_test)

    f1_arr = np.array(folds["f1"])
    f1_mean = float(np.mean(f1_arr))
    f1_ci = nb_interval(f1_arr, n_train=n_train, n_test=n_test)

    brier_fold_arr = np.array(folds["brier"])
    brier_fold_mean = float(e1["brier_foldmean"])
    brier_fold_ci = nb_interval(brier_fold_arr, n_train=n_train, n_test=n_test)
    brier_pooled = float(e1["brier_pooled"])

    lines = []
    lines.append("# BioFusionFormer — Final Audit & Pipeline Status (E1-Only)")
    lines.append("")
    lines.append("**Date:** 2026-10-24  ")
    lines.append("**Cohort:** TARGET-WT ($N=121$, Favorable Histology $N=82$, Unfavorable Histology $N=39$)  ")
    lines.append("**Target:** Wilms Tumor Histology (FH=0 vs UH=1)  ")
    lines.append("**Hardware & Environment:** Windows, CPU (`torch.set_num_threads(2)`):  ")
    lines.append("- **Frozen E1 Benchmark:** Python 3.14.7 | torch 2.14.0+cpu, source `logs/final_run_v1_all_sessions.log` (sessions from 2026-10-06 22:01)  ")
    lines.append("- **Explainability & Null Permutations:** Python 3.12.10 | torch 2.14.1+cpu (machine: `DESKTOP-H6DTFG6` (Windows-10-10.0.19045-SP0) as in `results/explainability/summary.md`)  ")
    lines.append("- **Reproducibility Note:** The reproducibility re-run used the same Python/torch as the frozen run; cross-environment reproducibility was not tested.  ")
    lines.append("- **Clinical Multimodal Integration:** Python 3.14.7 | torch 2.14.0+cpu (`logs/clinical_session_20261020_232506.log`)  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("This repository has completed a thorough, end-to-end scientific and code audit under the guidelines of `docs/AGENT_PROMPT.md` and `docs/AGENT_PROMPT_2.md` on git branch `final/e1-gui`.")
    lines.append("The repository cleanly focuses on the single frozen core architecture: **BioFusionFormer E1** (mRNA + miRNA + DNA Methylation, pathway attention bias ON, adaptive modality gating ON, strong Reactome mask).")
    lines.append("")
    lines.append("All computations were executed with strict adherence to the project rules:")
    lines.append("- **Zero-Leakage Cross-Validation:** All imputation (median) and standardization (z-score) are performed strictly inside training folds. Pathway selection (192, variance-based) used all 121 patients' omics, no labels.")
    lines.append("- **Valid Statistical Inference:** All confidence intervals and hypothesis tests use the Nadeau-Bengio corrected resampled $t$-interval ($K=25$, $n_{\\text{train}}=97$, $n_{\\text{test}}=24$), completely rejecting naive $\\pm 1.96 \\text{SD}/\\sqrt{K}$ intervals.")
    lines.append("- **Honest Scientific Claims:** Differences where $p \\ge 0.05$ are reported plainly as 'not distinguishable with $N=121$'. Attention rollout is noted as secondary and near-uniform (spread ratio 1.06x). Representation projections are documented as in-sample visualizations only (no generalization claims).")
    lines.append("- **Correct Lineage Framing:** Data origins reflect UCSC Xena GDC-hub derived omics and official GDC API clinical records.")
    lines.append("- **Strict CPU Resource Limits:** Never exceeding 2 CPU threads (`torch.set_num_threads(2)`).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Architecture & Exact Parameter Counts")
    lines.append("")
    lines.append("The parameter count discrepancy identified in Fork 7 and Finding F5 has been rigorously resolved:")
    lines.append("- **Dense Total Trainable Parameters:** **4,249,284** (~4.25M)")
    lines.append("- **Effective Free Parameters (non-zero by mask):** **47,221** (~47.2k)")
    lines.append("")
    lines.append("| Component | Dense Total Parameters | Effective Free Parameters | Description |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append("| **mRNA Modality Encoder** | 2,145,024 | 7,788 | 192 Reactome pathways $\\times$ 11,171 genes (7,148 mask non-zeros + 640 overhead) |")
    lines.append("| **miRNA Modality Encoder** | 198,912 | 6,408 | 192 Reactome pathways $\\times$ 1,035 unique miRNAs (5,768 mask non-zeros + 640 overhead) |")
    lines.append("| **Methylation Modality Encoder** | 1,877,568 | 6,589 | 192 Reactome pathways $\\times$ 9,778 CpG promoter means (5,949 mask non-zeros + 640 overhead) |")
    lines.append("| **Modality Embeddings + [CLS]** | 128 | 128 | 3 $\\times$ 32 modality embeddings + 32 [CLS] token |")
    lines.append("| **Cross-Modal Transformer** | 25,218 | 25,218 | 2 layers, 4 heads, $d_{\\text{model}}=32$, $d_{\\text{ff}}=128$ |")
    lines.append("| **Adaptive Modality Gate** | 545 | 545 | Modality-specific gating network (mRNA, miRNA, meth) |")
    lines.append("| **Classification Head** | 545 | 545 | 2-layer MLP: $32 \\rightarrow 16 \\rightarrow 1$ |")
    lines.append("| **Total** | **4,249,284** | **47,221** | Trainable parameters (documented in `results/model/parameter_count.md`) |")
    lines.append("")
    lines.append("*Clarification:* 62,450 appeared in an earlier draft and could not be reproduced. The verified free-parameter count is 47,221.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Cross-Validation Benchmark Performance (E1 Frozen)")
    lines.append("")
    lines.append("Evaluated across all 25 outer folds ($5 \\times 5$-fold CV) with seeds `[42, 123, 999]`:")
    lines.append("")
    lines.append("| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI | Type |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append(f"| **AUROC** | **{auroc_mean:.4f}** | **{format_ci(auroc_ci)}** | Fold mean |")
    lines.append(f"| **AUPRC** | **{auprc_mean:.4f}** | {format_ci(auprc_ci)} | Fold mean |")
    lines.append(f"| **Balanced Accuracy** | **{bal_mean:.4f}** | {format_ci(bal_ci)} | Fold mean |")
    lines.append(f"| **Macro F1 Score** | **{f1_mean:.4f}** | {format_ci(f1_ci)} | Fold mean |")
    lines.append(f"| **Brier Score (fold mean)** | **{brier_fold_mean:.4f}** | {format_ci(brier_fold_ci)} | Fold mean |")
    lines.append(f"| **Brier Score (pooled)** | **{brier_pooled:.4f}** | — | Pooled out-of-fold |")
    lines.append("")
    lines.append("- **Adaptive Modality Gate Weights:** mRNA: **0.321**, miRNA: **0.399**, Methylation: **0.281**.")
    lines.append("- **Attention Rollout Spread:** Max/min ratio = **1.06x** (near uniform; attention rollout cannot be claimed as an explanation mechanism).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Reproducibility Audit (Stage 3)")
    lines.append("")
    lines.append("A full 5-fold re-run was performed under tag `final_v1_repro` (`models/e1_folds_final_v1_repro/`):")
    lines.append("")
    lines.append("| Fold | Frozen AUROC | Re-run AUROC | Absolute $\\Delta$ AUROC | Max $|\\Delta \\text{Prob}|$ | Status |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
    lines.append("| **Fold 00** | 0.8611 | 0.8611 | 0.000000 | $7.03 \\times 10^{-6}$ | numerically identical (max |dp| 1.8e-5) |")
    lines.append("| **Fold 01** | 0.8438 | 0.8438 | 0.000000 | $1.77 \\times 10^{-5}$ | numerically identical (max |dp| 1.8e-5) |")
    lines.append("| **Fold 02** | 0.8655 | 0.8655 | 0.000000 | $4.80 \\times 10^{-6}$ | numerically identical (max |dp| 1.8e-5) |")
    lines.append("| **Fold 03** | 0.8571 | 0.8571 | 0.000000 | $9.15 \\times 10^{-6}$ | numerically identical (max |dp| 1.8e-5) |")
    lines.append("| **Fold 04** | 0.9531 | 0.9531 | 0.000000 | $1.14 \\times 10^{-5}$ | numerically identical (max |dp| 1.8e-5) |")
    lines.append("")
    lines.append("- **Verification:** Run-to-run variability under identical seeds is bounded by machine precision ($< 1.77 \\times 10^{-5}$ in probabilities, $\\Delta \\text{AUROC} = 0.000000$). The reproducibility re-run used the same Python/torch as the frozen run; cross-environment reproducibility was not tested.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Seed Sensitivity Benchmark (Stage C0)")
    lines.append("")
    lines.append("To test sensitivity to random initialization (Finding F2), 5 folds were retrained with new seeds `[7, 8, 9]` (`final_v1_seedcheck`):")
    lines.append("- **Mean AUROC:** **0.8653**")
    lines.append("- **Mean Fold Difference (vs Frozen E1 seeds 42, 123, 999):** $\\Delta \\text{AUROC} = -0.0108$")
    lines.append("- **Standard Deviation of Fold Differences:** **0.121**")
    lines.append("- **Conclusion:** The per-fold AUROC difference between seed sets has SD 0.121; mean AUROC initialisation uncertainty is about $\\pm 0.05$; differences below 0.05 are not interpretable. The SD is estimated from 5 folds only (95% interval 0.07-0.35), so the true 25-fold range could be anywhere from about +/-0.03 to +/-0.14. The 5-fold seed-check CI is not meaningful (upper bound above 1). Brier on the same 5 folds changed by +0.036 between seed sets (0.135 vs 0.171).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Clinical Multimodal Integration Benchmarks (Stage C1 & C2)")
    lines.append("")
    lines.append("Evaluated across 15 folds ($3 \\times 5$-fold CV, seeds `[42, 123, 999]`):")
    lines.append("")
    lines.append("| Model Variant | Mean AUROC | Nadeau-Bengio 95% CI | Mean AUPRC | Mean Brier | Comparison vs E1 / Baseline |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    lines.append("| **E1 (Omics Only)** | **0.8681** | [0.8060, 0.9303] | 0.8296 | 0.1405 | Reference frozen architecture (first 15 folds) |")
    lines.append("| **EHR2 (3 Omics + Age, Sex, Stage)** | **0.8506** | [0.7448, 0.9564] | 0.7962 | 0.1528 | $\\Delta = -0.0175$ vs E1 (NB $p = 0.7227$, Wilcoxon $p = 0.5097$). **Not distinguishable** with $N=121$. |")
    lines.append("| **EHR4 (Clinical-Only Transformer)** | **0.6214** | [0.4639, 0.7789] | 0.4851 | 0.2486 | $\\Delta = +0.0140$ vs Clinical ElasticNet (0.6073, NB $p = 0.7943$). Brier is worse than prevalence baseline. |")
    lines.append("| **Reference (Constant Prevalence)** | 0.5000 | — | 0.3223 | 0.2184 | Baseline prevalence prior ($p=0.3223$). EHR4 Brier (0.2486) is worse than constant guessing. |")
    lines.append("")
    lines.append("- **Paired EHR4 vs E1:** Mean difference $-0.2467$ (NB 95% CI: `[-0.4284, -0.0651]`, NB $p = 0.0113$, Wilcoxon $p = 0.0007$, naive Bootstrap 95% CI: `[-0.3192, -0.1726]`). Omics significantly outperform clinical features alone.")
    lines.append("- **Clinical Modality Added Value:** EHR2 − E1 paired difference has Nadeau-Bengio 95% CI `[-0.1213, +0.0863]`. A clinical gain or loss of up to ~0.09 AUROC cannot be excluded at $N=121$.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Explainability, Stability & Calibration (Stage A)")
    lines.append("")
    lines.append("Attributions evaluated across the 5 saved fold checkpoints with Captum's Integrated Gradients ($n_{\\text{steps}}=50$):")
    lines.append("1. **ODC Fallback Artefact:** Regulation of ornithine decarboxylase (ODC) (`R-HSA-350562`) was index $P-1$ where 726 unmapped miRNAs were routed. Its earlier high score was strictly an artefact of fallback token routing.")
    if explain and "size_bias" in explain and "stability_jaccard_top20" in explain:
        sb = explain["size_bias"]
        stab = explain["stability_jaccard_top20"]
        lines.append(f"2. **Dual Aggregations & Size Dominance:** Attributions are dominated by pathway size (Spearman $\\rho = {sb['spearman_genuine_sum']:.4f}$) and by translation/ribosome/splicing genes; after size normalisation stability is not above the matched null.")
        lines.append(f"3. **Permutation Null Stability:** At the **pathway level**, IG genuine sum stability is {stab['ig_genuine_sum']:.4f} (null mean {stab['null_matched_sum_mean']:.4f}, values: 0.491, 0.288, 0.397) and size-normalised stability is {stab['ig_size_norm']:.4f} (null mean {stab['null_matched_norm_mean']:.4f}, values: 0.372, 0.231, 0.259). At the **cluster level** (independent pathway clusters, Jaccard $\\ge 0.5$), stability is 0.462 (sum) and 0.437 (size-normalised) vs permutation nulls (0.43, 0.26, 0.31) and (0.36, 0.19, 0.23); with 3 permutations, no p-value is possible.")
    else:
        lines.append("2. **Dual Aggregations:** Attributions are reported side-by-side as Metric (i) Genuine Sum and Metric (ii) Size-Normalised Mean.")
        lines.append("3. **Permutation Null Stability:** Matched null permutation architecture evaluated.")
    if explain and "recalibration" in explain:
        cal = explain["recalibration"].get("ensemble_repeat_averaged", explain["recalibration"])
        lines.append(f"4. **Recalibration (Platt Scaling):** Unpenalized logistic regression on logit(pooled_oof) yields slope = **{cal['slope']:.3f}** (95% CI [{cal['slope_ci_95'][0]:.2f}, {cal['slope_ci_95'][1]:.2f}]) and intercept = **{cal['intercept']:.3f}** (95% CI [{cal['intercept_ci_95'][0]:.2f}, {cal['intercept_ci_95'][1]:.2f}]). Brier drops from {cal['brier_before']:.4f} to {cal['brier_after_5fold_cv']:.4f} under 5-fold CV.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Review findings and status")
    lines.append("")
    lines.append("| ID | Audit Finding | Severity | Resolution Status | Verified Resolution Detail |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    lines.append("| **F1** | ODC Fallback Routing Artefact | Critical | **RESOLVED** | Documented in `fallback_audit.md`. 726 unmapped miRNAs routed to P-1 caused artificial 4.98 score. Fallback excluded from reporting; dual genuine sum & size-normalised mean attributions reported. |")
    lines.append("| **F2** | Matched label-permutation null | High | **RESOLVED** | Resumable matched null permutation architecture built with atomic cache on disk; 3 permutations x 5 folds evaluated. |")
    lines.append("| **F3** | Reporting consistency across metrics | High | **RESOLVED** | All metrics (AUPRC, Bal Acc, Macro F1, Brier) strictly synchronized with E1_metrics.json and Nadeau-Bengio intervals. |")
    lines.append("| **F4** | Log vs Logit Recalibration Bug | High | **RESOLVED** | Fixed formula from faulty log(p) to logit(p). Calibrator slope = 2.254, intercept = -0.076. Brier reduced to 0.1213. |")
    lines.append("| **F5** | Parameter Count Documentation | High | **RESOLVED** | Verified exact parameters everywhere: 47,221 free parameters, 4,249,284 dense parameters. 62,450 noted as unreproduced earlier draft. |")
    lines.append("| **F6** | Prior significance & guide cleanup | Medium | **RESOLVED** | Removed false claims of prior superiority; E5 control real vs random prior showed no significant difference (p=0.86). |")
    lines.append("| **F7** | Seed variance & reproducibility | High | **measured and documented** | Rerun with seeds [7,8,9] yielded SD of differences = 0.121 (5 folds). True 25-fold range could be anywhere from +/-0.03 to +/-0.14. Differences below 0.05 are not interpretable. |")
    lines.append("| **F8** | In-sample representation & clinical models | High | **RESOLVED** | In-sample PCA clusters documented as training memorisation, not generalisation; clinical models EHR2/EHR4 evaluated across 15 folds. |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Open Architectural Issues & Practical Limitations")
    lines.append("")
    lines.append("1. **Fork 6 Fallback Wiring:** Unmapped miRNAs (726 features) are routed to pathway token index $P-1$ rather than a dedicated sink token. Future architectures should allocate an explicit unmapped token index $P$.")
    lines.append("2. **Fork 8 Missing Modalities:** BioFusionFormer requires all three omic modalities (mRNA, miRNA, Methylation) to be present at inference time. The architecture does not support missing modalities without explicit imputation.")
    lines.append("3. **Small Enriched Cohort:** Evaluated strictly on TARGET-WT ($N=121$, high-risk pediatric renal tumor). Wide Nadeau-Bengio confidence intervals reflect the modest sample size.")
    lines.append("4. **No External Validation:** evaluation is repeated stratified 5-fold cross-validation on TARGET-WT only; no TCGA data and no external cohort.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 10. Permitted vs. Forbidden Scientific Claims")
    lines.append("")
    lines.append("### Claims the Project IS Allowed to Make:")
    lines.append(f"1. BioFusionFormer E1 achieves cross-validated AUROC of **{auroc_mean:.4f}** (Nadeau-Bengio 95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]) on the $N=121$ TARGET-WT histology task.")
    lines.append("2. Omics-based models significantly outperform clinical features alone (AUROC 0.8681 vs 0.6214, paired diff $-0.2467$, $p=0.0113$).")
    lines.append("3. The architecture demonstrates deterministic reproducibility with same-seed variation bounded by $< 1.8 \\times 10^{-5}$ in probabilities.")
    lines.append("4. Integrated Gradients identifies pathways dominated by size and translation/ribosome genes; size normalisation substantially reduces size bias, though stability remains near the matched null.")
    lines.append("5. Platt recalibration on out-of-fold logits reduces Brier score from 0.1365 to 0.1213 under 5-fold cross-validation.")
    lines.append("")
    lines.append("### Claims FORBIDDEN by Empirical Evidence:")
    lines.append("1. **DO NOT CLAIM** that BioFusionFormer outperforms shallow linear baselines (mRNA ElasticNet 0.9169, Early Fusion 0.9295 are statistically indistinguishable with $N=121$).")
    lines.append("2. **DO NOT CLAIM** that adding clinical features (EHR2) improves omics performance ($p=0.7227$, indistinguishable).")
    lines.append("3. **DO NOT CLAIM** that the Reactome biological prior provides statistically significant advantage over random masks (E5 control $p=0.86$).")
    lines.append("4. **DO NOT CLAIM** that attention rollout explains model predictions (spread ratio is 1.06x, near uniform).")
    lines.append("5. **DO NOT CLAIM** that omics predict patient vital status or overall survival (AUROC ~0.50, random guessing).")
    lines.append("6. **DO NOT CLAIM** clinical utility without external multi-center validation cohorts.")
    lines.append("")

    return "\n".join(lines)


def update_model_card(e1):
    if not MODEL_CARD_PATH.exists():
        return
    text = MODEL_CARD_PATH.read_text(encoding="utf-8")

    folds = e1["folds"]
    n_train = e1.get("n_train", 97)
    n_test = e1.get("n_test", 24)

    auroc_mean = float(e1["auroc_mean"])
    auroc_ci = e1["auroc_nb_ci"]
    auprc_ci = nb_interval(folds["auprc"], n_train=n_train, n_test=n_test)
    bal_ci = nb_interval(folds["bal_acc"], n_train=n_train, n_test=n_test)
    f1_ci = nb_interval(folds["f1"], n_train=n_train, n_test=n_test)
    brier_ci = nb_interval(folds["brier"], n_train=n_train, n_test=n_test)
    brier_fold_mean = float(e1["brier_foldmean"])
    brier_pooled = float(e1["brier_pooled"])

    core_metrics_table = (
        "### Core Metrics\n\n"
        "| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI |\n"
        "|---|---|---|\n"
        f"| **AUROC** | **{auroc_mean:.4f}** | **{format_ci(auroc_ci)}** |\n"
        f"| **AUPRC** (prevalence 0.322) | **{auprc_ci[0]:.4f}** | {format_ci(auprc_ci)} |\n"
        f"| **Balanced Accuracy** | **{bal_ci[0]:.4f}** | {format_ci(bal_ci)} |\n"
        f"| **Macro-F1** | **{f1_ci[0]:.4f}** | {format_ci(f1_ci)} |\n"
        f"| **Brier Score (fold mean)** | **{brier_fold_mean:.4f}** | {format_ci(brier_ci)} |\n"
        f"| **Brier Score (pooled OOF)** | **{brier_pooled:.4f}** | — |\n"
    )

    text = re.sub(
        r"### Core Metrics\n\n\| Metric \|.*?\n\| \*\*Brier Score \(pooled OOF\)\*\*.*?\n",
        core_metrics_table,
        text,
        flags=re.DOTALL,
    )

    # Item 4: replace enrichment sentence
    text = re.sub(
        r"(Anaplasia is significantly enriched compared to population incidence \(~10%\)\.|Anaplasia is higher than the roughly 10% usually reported.*?\.|The unfavorable-histology share here \(32%\) is higher than the roughly 10% usually reported.*?\.)",
        "The unfavorable-histology share here (32%) is higher than the roughly 10% usually reported [reference needed].",
        text,
    )

    # Update parameters section if needed
    text = re.sub(
        r"\| \*\*Dense trainable parameters\*\*.*?\n\| \*\*Effective free parameters\*\*.*?\n",
        "| **Dense trainable parameters** (`count_parameters()`) | **4,249,284** (~4.25M) | Full memory footprint of MaskedLinear weight matrices. |\n"
        "| **Effective free parameters** (`count_effective_parameters()`) | **47,221** (~47.2k) | Degrees of freedom receiving gradients (inside biological mask). |\n",
        text,
    )
    # Update seed sensitivity wording if needed
    seed_repl = (
        "*Seed Sensitivity (Initialization Uncertainty on Identical Splits):* A 5-fold rerun on repeat 0 with alternative random seeds `[7, 8, 9]` yielded mean AUROC **0.8653** (mean per-fold $\\Delta \\text{AUROC} = -0.0108$, range $[-0.1953, +0.0938]$). The per-fold AUROC difference between seed sets has SD **0.121**; mean AUROC initialisation uncertainty is about $\\pm 0.05$, so differences below 0.05 are not interpretable. The SD is estimated from 5 folds only (95% interval 0.07-0.35), so the true 25-fold range could be anywhere from about +/-0.03 to +/-0.14. The 5-fold seed-check CI is not meaningful (upper bound above 1). Brier on the same 5 folds changed by +0.036 between seed sets (0.135 vs 0.171).\n"
    )
    text = re.sub(
        r"\*Seed Sensitivity \(Initialization (?:Variance|Uncertainty) on Identical Splits\)\:.*?\n",
        lambda _: seed_repl,
        text,
    )
    MODEL_CARD_PATH.write_text(text, encoding="utf-8")


def update_seed_sensitivity():
    seed_path = ROOT / "results" / "model" / "seed_sensitivity_c0.md"
    if not seed_path.exists():
        return
    text = seed_path.read_text(encoding="utf-8")
    finding_repl = (
        "> [!IMPORTANT]\n"
        "> **Scientific Finding (F2 Resolution):** The per-fold AUROC difference between seed sets has SD **0.121**; "
        "mean AUROC initialisation uncertainty is about $\\pm 0.05$, so differences below 0.05 are not interpretable. "
        "The SD is estimated from 5 folds only (95% interval 0.07-0.35), so the true 25-fold range could be anywhere from "
        "about +/-0.03 to +/-0.14. The 5-fold seed-check CI is not meaningful (upper bound above 1). "
        "Brier on the same 5 folds changed by +0.036 between seed sets (0.135 vs 0.171).\n"
    )
    text = re.sub(
        r"> \[!IMPORTANT\]\n> \*\*Scientific Finding \(F2 Resolution\):\*\*.*?\n",
        lambda _: finding_repl,
        text,
    )
    seed_path.write_text(text, encoding="utf-8")


def update_readme(e1):
    if not README_PATH.exists():
        return
    text = README_PATH.read_text(encoding="utf-8")

    auroc_mean = float(e1["auroc_mean"])
    auroc_ci = e1["auroc_nb_ci"]
    folds = e1["folds"]
    auprc_mean = float(np.mean(folds["auprc"]))
    brier_fold_mean = float(e1["brier_foldmean"])
    brier_pooled = float(e1["brier_pooled"])

    results_block = (
        f"* E1 cross-validated AUROC over the 25 outer folds: **{auroc_mean:.4f}** "
        f"(Nadeau-Bengio 95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]). "
        f"AUPRC: **{auprc_mean:.4f}**. Brier score: **{brier_fold_mean:.4f}** (fold mean), **{brier_pooled:.4f}** (pooled).\n"
        f"* Parameters: **47,221** effective free parameters (`count_effective_parameters()`), **4,249,284** dense trainable parameters.\n"
        f"* E1 is **not distinguishable** from the shallow elastic-net baselines (mRNA elastic net 0.9169, early fusion 0.9295) with N = 121.\n"
        f"* The pathway prior was **not** shown to help: the real-vs-random prior comparison (E5 control) gave p = 0.86.\n"
        f"* Attention rollout is almost uniform over pathways (max/min about 1.06x) and is **not** an explanation; Integrated Gradients is used instead.\n"
        f"* Omics do not predict vital status (AUROC about 0.50). Prognosis is therefore not claimed."
    )

    pattern = r"(## What is the headline result \(read this before quoting any number\)\n\n)(.*?)(\n\n## Layout)"
    new_text = re.sub(pattern, rf"\g<1>{results_block}\g<3>", text, flags=re.DOTALL)
    README_PATH.write_text(new_text, encoding="utf-8")


def main():
    e1 = load_e1_metrics()
    explain = load_explain_metrics()

    final_status_content = make_final_status(e1, explain)
    FINAL_STATUS_PATH.write_text(final_status_content, encoding="utf-8")
    print(f"Updated {FINAL_STATUS_PATH}")

    update_model_card(e1)
    print(f"Updated {MODEL_CARD_PATH}")

    update_seed_sensitivity()
    print("Updated results/model/seed_sensitivity_c0.md")

    update_readme(e1)
    print(f"Updated {README_PATH}")

    print("make_reports.py: Reports successfully generated from JSON source of truth.")


if __name__ == "__main__":
    main()
