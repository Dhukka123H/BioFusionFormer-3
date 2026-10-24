# Model Card — BioFusionFormer E1 (Frozen v1 Configuration)

## Model Overview
- **Model Name:** BioFusionFormer E1
- **Architecture:** Pathway-Aware Hierarchical Cross-Modal Transformer (Pre-LayerNorm, 2 layers, 4 heads, $d_{\text{model}}=32$, $d_{\text{ff}}=128$, dropout 0.3, attention dropout 0.2, modality dropout 0.2).
- **Modalities:** mRNA gene expression (11,171 genes), miRNA expression (1,035 unique precursors), DNA methylation promoter mean beta values (9,778 genes).
- **Target Task:** Binary classification of Wilms tumor histologic subtype: **Favorable Histology (FH = 0)** vs **Unfavorable Histology / Anaplasia (UH = 1)**.
- **Config Signature:** `a42642001739` (`configs/default.yaml`).

---

## Intended Use & Clinical Scope

> [!CAUTION]
> **Research Prototype Only — Not for Clinical Decisions**:
> This model was trained on an enriched, retrospectively collected cohort ($N=121$) with no external clinical validation. It is not cleared, approved, or validated for making diagnostic, staging, prognostic, or treatment decisions.

- **Intended Use:** Exploratory bioinformatics research, cross-modal biological representation learning, and pathway-level attribution analysis in pediatric nephroblastoma.
- **Out of Scope:** Clinical triage, replacement of histopathological biopsy review by a pediatric pathologist, or survival prediction.

---

## Training Cohort & Provenance

- **Cohort:** NCI TARGET-WT (Wilms tumor), $N = 121$ primary solid tumor patients with complete, aligned 3-omics data.
- **Enrichment:** High-risk selection bias: 82 Favorable Histology (67.8%), 39 Unfavorable Histology / Anaplasia (32.2%). The unfavorable-histology share here (32%) is higher than the roughly 10% usually reported [reference needed].
- **Data Lineage:**
  - Omics matrices (mRNA, miRNA, methylation) are GDC-derived data obtained through the **UCSC Xena GDC hub** (hg38, Xena-standardized pipelines).
  - Clinical labels and metadata were obtained directly from the **NCI GDC API** (Data Release 46.0 - August 10, 2026).
  - Unsupervised pathway selection (top 192 variance Reactome pathways) was performed on all 121 patients without labels.

---

## Model Parameters (Dense vs. Free)

| Parameter Metric | Count | Interpretation |
|---|---|---|
| **Dense trainable parameters** (`count_parameters()`) | **4,249,284** (~4.25M) | Full memory footprint of MaskedLinear weight matrices. |
| **Effective free parameters** (`count_effective_parameters()`) | **47,221** (~47.2k) | Degrees of freedom receiving gradients (inside biological mask). |
| Encoders | 4,222,848 dense (18,865 mask non-zeros) | Masked biology projections. |
| Transformer backbone | 25,218 | 2 layers, 4 heads, $d_{\text{model}}=32$. |
| Fusion gate | 545 | AdaptiveModalityGate MLP. |
| Classification head | 545 | 2-layer MLP head. |

---

## Cross-Validated Performance

All metrics are reported across **Repeated Stratified 5-Fold Cross-Validation** (5 repeats $\times$ 5 folds = **25 outer test folds**). **In-sample metrics are strictly excluded.**

Confidence intervals use the **Nadeau-Bengio corrected resampled $t$-interval** ($\sqrt{1/K + n_{\text{test}}/n_{\text{train}}} = 0.536$ for $K=25$), properly accounting for fold overlap.

### Core Metrics

| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI |
|---|---|---|
| **AUROC** | **0.8716** | **[0.8023, 0.9408]** |
| **AUPRC** (prevalence 0.322) | **0.8332** | [0.7434, 0.9230] |
| **Balanced Accuracy** | **0.7653** | [0.6821, 0.8486] |
| **Macro-F1** | **0.7697** | [0.6839, 0.8554] |
| **Brier Score (fold mean)** | **0.1436** | [0.1089, 0.1783] |
| **Brier Score (pooled OOF)** | **0.1365** | — |

*Per-repeat mean AUROCs:* 0.876, 0.861, 0.867, 0.898, 0.856 (run-to-run SD: 0.017).
*Seed Sensitivity (Initialization Uncertainty on Identical Splits):* A 5-fold rerun on repeat 0 with alternative random seeds `[7, 8, 9]` yielded mean AUROC **0.8653** (mean per-fold $\Delta \text{AUROC} = -0.0108$, range $[-0.1953, +0.0938]$). The per-fold AUROC difference between seed sets has SD **0.121**; mean AUROC initialisation uncertainty is about $\pm 0.05$, so differences below 0.05 are not interpretable. The SD is estimated from 5 folds only (95% interval 0.07-0.35), so the true 25-fold range could be anywhere from about +/-0.03 to +/-0.14. The 5-fold seed-check CI is not meaningful (upper bound above 1). Brier on the same 5 folds changed by +0.036 between seed sets (0.135 vs 0.171).

### Benchmark Comparison (Paired Nadeau-Bengio Corrected Test)

| Baseline Model | Baseline AUROC | Difference (E1 − Baseline) | $t$-statistic | $p$-value | Conclusion |
|---|---|---|---|---|---|
| Early Fusion ElasticNet | 0.9295 | −0.0580 | −1.39 | 0.1778 | Not distinguishable with $N=121$ ($p \ge 0.05$) |
| mRNA ElasticNet | 0.9169 | −0.0453 | −1.06 | 0.2996 | Not distinguishable with $N=121$ ($p \ge 0.05$) |
| miRNA ElasticNet | 0.8481 | +0.0235 | +0.50 | 0.6209 | Not distinguishable with $N=121$ ($p \ge 0.05$) |
| Methylation ElasticNet | 0.4601 | +0.4114 | +6.46 | $< 0.0001$ | Statistically significant improvement |
| Clinical-only (Age + Sex) | 0.6296 | +0.2419 | +3.43 | 0.0022 | Statistically significant improvement |

> [!NOTE]
> Differences with $p \ge 0.05$ indicate that with $N=121$, performance is **not statistically distinguishable** from high-dimensional penalized logistic regression on mRNA.

---

## Modality Gating & Attention Rollout

- **Adaptive Modality Gate:** Test-set weights are relatively uniform across patients:
  - mRNA: $0.321 \pm 0.035$
  - miRNA: $0.399 \pm 0.051$
  - Methylation: $0.281 \pm 0.041$
- **Attention Rollout:** Ratio of max pathway attention to min pathway attention is **1.06x** (near uniform). Consequently, attention rollout does not discriminate between pathways; **Integrated Gradients** attribution must be used instead.

---

## Known Limitations & Documented Architectural Issues

1. **Small Enriched Cohort:** $N=121$ from a single pediatric trial cohort (TARGET-WT); wide confidence intervals ($[0.80, 0.94]$).
2. **Fork 6 — Unmapped miRNA Token Routing:** 726 miRNA features without Reactome edges were mapped to the last pathway token index ($P-1$), rather than an isolated fallback token.
3. **Fork 8 — Missing Modalities at Inference:** The frozen E1 model requires all three omics at inference time and raises `KeyError` if any modality is missing.
4. **DNA Methylation Signal:** High-dimensional HM450 promoter beta values alone display near-random discrimination ($\text{AUROC} \approx 0.46$) in Wilms tumor histology.
