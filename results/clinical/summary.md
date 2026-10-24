# Clinical Modality Integration Report (Stage 5.3)

**Date:** 2026-10-22  
**Evaluation:** 15 CV folds (3 repeats x 5 folds) strictly matched across all variants.  
**Zero-Leakage Guarantee:** Imputation and scaling performed strictly within training folds. Histology/survival columns blacklisted.  

## 1. Summary Metrics & Nadeau-Bengio Intervals

| Model Variant | Modalities | Mean AUROC | Nadeau-Bengio 95% CI | Mean AUPRC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **E1 (Frozen Omics)** | mRNA + miRNA + Meth | 0.8681 | [0.8060, 0.9303] | 0.8296 | 0.1405 |
| **EHR2** | 3 omics + primary clinical [age, sex, stage] | **0.8506** | [0.7448, 0.9564] | 0.7962 | 0.1528 |
| **EHR4** | clinical-only through same architecture | **0.6214** | [0.4639, 0.7789] | 0.4851 | 0.2486 |
| **Reference (Constant Prevalence)** | Baseline prevalence prior ($p=0.3223$) | 0.5000 | — | 0.3223 | 0.2184 |

## 2. Statistical Comparisons vs Frozen Omics E1 (Paired Differences)

Primary statistical intervals are Nadeau-Bengio corrected resampled $t$-intervals on paired fold differences ($n_{\text{train}}=97, n_{\text{test}}=24$). Bootstrap 95% CIs are reported as naive, descriptive comparisons (assuming independent folds).

| Comparison | Delta AUROC (Mean) | Nadeau-Bengio 95% CI (Primary) | Nadeau-Bengio $p$-value | Wilcoxon $p$-value | Bootstrap 95% CI (Naive, Descriptive) | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EHR2 vs E1** | -0.0175 | [-0.1213, +0.0863] | 0.7227 | 0.5097 | [-0.0591, +0.0247] | not distinguishable with $N=121$ |
| **EHR4 vs E1** | -0.2467 | [-0.4284, -0.0651] | 0.0113 | 0.0007 | [-0.3192, -0.1726] | statistically significant drop |

## 3. Clinical-Only Baseline Comparison (EHR4 vs Elastic-Net & Prevalence)

- **EHR4 (Neural Architecture)** Mean AUROC: **0.6214** (Nadeau-Bengio 95% CI: `[0.4639, 0.7789]`)
- **Elastic-Net Clinical-Only Baseline** Mean AUROC: **0.6073**
- Paired Difference (EHR4 - EN): +0.0140 (Nadeau-Bengio $p = 0.7943$)
- **Brier Score Audit:** EHR4 Brier is **0.2486**, which is **worse than a trivial constant prevalence predictor** (0.2184). Always predicting the cohort prevalence achieves a lower Brier score than EHR4's uncalibrated neural probabilities.
- **Conclusion:** The neural clinical-only encoder is not distinguishable from the linear elastic-net baseline with $N=121$.

## 4. Key Scientific Findings

1. **Clinical Modality Added Value:** EHR2 and E1 are not distinguishable with $N=121$ (paired difference mean $\Delta = -0.0175$, Nadeau-Bengio 95% CI `[-0.1213, +0.0863]`, $p = 0.7227$, Wilcoxon $p = 0.5097$). While no gain is observed, the confidence interval is wide and cannot rule out a clinical gain or loss of up to ~0.09 AUROC.
2. **Clinical-Only Performance:** Clinical features alone (EHR4) achieve a mean AUROC of 0.6214 with a Nadeau-Bengio 95% CI of `[0.4639, 0.7789]`. Because this interval includes 0.50 (chance level), clinical features are not demonstrated to perform above chance with $N=121$.
3. **Calibration Deficit in Clinical-Only Encoder:** EHR4's Brier score (0.2486) is worse than constant prevalence prediction (0.2184 for base rate $p = 0.3223$), indicating severe miscalibration in the clinical-only neural encoder.
4. **Statistical Interval Correction:** The naive bootstrap interval `[-0.0591, +0.0247]` assumes independent folds and artificially overstates precision. The corrected Nadeau-Bengio paired interval is `[-0.1213, +0.0863]`, appropriately reflecting resampled fold dependency.

## 5. Seed Sensitivity Benchmark (Stage C0) & Cross-Machine Audit

### Pairwise Seed Consistency (In-Sample Deployment Models; Not Evidence of Stability)

> **Important Caveat:** The Pearson $r = 1.0000$ values below reflect in-sample predictions from whole-cohort deployment models on their own overfitted training data ($N=121$), measuring memorisation rather than generalisation. This is **not evidence of stability**. On cross-validated test splits, initialization sensitivity produces an SD of fold differences of $0.1210$.

| Comparison | Pearson $r$ | Spearman $\rho$ | Mean $|\Delta p|$ | 95th %ile $|\Delta p|$ | Max $|\Delta p|$ | Binary Concordance (thr=0.5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42 vs Seed 123** | 1.0000 | 0.7732 | 0.0055 | 0.0088 | 0.0100 | 100.0% |
| **Seed 42 vs Seed 999** | 1.0000 | 0.6110 | 0.0022 | 0.0043 | 0.0052 | 100.0% |
| **Seed 123 vs Seed 999** | 1.0000 | 0.6783 | 0.0035 | 0.0059 | 0.0076 | 100.0% |

### Modality Gate Weights Across Seeds (Gate Weights Are Not Stable)

> **Finding on Gate Instability:** Gate weights differ markedly by random seed (mRNA: 0.32 vs 0.44 vs 0.40; Methylation: 0.33 vs 0.23 vs 0.31). Because gating weights fluctuate depending on random seed initialization, gate weights are not stable, and "adaptive gate" cannot be claimed as a stable biological mechanism.

| Random Seed | mRNA Gate Weight | miRNA Gate Weight | Methylation Gate Weight | Modality Gate Spread |
| :---: | :---: | :---: | :---: | :---: |
| Seed 42 | 0.3196 | 0.3535 | 0.3268 | 0.0339 |
| Seed 123 | 0.4360 | 0.3317 | 0.2323 | 0.2037 |
| Seed 999 | 0.4026 | 0.2857 | 0.3117 | 0.1169 |

### Cross-Machine Fold-0 Audit (`pcb_check`)

- **Status:** Not run on this machine.

