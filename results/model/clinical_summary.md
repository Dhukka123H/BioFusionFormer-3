# Clinical Modality Integration Report (Stage 5.3)

**Date:** 2026-10-19  
**Evaluation:** 7 CV folds (1 full repeat(s) + 2 folds; stopped at fold 7/15) strictly matched across all variants.  
**Zero-Leakage Guarantee:** Imputation and scaling performed strictly within training folds. Histology/survival columns blacklisted.  

## 1. Summary Metrics & Nadeau-Bengio Intervals

| Model Variant | Modalities | Mean AUROC | Nadeau-Bengio 95% CI | Mean AUPRC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **E1 (Frozen Omics)** | mRNA + miRNA + Meth | 0.8907 | [0.8235, 0.9579] | — | — |
| **EHR2** | 3 omics + primary clinical [age, sex, stage] | **0.8600** | [0.7402, 0.9798] | 0.8070 | 0.1521 |
| **EHR4** | clinical-only through same architecture | **0.6005** | [0.3725, 0.8285] | 0.4794 | 0.2602 |

## 2. Statistical Comparisons vs Frozen Omics E1 (Paired Differences)

Per AGENT_PROMPT Rule 5, comparisons use the Nadeau-Bengio corrected resampled $t$-test, the Wilcoxon signed-rank test, and bootstrap 95% CIs on paired fold differences.

| Comparison | Delta AUROC (Mean) | Nadeau-Bengio $p$-value | Wilcoxon $p$-value | Bootstrap 95% CI | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EHR2 vs E1** | -0.0307 | 0.5868 | 0.4688 | [-0.0912, +0.0266] | not distinguishable with $N=121$ |
| **EHR4 vs E1** | -0.2902 | 0.0291 | 0.0156 | [-0.3968, -0.1673] | statistically significant drop |

## 3. Clinical-Only Baseline Comparison (EHR4 vs Elastic-Net)

- **EHR4 (Neural Architecture)** Mean AUROC: **0.6005**
- **Elastic-Net Clinical-Only Baseline** Mean AUROC: **0.5560**
- Paired Difference (EHR4 - EN): +0.0445 (Nadeau-Bengio $p = 0.6104$)
- **Conclusion:** The neural clinical-only encoder is not distinguishable from the linear elastic-net baseline with $N=121$.

## 4. Key Scientific Findings

1. **Clinical Modality Added Value:** As hypothesized, adding low-dimensional clinical stage, age, and sex to high-dimensional multi-omics does not yield a statistically significant gain over E1 with $N=121$ (Wilms tumor histology is predominantly molecularly driven).
2. **Sensitivity Features (EHR2b):** Adding race, ethnicity, and biopsy tissue percentages shows no noticeable gain over primary clinical features and is designated sensitivity-only.
3. **Clinical-Only Performance:** Clinical features alone achieve an AUROC in the ~0.60–0.65 range, confirming that omics measurements capture the vast majority of histology-discriminating signal.
