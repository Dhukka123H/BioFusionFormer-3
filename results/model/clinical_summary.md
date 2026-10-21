# Clinical Modality Integration Report (Stage 5.3)

**Date:** 2026-10-20  
**Evaluation:** 15 CV folds (3 repeats x 5 folds) strictly matched across all variants.  
**Zero-Leakage Guarantee:** Imputation and scaling performed strictly within training folds. Histology/survival columns blacklisted.  

## 1. Summary Metrics & Nadeau-Bengio Intervals

| Model Variant | Modalities | Mean AUROC | Nadeau-Bengio 95% CI | Mean AUPRC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **E1 (Frozen Omics)** | mRNA + miRNA + Meth | 0.8681 | [0.8060, 0.9303] | — | — |
| **EHR2** | 3 omics + primary clinical [age, sex, stage] | **0.8506** | [0.7448, 0.9564] | 0.7962 | 0.1528 |
| **EHR4** | clinical-only through same architecture | **0.6214** | [0.4639, 0.7789] | 0.4851 | 0.2486 |

## 2. Statistical Comparisons vs Frozen Omics E1 (Paired Differences)

Per AGENT_PROMPT Rule 5, comparisons use the Nadeau-Bengio corrected resampled $t$-test, the Wilcoxon signed-rank test, and bootstrap 95% CIs on paired fold differences.

| Comparison | Delta AUROC (Mean) | Nadeau-Bengio $p$-value | Wilcoxon $p$-value | Bootstrap 95% CI | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EHR2 vs E1** | -0.0175 | 0.7227 | 0.5097 | [-0.0591, +0.0247] | not distinguishable with $N=121$ |
| **EHR4 vs E1** | -0.2467 | 0.0113 | 0.0007 | [-0.3192, -0.1726] | statistically significant drop |

## 3. Clinical-Only Baseline Comparison (EHR4 vs Elastic-Net)

- **EHR4 (Neural Architecture)** Mean AUROC: **0.6214**
- **Elastic-Net Clinical-Only Baseline** Mean AUROC: **0.6073**
- Paired Difference (EHR4 - EN): +0.0140 (Nadeau-Bengio $p = 0.7943$)
- **Conclusion:** The neural clinical-only encoder is not distinguishable from the linear elastic-net baseline with $N=121$.

## 4. Key Scientific Findings

1. **Clinical Modality Added Value:** As hypothesized, adding low-dimensional clinical stage, age, and sex to high-dimensional multi-omics does not yield a statistically significant gain over E1 with $N=121$ (Wilms tumor histology is predominantly molecularly driven).
2. **Sensitivity Features (EHR2b):** Adding race, ethnicity, and biopsy tissue percentages shows no noticeable gain over primary clinical features and is designated sensitivity-only.
3. **Clinical-Only Performance:** Clinical features alone achieve an AUROC in the ~0.60–0.65 range, confirming that omics measurements capture the vast majority of histology-discriminating signal.
