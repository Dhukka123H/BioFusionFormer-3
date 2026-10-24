# BioFusionFormer E1 — Seed Sensitivity Audit (Stage C0)

**Date:** 2026-10-22  
**Scope:** Random initialization sensitivity analysis comparing default seeds `[42, 123, 999]` against alternate initialization seeds `[7, 8, 9]`.

---

## 1. Cross-Validated Seed Sensitivity Benchmark (Stage C0)

To test sensitivity to random initialization (Peer Review Finding F2), Repeat 0 (5 folds) was evaluated with alternate initialization seeds `[7, 8, 9]` (`experiments/final_v1_seedcheck/E1/folds/`) and compared fold-for-fold against the baseline frozen run (`experiments/final_v1/E1/folds/`):

| Fold | Frozen Run AUROC (Seeds 42, 123, 999) | Seed Check AUROC (Seeds 7, 8, 9) | $\Delta$ AUROC (SeedCheck − Frozen) |
| :---: | :---: | :---: | :---: |
| **Fold 00** | 0.8611 | 0.8750 | +0.0139 |
| **Fold 01** | 0.8438 | 0.9375 | +0.0938 |
| **Fold 02** | 0.8655 | 0.9580 | +0.0924 |
| **Fold 03** | 0.8571 | 0.7983 | -0.0588 |
| **Fold 04** | 0.9531 | 0.7578 | -0.1953 |

### Benchmark Metrics:
- **Mean AUROC (Seeds 7, 8, 9):** **`0.8653`** (Nadeau-Bengio 95% CI: `[0.7047, 1.0260]`)
- **Mean AUROC (Frozen Seeds 42, 123, 999):** **`0.8761`**
- **Mean Fold Difference ($\Delta$ AUROC):** **`-0.0108`**
- **Standard Deviation of Fold Differences:** **`0.121`**

> [!IMPORTANT]
> **Scientific Finding (F2 Resolution):** The per-fold AUROC difference between seed sets has SD **0.121**; mean AUROC initialisation uncertainty is about $\pm 0.05$, so differences below 0.05 are not interpretable. The SD is estimated from 5 folds only (95% interval 0.07-0.35), so the true 25-fold range could be anywhere from about +/-0.03 to +/-0.14. The 5-fold seed-check CI is not meaningful (upper bound above 1). Brier on the same 5 folds changed by +0.036 between seed sets (0.135 vs 0.171).

---

## 2. Verification of Seed-Specific Weight Initialization

To verify that `--seeds` properly reaches `torch.manual_seed()` and alters model initialization before training, the `BioFusionFormerModel` architecture was initialized with seeds 7, 8, and 9:

- **Seed 7 first 5 mRNA-encoder weights:** `[+0.003124, -0.026940, +0.014240, +0.014033, -0.023903]`
- **Seed 8 first 5 mRNA-encoder weights:** `[+0.008759, +0.030884, +0.039928, -0.018199, +0.001234]`
- **Seed 9 first 5 mRNA-encoder weights:** `[+0.013934, -0.017708, -0.001798, +0.024809, +0.037384]`

All parameter tensors are distinct across seeds, confirming proper initialization stochasticity.

---

## 3. In-Sample Training Fit Audit (Whole Cohort Deployment Models)

The three whole-cohort models (`models/final/model_seed42.pt`, `model_seed123.pt`, `model_seed999.pt`) trained on all 121 patients were also evaluated in-sample across the cohort to confirm training convergence:

| Comparison | Pearson $r$ | Spearman $\rho$ | Mean $|\Delta p|$ | 95th %ile $|\Delta p|$ | Max $|\Delta p|$ | Binary Concordance (thr=0.5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42 vs Seed 123** | 1.0000 | 0.7732 | 0.0055 | 0.0088 | 0.0100 | 100.0% |
| **Seed 42 vs Seed 999** | 1.0000 | 0.6110 | 0.0022 | 0.0043 | 0.0052 | 100.0% |
| **Seed 123 vs Seed 999** | 1.0000 | 0.6783 | 0.0035 | 0.0059 | 0.0076 | 100.0% |

### Modality Gate Weights Across Seeds (Whole Cohort Fit)
| Random Seed | mRNA Gate Weight | miRNA Gate Weight | Methylation Gate Weight | Modality Gate Spread |
| :---: | :---: | :---: | :---: | :---: |
| Seed 42 | 0.3196 | 0.3535 | 0.3268 | 0.0339 |
| Seed 123 | 0.4360 | 0.3317 | 0.2323 | 0.2037 |
| Seed 999 | 0.4026 | 0.2857 | 0.3117 | 0.1169 |

> [!NOTE]
> **In-Sample Disclaimer:** The high Pearson correlation ($r \approx 1.0000$) in this section reflects near-saturated training fit on $N=121$ (in-sample AUROC = 1.0000, Brier = 0.0009). True generalisation variance is quantified strictly by the cross-validated benchmark in Section 1.
