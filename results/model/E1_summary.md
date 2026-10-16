# E1 summary (frozen v1 configuration)

- Config signature: `a42642001739`; folds used: **25** (n_train~97, n_test~24); seeds per fold: 3
- Source of numbers: per-fold checkpoints in `experiments/final_v1/E1/folds/` (nothing is re-tuned here).

## Discrimination

| Metric | Mean over folds |
|---|---|
| AUROC | **0.8716** |
| AUPRC (prevalence 0.322) | 0.8332 |
| Balanced accuracy | 0.7653 |
| Macro-F1 | 0.7697 |
| Brier (mean of per-fold) | 0.1436 |
| Brier (pooled out-of-fold, predictions averaged over repeats) | 0.1365 |

AUROC 95% interval, **Nadeau-Bengio corrected**: [0.8023, 0.9408] (half-width 0.0693).
Naive interval that treats overlapping folds as independent: [0.8470, 0.8961] (half-width 0.0245) - **too narrow; do not quote it**.
Per-repeat mean AUROC: 0.876, 0.861, 0.867, 0.898, 0.856 (SD across repeats 0.017).

## Comparison with baselines on the same folds (Nadeau-Bengio corrected paired test)

| Baseline | Baseline AUROC | E1 - baseline | t | p |
|---|---|---|---|---|
| mRNA (ElasticNet) | 0.9169 | -0.0453 | -1.06 | 0.2996 |
| Early Fusion (ElasticNet) | 0.9295 | -0.0580 | -1.39 | 0.1778 |
| miRNA (ElasticNet) | 0.8481 | +0.0235 | 0.50 | 0.6209 |
| Methylation (ElasticNet) | 0.4601 | +0.4114 | 6.46 | 0.0000 |
| Clinical-only (Age + Gender) | 0.6296 | +0.2419 | 3.43 | 0.0022 |

p >= 0.05 means 'not distinguishable with N=121', not 'equal'.

## Modality gate (seed-42 model of each fold, test patients)

| Modality | Favorable | Unfavorable |
|---|---|---|
| mRNA | 0.321 +/- 0.035 | 0.323 +/- 0.032 |
| miRNA | 0.399 +/- 0.051 | 0.379 +/- 0.050 |
| Methylation | 0.281 +/- 0.041 | 0.298 +/- 0.046 |

## Attention rollout from [CLS] (summed over modalities, mean over test patients)

Largest / smallest pathway weight = **1.06x**. If this is close to 1, attention does not discriminate between pathways and should not be used as an explanation; use gradient-based attribution instead.

- mRNA 3'-end processing (R-HSA-72187): 0.00396
- Eukaryotic Translation Initiation (R-HSA-72613): 0.00396
- mRNA Splicing (R-HSA-72172): 0.00396
- SRP-dependent cotranslational protein targeting to membrane (R-HSA-1799339): 0.00395
- mRNA Splicing - Major Pathway (R-HSA-72163): 0.00395
- mRNA Polyadenylation (R-HSA-9770562): 0.00394
- Signaling by CTNNB1 phospho-site mutants (R-HSA-4839743): 0.00394
- Mismatch Repair (R-HSA-5358508): 0.00394
- Diseases of DNA Double-Strand Break Repair (R-HSA-9675136): 0.00394
- Cap-dependent Translation Initiation (R-HSA-72737): 0.00394

## Limitations that must accompany any quoted number

- N = 121 patients from one enriched, high-risk cohort; no independent test set.
- Omics matrices are Xena-derived (see data/DATA_CARD.json) unless the GDC-native rebuild has been completed.
- Pathway selection (top-variance, unsupervised) used all 121 patients' omics, no labels.
- E1 was first run once with an unseeded initialisation (AUROC 0.846, discarded) and then re-run with seeding (this result). See results/model/forking_paths.md.