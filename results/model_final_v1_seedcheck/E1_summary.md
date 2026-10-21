# E1 summary (frozen v1 configuration)

- Config signature: `c33ad9b8ad20`; folds used: **5** (n_train~97, n_test~24); seeds per fold: 3
- Source of numbers: per-fold checkpoints in `experiments/final_v1/E1/folds/` (nothing is re-tuned here).

## Discrimination

| Metric | Mean over folds |
|---|---|
| AUROC | **0.8653** |
| AUPRC (prevalence 0.322) | 0.7895 |
| Balanced accuracy | 0.7693 |
| Macro-F1 | 0.7740 |
| Brier (mean of per-fold) | 0.1712 |
| Brier (pooled out-of-fold, predictions averaged over repeats) | 0.1711 |

AUROC 95% interval, **Nadeau-Bengio corrected**: [0.7047, 1.0260] (half-width 0.1607).
Naive interval that treats overlapping folds as independent: [0.7895, 0.9412] (half-width 0.0758) - **too narrow; do not quote it**.
Per-repeat mean AUROC: 0.865 (SD across repeats nan).

## Comparison with baselines on the same folds (Nadeau-Bengio corrected paired test)

| Baseline | Baseline AUROC | E1 - baseline | t | p |
|---|---|---|---|---|
| mRNA (ElasticNet) | 0.9093 | -0.0440 | -1.16 | 0.3091 |
| Early Fusion (ElasticNet) | 0.9143 | -0.0490 | -1.56 | 0.1943 |
| miRNA (ElasticNet) | 0.8323 | +0.0330 | 0.72 | 0.5133 |
| Methylation (ElasticNet) | 0.4588 | +0.4065 | 8.48 | 0.0011 |
| Clinical-only (Age + Gender) | 0.6084 | +0.2570 | 6.41 | 0.0030 |

p >= 0.05 means 'not distinguishable with N=121', not 'equal'.

## Modality gate (seed-42 model of each fold, test patients)

| Modality | Favorable | Unfavorable |
|---|---|---|
| mRNA | 0.333 +/- 0.010 | 0.329 +/- 0.008 |
| miRNA | 0.311 +/- 0.028 | 0.308 +/- 0.030 |
| Methylation | 0.357 +/- 0.026 | 0.363 +/- 0.026 |

## Attention rollout from [CLS] (summed over modalities, mean over test patients)

Largest / smallest pathway weight = **1.09x**. If this is close to 1, attention does not discriminate between pathways and should not be used as an explanation; use gradient-based attribution instead.

- mRNA 3'-end processing (R-HSA-72187): 0.00417
- Cap-dependent Translation Initiation (R-HSA-72737): 0.00413
- PELO:HBS1L and ABCE1 dissociate a ribosome on a non-stop mRNA (R-HSA-9954714): 0.00408
- mRNA Splicing (R-HSA-72172): 0.00408
- Mitochondrial translation initiation (R-HSA-5368286): 0.00406
- mRNA Splicing - Major Pathway (R-HSA-72163): 0.00406
- Scavenging of heme from plasma (R-HSA-2168880): 0.00406
- Nonsense Mediated Decay (NMD) independent of the Exon Junction Complex (EJC) (R-HSA-975956): 0.00405
- L13a-mediated translational silencing of Ceruloplasmin expression (R-HSA-156827): 0.00405
- Amplification  of signal from unattached  kinetochores via a MAD2  inhibitory signal (R-HSA-141444): 0.00404

## Limitations that must accompany any quoted number

- N = 121 patients from one enriched, high-risk cohort; no independent test set.
- Omics matrices are Xena-derived (see data/DATA_CARD.json) unless the GDC-native rebuild has been completed.
- Pathway selection (top-variance, unsupervised) used all 121 patients' omics, no labels.
- E1 was first run once with an unseeded initialisation (AUROC 0.846, discarded) and then re-run with seeding (this result). See results/model/forking_paths.md.