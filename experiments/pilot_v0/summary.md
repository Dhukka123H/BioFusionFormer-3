# BioFusionFormer PILOT v0 Benchmark Results (Preliminary)

**Source:** task-1411 (preliminary run before Step A correctness fixes)
**Cohort:** TARGET-WT, N=121 (Core 3-omics)
**Evaluation:** 25 Outer Test Folds (5 repeats × 5 folds, splits_5x5.json)
**Status:** Frozen PILOT v0 — DO NOT EDIT

## Summary of Pilot v0 Performance

| Experiment | Name | Mean AUROC | 95% CI | AUPRC | Bal Acc | Macro F1 | Brier Score |
|---|---|---|---|---|---|---|---|
| E1 | BioFusionFormer (full 3-omics, bias ON, gate ON) | **0.8960** | [0.8767, 0.9154] | 0.8313 | 0.8305 | 0.8168 | 0.1227 |
| E2 | mRNA only (same architecture) | **0.8780** | [0.8574, 0.8986] | 0.7893 | 0.8186 | 0.7938 | 0.1358 |
| E3 | mRNA + miRNA (no methylation) | **0.8969** | [0.8770, 0.9167] | 0.8281 | 0.8367 | 0.8360 | 0.1231 |
| E4 | No pathway attention bias (full attention) | **0.8870** | [0.8676, 0.9063] | 0.7789 | 0.7943 | 0.7805 | 0.1474 |
| E5 | RANDOM PRIOR control | **0.8417** | [0.8144, 0.8690] | 0.7483 | 0.7493 | 0.7463 | 0.1652 |
| E6 | Broad miRNA mask | **0.8881** | [0.8690, 0.9073] | 0.8158 | 0.8303 | 0.8120 | 0.1373 |
| E7 | Mean fusion instead of gated fusion | **0.8626** | [0.8429, 0.8823] | 0.8096 | 0.8050 | 0.7994 | 0.1335 |
| E8 | Vital status exploratory | **0.5630** | [0.5410, 0.5851] | 0.4520 | 0.5428 | 0.5406 | 0.2466 |

## Key Observations from Pilot v0
1. E1 achieved AUROC ~0.8984 with the initial un-deduplicated feature space.
2. 762 duplicate miRNA features were redundantly modeled.
3. Over 58% of miRNAs had zero pathway links and were unmodeled.
4. Step A correctness fixes were mandated before conducting the final v1 run.
