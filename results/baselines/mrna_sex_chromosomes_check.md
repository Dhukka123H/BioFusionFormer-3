# mRNA ElasticNet Sex-Chromosome Sensitivity Check

**Date:** 2026-10-24  
**Cohort:** TARGET-WT ($N=121$, 39 UH vs 82 FH)  
**Folds:** 25 Outer Test Folds ($5 \times 5$-fold Repeated Stratified Cross-Validation)  

## Purpose & Background
In the top-20 mRNA feature attribution list for E1, two Y-linked genes (*RPS4Y1* and *RPS4Y2*) appear with high integrated gradients. In this cohort, sex is not associated with histology (Cram&eacute;r's $V = 0.00$, $p = 1.000$). To determine whether model performance leans on sex-linked genes or whether gene signals are distributed across autosomal biology, we evaluated 25-fold ElasticNet models with and without genes on chromosomes X and Y.

## Chromosome Filtering Summary
- **Total mRNA genes:** 11,171
- **Sex-linked genes identified (HGNC):** 2,335 total annotated (chrX: 2,160, chrY: 175)
- **Sex-linked genes present in cohort:** 375 (chrX: 355, chrY: 20)
- **Autosomal genes retained:** 10,796

## Results Across 25 Cross-Validation Folds

| Feature Set | Gene Count | Mean AUROC | Nadeau-Bengio 95% CI | SD Across Folds |
| :--- | :---: | :---: | :---: | :---: |
| **All mRNA Genes** | 11,171 | **0.9357** | [0.8780, 0.9934] | 0.0521 |
| **Autosomal Only (excl. chrX / chrY)** | 10,796 | **0.9329** | [0.8687, 0.9971] | 0.0581 |

### Paired Statistical Comparison (Autosomal - All Genes)
- **Mean Difference ($\Delta$ AUROC):** `-0.0028`
- **Nadeau-Bengio Corrected $t$-statistic:** `-0.41`
- **Nadeau-Bengio $p$-value:** `0.6850` (df = 24)

## Per-Fold Breakdown

| Fold Index | Repeat / Fold | All mRNA AUROC | Autosomal AUROC | Difference ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | R0 Fold 0 | 0.9118 | 0.9338 | +0.0221 |
| 2 | R0 Fold 1 | 1.0000 | 1.0000 | +0.0000 |
| 3 | R0 Fold 2 | 0.9453 | 0.9141 | -0.0312 |
| 4 | R0 Fold 3 | 0.9297 | 0.9219 | -0.0078 |
| 5 | R0 Fold 4 | 0.9664 | 0.9748 | +0.0084 |
| 6 | R1 Fold 0 | 0.9926 | 0.9926 | +0.0000 |
| 7 | R1 Fold 1 | 0.9688 | 0.9609 | -0.0078 |
| 8 | R1 Fold 2 | 0.9609 | 0.9531 | -0.0078 |
| 9 | R1 Fold 3 | 0.8594 | 0.8516 | -0.0078 |
| 10 | R1 Fold 4 | 0.9496 | 0.9496 | +0.0000 |
| 11 | R2 Fold 0 | 0.9412 | 0.9265 | -0.0147 |
| 12 | R2 Fold 1 | 0.9375 | 0.9453 | +0.0078 |
| 13 | R2 Fold 2 | 0.9844 | 0.9844 | +0.0000 |
| 14 | R2 Fold 3 | 0.7734 | 0.7422 | -0.0312 |
| 15 | R2 Fold 4 | 0.9748 | 0.9916 | +0.0168 |
| 16 | R3 Fold 0 | 0.9706 | 0.9706 | +0.0000 |
| 17 | R3 Fold 1 | 0.9141 | 0.8906 | -0.0234 |
| 18 | R3 Fold 2 | 0.9453 | 0.9453 | +0.0000 |
| 19 | R3 Fold 3 | 0.9375 | 0.9375 | +0.0000 |
| 20 | R3 Fold 4 | 0.8824 | 0.8824 | +0.0000 |
| 21 | R4 Fold 0 | 0.9118 | 0.9191 | +0.0074 |
| 22 | R4 Fold 1 | 0.8984 | 0.9062 | +0.0078 |
| 23 | R4 Fold 2 | 0.9766 | 0.9766 | +0.0000 |
| 24 | R4 Fold 3 | 0.8594 | 0.8516 | -0.0078 |
| 25 | R4 Fold 4 | 1.0000 | 1.0000 | +0.0000 |

## Scientific Conclusion
Elastic net AUROC did not change without X/Y genes (0.933 vs 0.936, p = 0.69; this elastic net is not the nested baseline 0.917); this does not show that E1 ignores them; the reason for RPS4Y1/2 attribution is unknown.
