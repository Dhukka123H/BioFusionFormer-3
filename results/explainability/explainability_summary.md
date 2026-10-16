# BioFusionFormer E1 — Explainability Summary (Stage 4)

> **Scope**: 5 fold models from the Stage 3 reproducibility re-run (seed 42, `models/e1_folds_final_v1_repro/`). Attributions computed on TEST patients of each fold only.

## 4.1 Integrated Gradients (Primary Attribution)

- n_steps = 50; baseline = zero vector = training-fold mean in standardised space.
- Computed independently for all three modalities; aggregated to 192 pathways via the same prior masks as the model.

### Top-20 pathways by mean |IG| (summed over modalities, 5-fold mean)

1. Regulation of ornithine decarboxylase (ODC) (`R-HSA-350562`): 4.98124
2. mRNA Splicing (`R-HSA-72172`) *(generic)*: 2.50729
3. mRNA Splicing - Major Pathway (`R-HSA-72163`) *(generic)*: 2.48158
4. Eukaryotic Translation Initiation (`R-HSA-72613`) *(generic)*: 2.21289
5. Cap-dependent Translation Initiation (`R-HSA-72737`) *(generic)*: 2.21289
6. GTP hydrolysis and joining of the 60S ribosomal subunit (`R-HSA-72706`) *(generic)*: 2.08484
7. L13a-mediated translational silencing of Ceruloplasmin expression (`R-HSA-156827`) *(generic)*: 2.07988
8. Response of EIF2AK4 (GCN2) to amino acid deficiency (`R-HSA-9633012`): 2.07911
9. mRNA 3'-end processing (`R-HSA-72187`): 1.87507
10. SRP-dependent cotranslational protein targeting to membrane (`R-HSA-1799339`) *(generic)*: 1.82953
11. Formation of a pool of free 40S subunits (`R-HSA-72689`): 1.82599
12. Selenoamino acid metabolism (`R-HSA-2408522`): 1.80149
13. Constitutive Signaling by AKT1 E17K in Cancer (`R-HSA-5674400`): 1.77259
14. Signaling by phosphorylated juxtamembrane, extracellular and kinase domain KIT mutants (`R-HSA-9670439`): 1.75538
15. Signaling by KIT in disease (`R-HSA-9669938`): 1.75538
16. Regulation of RUNX1 Expression and Activity (`R-HSA-8934593`): 1.73377
17. Nonsense Mediated Decay (NMD) independent of the Exon Junction Complex (EJC) (`R-HSA-975956`): 1.73159
18. mRNA Polyadenylation (`R-HSA-9770562`): 1.71133
19. Selenocysteine synthesis (`R-HSA-2408557`): 1.69628
20. PELO:HBS1L and ABCE1 dissociate a ribosome on a non-stop mRNA (`R-HSA-9954714`) *(generic)*: 1.69131

### Per-modality top-3 pathways (5-fold mean)

**mRNA**: mRNA Splicing; mRNA Splicing - Major Pathway; Cap-dependent Translation Initiation
**miRNA**: Regulation of ornithine decarboxylase (ODC); Constitutive Signaling by AKT1 E17K in Cancer; Regulation of RUNX1 Expression and Activity
**Methylation**: Cap-dependent Translation Initiation; Eukaryotic Translation Initiation; GTP hydrolysis and joining of the 60S ribosomal subunit

### Attention rollout (secondary reference only)

Mean max/min ratio: **1.095x** (range [1.056, 1.119]). A ratio near 1 means near-uniform attention; rollout is not a reliable explanation here.

## 4.2 Elastic-Net Baseline

Logistic regression (saga, C=0.5, l1_ratio=0.5, max_iter=500) fitted on mRNA training fold only. Coefficients mapped to pathways via the same mask. **Never fitted on all 121 patients.**

## 4.3 Stability

| Attribution | Mean pairwise Jaccard (top-20, 5 folds) |
|---|---|
| IG (BioFusionFormer) | 0.3885 |
| ElasticNet (mRNA, train-fold) | 0.5682 |
| Matched null | skipped |

## 4.4 Matched Null Permutation

Null permutation was skipped (`--skip-null`). Re-run without this flag to include.

## 4.4b Generic Pathway Fraction

Fraction of fold-0 top-20 classified as generic (translation/ribosome/splicing): **55%**.
These pathways may reflect global proliferation rate, tumour cell content, or sequencing quality rather than histology-specific biology. They are retained in ranked lists but flagged with *(generic)* above.

## 4.5 IG vs ElasticNet Agreement

| Fold | Jaccard (top-20) | Spearman ρ (all 192) |
|---|---|---|
| 1 | 0.667 | 0.453 |
| 2 | 0.111 | 0.212 |
| 3 | 0.000 | -0.025 |
| 4 | 0.600 | 0.452 |
| 5 | 0.333 | 0.401 |
| **Mean** | **0.342** | **0.298** |
| **Range** | [0.000, 0.667] | [-0.025, 0.453] |

## 4.6 Patient Representation

One model trained on all 121 patients (3 seeds, embeddings averaged). **In-sample, visualisation only. No silhouette or separation claims.**
See `figures/explain/representation_pca_umap.png`.

## 4.7 Calibration

Pooled OOF predictions from the 25 frozen E1 folds (averaged over repeats where a patient appears more than once).

- **Brier score**: 0.1365
- **Logistic recalibration (Platt scaling)**: slope = 4.207, intercept = 3.080
  - Slope > 1 = overconfident; slope < 1 = underconfident; ideal = 1.0.
  - With N = 121 recalibration estimates are themselves uncertain.

See `figures/explain/calibration_reliability.png`.

## Known Limitations

- Only 5 fold models (1 repeat) used; attribution estimates are noisy (N_test ≈ 25).
- Jaccard stability values below 0.3 are common at this sample size.
- E1 was not distinguishable from the elastic-net baseline (Nadeau-Bengio corrected test).
- Fork 6: 726/1035 miRNA features connect to pathway P-1 (last Reactome pathway), not a dedicated unmapped token.
