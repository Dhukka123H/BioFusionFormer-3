# BioFusionFormer E1 — Explainability Summary v2 (Stage A)

> **Scope**: 5 fold models from the Stage 3 reproducibility re-run (`models/e1_folds_final_v1_repro/`). Attributions computed on TEST patients of each fold with Integrated Gradients ($n_{\text{steps}}=50$).

## 1. Fallback Audit & Resolving the ODC Artefact

- **ODC Pathway Status:** In the earlier draft, *Regulation of ornithine decarboxylase (ODC)* (`R-HSA-350562`) appeared with an attribution score of 4.98 (~2x the runner-up).
- **Audit Finding:** Pathway `R-HSA-350562` is exactly index $P-1 = 191$ (the last selected pathway). Under Fork 6 fallback wiring, all 726 miRNA features without a prior edge were wired into this single pathway token. Summing $|IG|$ over its member features accumulated the background gradients of 726 unmapped miRNAs in addition to its 40 genes. **This top finding was strictly an artefact of fallback token routing.**
- See full quantitative audit in `results/explainability/fallback_audit.md`.

## 2. Integrated Gradients: Side-by-Side Aggregations

Per Stage A2, attributions are reported in two complementary forms:
1. **Metric (i) Genuine Sum**: Sum of $|IG|$ over genuine member features, strictly excluding the 726 fallback miRNA edges.
2. **Metric (ii) Size-Normalised Mean**: Mean $|IG|$ per member feature (attribution divided by genuine feature count present in data).

| Metric | Spearman $\rho$ with Pathway Size | Size Bias Status | Generic Pathway Fraction in Top-20 |
|---|:---:|---|:---:|
| **(i) Genuine Sum** | **0.8473** ($p = 4.09e-54$) | Substantial size bias towards large pathways | 8/20 (40%) |
| **(ii) Normalised Mean** | **0.2935** ($p = 3.61e-05$) | Size bias removed | 0/20 (0%) |

### Top-10 Pathways by Metric (i) Genuine Sum

1. **mRNA Splicing** (`R-HSA-72172`) *(generic)*: 2.5073
2. **mRNA Splicing - Major Pathway** (`R-HSA-72163`) *(generic)*: 2.4816
3. **Cap-dependent Translation Initiation** (`R-HSA-72737`) *(generic)*: 2.2129
4. **Eukaryotic Translation Initiation** (`R-HSA-72613`) *(generic)*: 2.2129
5. **GTP hydrolysis and joining of the 60S ribosomal subunit** (`R-HSA-72706`) *(generic)*: 2.0848
6. **L13a-mediated translational silencing of Ceruloplasmin expression** (`R-HSA-156827`) *(generic)*: 2.0799
7. **Response of EIF2AK4 (GCN2) to amino acid deficiency** (`R-HSA-9633012`): 2.0791
8. **mRNA 3'-end processing** (`R-HSA-72187`): 1.8751
9. **SRP-dependent cotranslational protein targeting to membrane** (`R-HSA-1799339`) *(generic)*: 1.8295
10. **Formation of a pool of free 40S subunits** (`R-HSA-72689`): 1.8260

### Top-10 Pathways by Metric (ii) Size-Normalised Mean

1. **Signaling by FGFR4 in disease** (`R-HSA-5655291`): 0.01358
2. **Signaling by PDGFRA extracellular domain mutants** (`R-HSA-9673770`): 0.01351
3. **Signaling by PDGFRA transmembrane, juxtamembrane and kinase domain mutants** (`R-HSA-9673767`): 0.01351
4. **Activated NTRK2 signals through FRS2 and FRS3** (`R-HSA-9028731`): 0.01321
5. **Signaling by EGFRvIII in Cancer** (`R-HSA-5637812`): 0.01290
6. **Constitutive Signaling by EGFRvIII** (`R-HSA-5637810`): 0.01290
7. **Signaling by CTNNB1 phospho-site mutants** (`R-HSA-4839743`): 0.01285
8. **CTNNB1 T41 mutants aren't phosphorylated** (`R-HSA-5358752`): 0.01285
9. **CTNNB1 S33 mutants aren't phosphorylated** (`R-HSA-5358747`): 0.01285
10. **CTNNB1 S45 mutants aren't phosphorylated** (`R-HSA-5358751`): 0.01285

## 3. Stability Across 5 Saved Folds & Matched Null Permutation

| Method | Mean Pairwise Jaccard (Top-20) | Note |
|---|:---:|---|
| **BioFusionFormer IG (i) Genuine Sum** | **0.3987** | Fallback edges excluded |
| **BioFusionFormer IG (ii) Normalised Mean** | **0.2808** | Size-normalised |
| **ElasticNet (mRNA, train-fold only)** | **0.5682** | High-dimensional linear baseline |

> [!NOTE]
> With 3 permutations, no p-value is computed or quoted.

## 4. Recalibration (Stage A5)

Fitted logistic regression without penalty on `logit(pooled_oof_probability)`:
- **Recalibration slope**: **2.254** (1000-draw patient bootstrap 95% CI: **[1.769, 3.154]**)
- **Recalibration intercept**: **-0.076** (1000-draw patient bootstrap 95% CI: **[-0.602, 0.524]**)
- **Brier score before**: **0.1365**
- **Brier score after 5-fold CV calibrator**: **0.1213**

*Why the previous slope 4.207 / intercept 3.080 occurred:* The previous script computed `np.log(oof)` (log-probability) instead of `scipy.special.logit(oof)` (log-odds). Under true logit recalibration, the slope is ~2.25 and intercept is ~ -0.08. A slope > 1 indicates the raw model is under-confident (predicted probabilities are moderately compressed toward the center).

## 5. Representation Learning (Stage A6)

Trained on all 121 cohort patients across 3 seeds (`figures/explain/representation_pca_umap.png`).
**Caption:** *in-sample; separation reflects training labels; not evidence of generalisation.*
