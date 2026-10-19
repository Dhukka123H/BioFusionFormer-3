# BioFusionFormer E1 — Explainability Summary v2 + Addendum (Stage A)

> **Source:** Generated strictly from `results/explainability/explainability_results.json`.

## 1. Technical Fallback Audit (ODC Artefact)

In the frozen E1 architecture, 726 miRNA features without a prior edge were wired into the last selected pathway token ($P-1$), which happened to be **Regulation of ornithine decarboxylase (ODC)** (`R-HSA-350562`). This artificial concentration of 726 features caused ODC to falsely appear as the #1 attributed pathway in earlier drafts. **ODC is confirmed to be an artefact of the fallback wiring.** When fallback edges are excluded, attributions are dominated by pathway size (Spearman 0.91) and by translation/ribosome/splicing genes; after size normalisation stability is not above the matched null.

## 2. Integrated Gradients: Genuine Sum vs. Size-Normalised Mean

Attributions are reported side-by-side to expose and quantify pathway size bias:

| Rank | (i) Genuine Member Sum (Fallback Excluded) | Score | Generic? | (ii) Size-Normalised Mean (|IG| / size) | Score | Generic? |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| 1 | Cap-dependent Translation Initiation (`R-HSA-72737`) | 3.1622 | Yes | Peptide chain elongation (`R-HSA-156902`) | 0.0141 | Yes |
| 2 | Eukaryotic Translation Initiation (`R-HSA-72613`) | 3.1622 | Yes | PELO:HBS1L and ABCE1 dissociate a ribosome (`R-HSA-9954714`) | 0.0139 | Yes |
| 3 | GTP hydrolysis and joining of the 60S ribo (`R-HSA-72706`) | 3.0344 | Yes | Selenocysteine synthesis (`R-HSA-2408557`) | 0.0138 | No |
| 4 | L13a-mediated translational silencing of C (`R-HSA-156827`) | 3.0256 | Yes | Nonsense Mediated Decay (NMD) independent  (`R-HSA-975956`) | 0.0137 | No |
| 5 | mRNA Splicing (`R-HSA-72172`) | 3.0124 | Yes | Eukaryotic Translation Termination (`R-HSA-72764`) | 0.0136 | Yes |
| 6 | mRNA Splicing - Major Pathway (`R-HSA-72163`) | 2.9704 | Yes | Formation of a pool of free 40S subunits (`R-HSA-72689`) | 0.0136 | No |
| 7 | Response of EIF2AK4 (GCN2) to amino acid d (`R-HSA-9633012`) | 2.8619 | No | Eukaryotic Translation Elongation (`R-HSA-156842`) | 0.0136 | Yes |
| 8 | Selenoamino acid metabolism (`R-HSA-2408522`) | 2.7191 | No | Response of EIF2AK4 (GCN2) to amino acid d (`R-HSA-9633012`) | 0.0134 | No |
| 9 | Formation of a pool of free 40S subunits (`R-HSA-72689`) | 2.7021 | No | L13a-mediated translational silencing of C (`R-HSA-156827`) | 0.0134 | Yes |
| 10 | SRP-dependent cotranslational protein targ (`R-HSA-1799339`) | 2.6733 | Yes | GTP hydrolysis and joining of the 60S ribo (`R-HSA-72706`) | 0.0133 | Yes |
| 11 | Nonsense Mediated Decay (NMD) independent  (`R-HSA-975956`) | 2.5944 | No | ZNF598 and the Ribosome-associated Quality (`R-HSA-9954716`) | 0.0132 | Yes |
| 12 | Selenocysteine synthesis (`R-HSA-2408557`) | 2.5368 | No | Eukaryotic Translation Initiation (`R-HSA-72613`) | 0.0129 | Yes |
| 13 | ZNF598 and the Ribosome-associated Quality (`R-HSA-9954716`) | 2.5244 | Yes | Cap-dependent Translation Initiation (`R-HSA-72737`) | 0.0129 | Yes |
| 14 | PELO:HBS1L and ABCE1 dissociate a ribosome (`R-HSA-9954714`) | 2.5204 | Yes | Formation of the ternary complex, and subs (`R-HSA-72695`) | 0.0123 | No |
| 15 | Eukaryotic Translation Elongation (`R-HSA-156842`) | 2.5080 | Yes | SRP-dependent cotranslational protein targ (`R-HSA-1799339`) | 0.0122 | Yes |
| 16 | Eukaryotic Translation Termination (`R-HSA-72764`) | 2.5029 | Yes | Signaling by CTNNB1 phospho-site mutants (`R-HSA-4839743`) | 0.0120 | No |
| 17 | Peptide chain elongation (`R-HSA-156902`) | 2.4887 | Yes | CTNNB1 S45 mutants aren't phosphorylated (`R-HSA-5358751`) | 0.0120 | No |
| 18 | mRNA 3'-end processing (`R-HSA-72187`) | 2.3765 | No | CTNNB1 T41 mutants aren't phosphorylated (`R-HSA-5358752`) | 0.0120 | No |
| 19 | mRNA Polyadenylation (`R-HSA-9770562`) | 2.0612 | No | Signaling by GSK3beta mutants (`R-HSA-5339716`) | 0.0120 | No |
| 20 | Mitotic Spindle Checkpoint (`R-HSA-69618`) | 1.6123 | No | CTNNB1 S37 mutants aren't phosphorylated (`R-HSA-5358749`) | 0.0120 | No |

### Size Bias Quantification
- **Genuine Member Sum vs. Pathway Size:** Spearman $\rho = 0.9099$ ($p = 1.60e-74$). Larger pathways accumulate higher attribution sums.
- **Size-Normalised Mean vs. Pathway Size:** Spearman $\rho = 0.4413$ ($p = 1.49e-10$). Normalising by feature count controls for pathway size.
- **Generic Pathway Fraction:** Genuine sum top-20 is 60% generic; size-normalised top-20 is 50% generic.

## 3. Pathway Redundancy & Clustering Analysis

Pairwise gene-set Jaccard was computed across pathways. Grouping pathways with gene-set Jaccard $\ge 0.5$ reveals high biological redundancy:
- **Genuine Member Sum Top-20:** Collapses from 20 nominal pathways into **4 independent clusters**:
  - **Cluster 1 (15 pathways):** Cap-dependent Translation Initiation, Eukaryotic Translation Initiation...
  - **Cluster 2 (2 pathways):** mRNA Splicing, mRNA Splicing - Major Pathway
  - **Cluster 3 (2 pathways):** mRNA 3'-end processing, mRNA Polyadenylation
  - **Cluster 4 (1 pathways):** Mitotic Spindle Checkpoint
- **Size-Normalised Mean Top-20:** Collapses from 20 nominal pathways into **3 independent clusters**:
  - **Cluster 1 (14 pathways):** Peptide chain elongation, PELO:HBS1L and ABCE1 dissociate a ribosome on a non-stop mRNA...
  - **Cluster 3 (5 pathways):** Signaling by CTNNB1 phospho-site mutants, CTNNB1 S45 mutants aren't phosphorylated...
  - **Cluster 2 (1 pathways):** Formation of the ternary complex, and subsequently, the 43S complex

### Cluster-Level Top-5 Stability Across 5 Folds
| Attribution Metric | Mean Pairwise Top-5 Cluster Jaccard | Matched Null Cluster Jaccard |
| :--- | :---: | :---: |
| (i) Genuine Member Sum (Cluster-level) | 0.4619 | — |
| (ii) Size-Normalised Mean (Cluster-level) | 0.4369 | — |

## 4. Pathway Attribution Stability & Matched Null (Like-for-Like)

| Attribution Metric | Mean Pairwise Top-20 Jaccard (5 folds) | Per-Permutation Null Jaccard Values | Matched Null Mean |
| :--- | :---: | :---: | :---: |
| (i) Genuine Member Sum | **0.6170** | `[0.4906, 0.2880, 0.3975]` | 0.3920 |
| (ii) Size-Normalised Mean | **0.2765** | `[0.3719, 0.2306, 0.2586]` | 0.2870 |
| ElasticNet mRNA (Sum) | 0.5682 | — | — |
| ElasticNet mRNA (Size-Normalised) | 0.3683 | — | — |

### Agreement Between Integrated Gradients and ElasticNet (Like-for-Like)
- **Raw Sum Metric:** Mean Top-20 Jaccard = `0.4899`, Mean Spearman $\rho = 0.4732$.
- **Size-Normalised Metric:** Mean Top-20 Jaccard = `0.0105`, Mean Spearman $\rho = 0.0249$.

> *Note on matched null:* 3 label permutations were evaluated with the exact same E1 training procedure (seed 42 only). With 3 permutations, no p-value is given.

## 5. Descriptive Gene- and Feature-Level View

> **Note on Null Baseline:** Null attributions were not stored at feature level; no matched null baseline exists for feature-level rankings. This view is descriptive only.

- **mRNA Gene Top-20 Stability (Mean Pairwise Jaccard, 5 folds):** `0.0947`
- **miRNA Feature Top-20 Stability (Mean Pairwise Jaccard, 5 folds):** `0.0885`
- **DNA Methylation Feature Top-20 Stability (Mean Pairwise Jaccard, 5 folds):** `0.0508`

| Rank | Top mRNA Genes | Score | Top miRNAs | Score | Top Methylation CpGs | Score |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| 1 | ENSG00000161970 | 0.0759 | hsa-miR-34a-5p+coll_2 | 0.0414 | ENSG00000229117 | 0.0186 |
| 2 | ENSG00000182774 | 0.0612 | hsa-miR-100-5p+coll_2 | 0.0321 | ENSG00000147604 | 0.0162 |
| 3 | ENSG00000109475 | 0.0528 | hsa-miR-145-5p+coll_2 | 0.0307 | ENSG00000145425 | 0.0149 |
| 4 | ENSG00000100316 | 0.0461 | hsa-miR-155-5p+coll_2 | 0.0299 | ENSG00000069248 | 0.0147 |
| 5 | ENSG00000177600 | 0.0422 | hsa-miR-302b-5p+coll_2 | 0.0297 | ENSG00000085415 | 0.0144 |
| 6 | ENSG00000265681 | 0.0403 | hsa-miR-302d-5p+coll_2 | 0.0294 | ENSG00000145592 | 0.0141 |
| 7 | ENSG00000185088 | 0.0361 | hsa-miR-214-5p+coll_2 | 0.0279 | ENSG00000151247 | 0.0141 |
| 8 | ENSG00000108559 | 0.0352 | hsa-miR-132-5p+coll_2 | 0.0275 | ENSG00000149273 | 0.0128 |
| 9 | ENSG00000167526 | 0.0341 | hsa-miR-506-5p+coll_2 | 0.0261 | ENSG00000089009 | 0.0127 |
| 10 | ENSG00000280969 | 0.0339 | hsa-miR-223-5p+coll_2 | 0.0250 | ENSG00000134419 | 0.0116 |
| 11 | ENSG00000186468 | 0.0331 | hsa-miR-106a-5p+coll_2 | 0.0244 | ENSG00000137818 | 0.0115 |
| 12 | ENSG00000110700 | 0.0320 | hsa-miR-139-5p+coll_2 | 0.0228 | ENSG00000161970 | 0.0110 |
| 13 | ENSG00000100353 | 0.0312 | hsa-miR-3925-5p+coll_2 | 0.0223 | ENSG00000120253 | 0.0109 |
| 14 | ENSG00000125691 | 0.0300 | hsa-let-7b-5p+coll_2 | 0.0222 | ENSG00000221983 | 0.0106 |
| 15 | ENSG00000163923 | 0.0294 | hsa-miR-203a-5p+coll_2 | 0.0220 | ENSG00000182774 | 0.0106 |
| 16 | ENSG00000105193 | 0.0292 | hsa-miR-3174-5p+coll_2 | 0.0214 | ENSG00000162244 | 0.0102 |
| 17 | ENSG00000129824 | 0.0292 | hsa-miR-29b-3p | 0.0211 | ENSG00000112306 | 0.0097 |
| 18 | ENSG00000171858 | 0.0287 | hsa-miR-6084-3p | 0.0208 | ENSG00000105568 | 0.0095 |
| 19 | ENSG00000166441 | 0.0280 | hsa-miR-7-5p | 0.0207 | ENSG00000110713 | 0.0093 |
| 20 | ENSG00000142676 | 0.0272 | hsa-miR-503-5p+coll_2 | 0.0205 | ENSG00000177954 | 0.0092 |

## 6. Deployment Recalibration Check

In deployment, the model evaluates single-repeat out-of-fold predictions rather than a 5-repeat ensemble. To audit calibration stability, Platt scaling calibrators were fitted on each single repeat:

| Repeat | Slope | Intercept | Brier (Before / After) | Within Repeat-Averaged 95% CI `[1.769, 3.154]`? |
| :---: | :---: | :---: | :---: | :---: |
| Repeat 1 | 1.974 | -0.223 | 0.1355 / 0.1194 | ✅ Yes |
| Repeat 2 | 1.949 | -0.221 | 0.1515 / 0.1337 | ✅ Yes |
| Repeat 3 | 1.686 | -0.272 | 0.1343 / 0.1199 | ⚠️ No (outside) |
| Repeat 4 | 1.739 | -0.177 | 0.1317 / 0.1207 | ⚠️ No (outside) |
| Repeat 5 | 1.434 | -0.386 | 0.1650 / 0.1539 | ⚠️ No (outside) |

- **Pooled Single-Repeat Slope:** **1.731** (95% patient bootstrap CI: `[1.352, 2.286]`)
- **Pooled Single-Repeat Intercept:** **-0.271** (95% patient bootstrap CI: `[-0.719, 0.202]`)

> **Clinical Deployment Rule & GUI Verification:** Because single-repeat slopes (ranging from 1.434 to 1.974, with repeats 2, 3, and 4 falling below 1.769) are not strictly consistent within the repeat-averaged CI [1.769, 3.154], deployment interfaces (GUI and CLI) must display raw predicted scores with a **"Not Calibrated"** label rather than claiming a calibrated probability.

## 7. Patient Representation

PCA representations (`figures/explain/representation_pca_umap.png`) are in-sample and reflect training labels; they do not provide evidence of generalisation.
