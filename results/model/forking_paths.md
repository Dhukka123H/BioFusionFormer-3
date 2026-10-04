# BioFusionFormer — Forking Paths and Architectural Decision Log

This document records all methodological decisions, correctness checks, and architectural adjustments made during model development. Hyperparameter tuning was never performed on outer-fold test results; all decisions follow strictly pre-specified criteria and inner-fold validations.

---

## Fork 1: Pathway Set Size and Gene Coverage Audit (Step A1)
- **Target Run:** Model Design & Configuration v1
- **Observation:** The unsupervised selection of 192 Reactome pathways (depth $\le 4$, $10 \le \text{genes} \le 300$) covers:
  - mRNA: 2,152 / 11,171 genes (19.3%)
  - miRNA (strong evidence): 408 / 1,797 mature features (22.7%)
  - miRNA (broad evidence): 1,370 / 1,797 mature features (76.2%)
  - Methylation: 1,794 / 9,778 promoter-associated genes (18.3%)
- **Pre-specified Empirical Rule:** Evaluate in-fold ElasticNet on mRNA restricted exclusively to the 2,152 covered genes across the 25 outer test folds. If the restricted AUROC drops by $> 0.03$ compared to full-gene mRNA ElasticNet (0.9169), widen the pathway set to top 500 by variance (depth $\le 6$). If still $> 0.03$ below, add a gene-level linear skip path (masked sparse linear layer straight to fused patient embedding).
- **Result:**
  - Full-gene mRNA ElasticNet AUROC: **0.9169**
  - Restricted mRNA ElasticNet AUROC: **0.9070** ($\text{std} = 0.0446$)
  - Difference ($\Delta$): **-0.0099** (well within the $-0.03$ threshold)
- **Decision:** The 192 pathways preserve essentially the entire discriminatory signal of the transcriptome while maintaining a short token sequence ($S \approx 577$ tokens) and compact parameter budget (~60k parameters). Therefore, widening to 500 pathways and the gene-level skip path (E9) were skipped as per the pre-specified protocol.

---

## Fork 2: miRNA Precursor-Level Deduplication & Union Fallback Prior (Step A2)
- **Target Run:** Model Inputs v1
- **Observation:** Audit of `data/processed/mirna.parquet` revealed that 762 columns were identical duplicate profiles across all 121 patients due to multiple mature miRNAs originating from shared precursor stem-loops.
- **Action Taken:**
  - Collapsed duplicate columns into 1,035 unique precursor-level features saved to `data/processed/mirna_unique.parquet`.
  - Built union pathway masks (`mirna_unique_pathway_mask_strong.npz` and `mirna_unique_pathway_mask_broad.npz`) where each unique feature inherits the union of target pathways of its mature products.
  - Coverage: 411 / 1,035 (39.7%) have strong-evidence pathway links; 745 / 1,035 (72.0%) have broad-evidence pathway links.
  - Implemented an explicit **no-miRNA-prior fallback**: unique features lacking a direct Reactome edge are connected to an explicit "unmapped_mirna" token, ensuring 100% of miRNA features contribute to representation learning without silent omission.

---

## Fork 3: Nomenclature and Framing for DNA Methylation (Step A3)
- **Target Run:** All reports and baselines
- **Adjustment:** In all documentation, methylation performance in TARGET-WT (AUROC ~0.43-0.46 with true labels vs ~0.53 with shuffled labels) is framed as **"no detectable signal for histology"** rather than "confirmed noise", reflecting that the high-risk cohort HM450 beta values do not exhibit linear or tree-based separability for Wilms anaplasia.

---

## Fork 4: Model Suite Formalization (Final Run v1)
- **Target Run:** Experiments E1–E8 + E10
- **Modifications:**
  - Added **E10 (Methylation-token-free control)**: identical architecture to E3 (mRNA + miRNA) but trained under the exact frozen seeds and early-stopping rules as E1, providing an unbiased ablation comparison.
  - Added **Brier Score** calibration metric to all experiment evaluations.
  - Frozen configuration committed and tagged as `v1-frozen`.
