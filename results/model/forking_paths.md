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


---

## Cleanup audit (added when the repository was reduced to E1 only)

These items were found by re-reading the code and logs. None of them changes the stored E1 fold results; they change
how those results must be described.

### Fork 5: E1 was run twice
The first full E1 run (old, unseeded model initialisation, finished 2026-10-05) gave mean AUROC 0.8460. After seeding was
added the run was repeated (sessions 20261006_220316 and _235247, 25 folds) and gave 0.8716, which is the stored result.
Both runs used the same data and configuration. The difference (0.026) is the run-to-run variability of this model and is
about as large as the differences between the ablations that were later interpreted. Report 0.8716 together with this fact.

### Fork 6: miRNA "fallback token" is not a separate token
`mask_mirna_strong[zero_strong, P-1] = True` connects the 726 miRNA features without an edge in the 192 selected pathways
to **pathway column P-1, which is a real Reactome pathway** (the 192nd by variance), not a dedicated "unmapped_mirna" token as
Fork 2 stated. Only 309 of 1,035 miRNA features have a prior edge inside the selected pathways. The frozen E1 contains this
behaviour. A dedicated extra token is an architecture change and would be a new, exploratory run.

### Fork 7: parameter count
`count_parameters()` counts the dense masked weight matrices (192 x 11,171 + 192 x 1,035 + 192 x 9,778 = 4,220,928 entries);
only 18,865 of them are allowed to be non-zero. The "62,450 trainable parameters" figure refers to free parameters.
Use `BioFusionFormerModel.count_effective_parameters()` and quote both numbers.

### Fork 8: modality dropout only affects the gate
During training modality dropout masks gate logits only; the transformer and the [CLS] token always see every modality, and
`forward` requires all three inputs. The model is therefore **not** robust to a missing modality and cannot be used for
inference with a missing omics layer (tested in `tests/test_e1_model.py`).

### Fork 9: reporting corrections
* The 95% CI printed by the old runner was mean +/- 1.96 SD/sqrt(25) (naive). The Nadeau-Bengio interval for E1 is
  [0.802, 0.941] versus the naive [0.847, 0.896].
* Old ablations E3 and E10 had identical configurations and seeds and therefore identical outputs; E10 was not an independent control.
* Old Stage 4 analyses were not comparable and were removed: IG used one fold and only the mRNA input and was compared with an
  elastic net fitted on all patients; the label-permutation null was trained for 35 epochs (patience 10) versus 100 (20) for E1;
  the representation plots used out-of-fold embeddings from different fold models, which do not share an embedding space;
  the early-fusion calibration baseline was refit with a fixed C instead of the nested-CV model. Attention rollout was uniform
  (max/min 1.06x). All seven figures were deleted and are to be regenerated by corrected scripts.
* The old unit tests covered the discarded pilot model, not the classes used for E1. New tests: `tests/test_e1_model.py`.

### Fork 10: data source
Omics matrices come from the UCSC Xena GDC hub (GDC-derived). Clinical labels are now taken from the GDC (see
`scripts/build_labels.py`, which verifies the fold-wise class counts against `splits_5x5.json`).
