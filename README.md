# BioFusionFormer

**A Pathway-Aware Hierarchical Cross-Modal Transformer for Multi-Modal Data Fusion**

Final-year project integrating mRNA-seq, miRNA-seq, and DNA methylation data from the pediatric Wilms' tumor (`TARGET-WT`) cohort, hierarchically structured by Reactome pathway topology and miRTarBase microRNA regulatory priors.

---

## Project Status & Milestones

- **Phase 0** – Project Skeleton & Virtual Environment: **COMPLETE**
- **Phase 1** – Reactome Knowledge Graph & Priors Download: **COMPLETE**
- **Phase A** – Pathway Graphs, CpG Annotations & miRNA Target Priors: **COMPLETE**
- **Fallback Phase** – UCSC Xena Hub Migration (GDC 503 Resiliency): **COMPLETE**
- **Leakage Scan & Confounder Audit** – Systematic Association Analysis: **COMPLETE**
- **Phase 4** – Multi-Modal Feature Construction & In-Fold Transformers: **COMPLETE**
- **Phase 5** – Nested 5x5 Cross-Validation & Baseline Benchmarking: **COMPLETE**
- **Phase 6** – Project Freeze & Governance Verification: **COMPLETE**

*(Note: In accordance with project instructions, zero model architecture code has been written for BioFusionFormer itself; data engineering, feature representations, and baselines are fully established.)*

---

## Project Structure

```
BioFusionFormer/
├── data/
│   ├── raw/
│   │   ├── reactome/                      # Reactome pathway flat-files & hierarchy
│   │   ├── mirna_targets/                 # miRTarBase hsa_MTI.csv & miRBase hsa.gff3
│   │   ├── cpg_annotations/               # Illumina Infinium 450k probe manifest
│   │   ├── target_wt_xena/                # UCSC Xena GDC TARGET-WT molecular & clinical matrices
│   │   └── CHECKSUMS.txt                  # MD5/SHA256 checksums for raw files
│   ├── interim/
│   │   ├── cohort_patients.csv            # Aligned 121-patient 3-omics core manifest
│   │   ├── cpg2gene.csv                   # Promoter CpG-to-gene mapping table
│   │   ├── mirna2gene_strong.parquet      # Strong experimental miRNA targets (348 KB)
│   │   ├── mirna2gene_all.parquet         # Comprehensive miRNA targets (28.4 MB)
│   │   ├── priors/                        # miRNA-pathway bipartite adjacency masks (.npz)
│   │   └── reactome/                      # Parsed pathway graph nodes and edges
│   ├── processed/
│   │   ├── mrna_gene.parquet              # (121, 11171) Reactome genes, log2(count+1)
│   │   ├── mirna.parquet                  # (121, 1797) mature miRNAs, log2(RPM+1)
│   │   ├── meth_gene.parquet              # (121, 9778) promoter mean beta values
│   │   ├── meth_cpg_top.parquet           # (121, 20000) top variable promoter CpGs
│   │   ├── pathway_features_fold_agnostic.parquet # (121, 3368) mRNA + Meth pathway scores
│   │   └── pathway_hierarchy_mapping.json # Reactome hierarchy depths (1 to 13) and DAG edges
│   └── splits/
│       ├── splits_5x5.json                # Canonical joint repeated stratified 5x5 CV splits
│       ├── splits_histology_5x5.json      # Primary task stratified splits
│       └── splits_vital_status_5x5.json   # Exploratory task stratified splits
├── src/
│   ├── preprocessing/
│   │   └── transformers.py                # Scikit-learn leak-free transformers
│   └── pathways/
│       └── scorer.py                      # Pathway activity z-score encoder
├── scripts/
│   ├── generate_splits.py                 # 5x5 CV split generator (seeds 42-46)
│   ├── phase4_feature_construction.py     # Label-independent feature matrix construction
│   ├── phase5_baselines.py                # Nested CV baseline evaluation engine
│   └── wait_for_gdc.py                    # Resilient GDC API poller
├── results/
│   ├── leakage_scan.md                    # Systematic clinical feature association scan
│   ├── confound_report.md                 # Batch, plate, protocol, and confounder analysis
│   └── baselines/
│       ├── summary.md                     # Full nested CV baseline benchmark report
│       ├── per_modality_performance.png   # Comparative AUROC bar chart across modalities
│       └── permutation_test.png           # 100-permutation null distribution vs real AUROC
├── tests/
│   └── test_preprocessing_leakage.py      # Unit test verifying zero test-fold leakage
└── logs/                                  # Detailed execution and audit logs
```

---

## Cohort & Clinical Tasks

- **Cohort Size:** $N = 121$ primary solid tumor patients with aligned mRNA-seq, miRNA-seq, Illumina 450k DNA methylation, and clinical data.
- **Primary Clinical Task:** Binary Classification of Histological Subtype (`wilms_tumor_histologic_subtype.diagnoses`)
  - **Favorable Histology (FH):** $N = 82$ (67.8%)
  - **Unfavorable Histology / Anaplasia (UH):** $N = 39$ (32.2%)
- **Exploratory Secondary Task:** Binary Classification of Vital Status (`vital_status.demographic`)
  - **Alive (Not Known Dead):** $N = 71$ (58.7%)
  - **Dead:** $N = 50$ (41.3%)
  - *Caveat:* Alive patients in TARGET-WT lack recorded post-diagnosis follow-up times (`days_to_last_follow_up = NaN`). Therefore, Cox proportional hazards regression is invalid.

---

## Baseline Benchmark Results Summary (Nested 5x5 CV)

Evaluated across 25 outer test folds (5 repeats $\times$ 5 folds) with inner 3-fold CV hyperparameter tuning:

| Model / Modality | Primary Task (Histology) AUROC [95% CI] | Exploratory Task (Vital Status) AUROC [95% CI] |
|---|---|---|
| **Clinical-only (Age + Gender)** | **0.630** [0.583, 0.676] | **0.581** [0.547, 0.615] |
| **Clinical-only (+ Stage)** | **0.621** [0.573, 0.668] | **0.706** [0.675, 0.737] |
| **mRNA (ElasticNet)** | **0.917** [0.895, 0.939] | **0.502** [0.463, 0.542] |
| **mRNA (Random Forest)** | **0.911** [0.890, 0.932] | **0.517** [0.472, 0.563] |
| **miRNA (ElasticNet)** | **0.848** [0.817, 0.879] | **0.480** [0.438, 0.522] |
| **miRNA (Random Forest)** | **0.847** [0.817, 0.878] | **0.514** [0.475, 0.552] |
| **Methylation (ElasticNet)** | **0.460** [0.424, 0.496] | **0.550** [0.507, 0.593] |
| **Early Fusion (ElasticNet)** | **0.930** [0.912, 0.948] | **0.521** [0.483, 0.558] |
| **Early Fusion (Random Forest)** | **0.910** [0.892, 0.927] | **0.483** [0.433, 0.533] |
| **Late Fusion (Ensemble)** | **0.851** [0.824, 0.878] | **0.514** [0.481, 0.546] |
| **Pathway MLP** | **0.704** [0.649, 0.759] | **0.490** [0.434, 0.546] |

- **Top Model:** `Early Fusion (ElasticNet)` (AUROC = **0.930**, AUPRC = **0.895**, Macro F1 = **0.833**, Balanced Accuracy = **0.826**).
- **Permutation Test Control:** 100 label permutations yielded a null distribution mean AUROC of **0.507** (range: 0.370–0.686) vs real AUROC of **0.9295**, establishing an empirical p-value of **p = 0.0099** ($p < 0.01$).
- **Key Takeaway:** Omics significantly outperform clinical demographics for histology ($p < 10^{-11}$). Standalone mRNA carries the primary signal; Early Fusion delivers modest further gain; stage dominates vital status.

---

## Reproducibility & Testing

To reproduce all preprocessing tests, CV splits, and baseline runs:

```bash
# 1. Run unit tests for leak-free transformers
python -m unittest tests/test_preprocessing_leakage.py

# 2. Re-generate repeated stratified splits
python scripts/generate_splits.py

# 3. Run full nested cross-validation and generate baseline figures
python scripts/phase5_baselines.py
```
