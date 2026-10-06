# BioFusionFormer

**A Pathway-Aware Hierarchical Cross-Modal Transformer for Multi-Modal Data Fusion**

Final-year biomedical AI project integrating mRNA-seq, miRNA-seq, and DNA methylation data from the pediatric Wilms' tumor (`TARGET-WT`) cohort, hierarchically structured by Reactome pathway topology and miRTarBase microRNA regulatory priors.

> 📖 **Comprehensive Project Guide & Academic Defense Manual:**  
> For a complete, deep-dive explanation of the biology, mathematics, architecture, zero-leakage protocols, and viva/defense Q&A, please read [**`PROJECT_GUIDE.md`**](file:///c:/Users/dhukk/Downloads/BioFusionFormer/PROJECT_GUIDE.md).

---

## Project Status & Milestones

- [x] **Phase 0** – Project Skeleton & Virtual Environment: **COMPLETE**
- [x] **Phase 1** – Reactome Knowledge Graph & Priors Download: **COMPLETE**
- [x] **Phase A** – Pathway Graphs, CpG Annotations & miRNA Target Priors: **COMPLETE**
- [x] **Fallback Phase** – UCSC Xena Hub Migration (GDC 503 Resiliency): **COMPLETE**
- [x] **Leakage Scan & Confounder Audit** – Systematic Association Analysis: **COMPLETE**
- [x] **Phase 4** – Multi-Modal Feature Construction & In-Fold Transformers: **COMPLETE**
- [x] **Phase 5** – Nested 5x5 Cross-Validation & Baseline Benchmarking: **COMPLETE**
- [x] **Stage 1 & 2** – BioFusionFormer Architecture, Pathway Tokens & Biased Attention: **COMPLETE**
- [x] **Stage 3** – Final Benchmark Suite (E1–E8, E10) across 25 Repeated Folds: **COMPLETE / FROZEN (v1-frozen)**
- [x] **Stage 4** – Explainability (Attention Rollout, Null Permutation, IG vs SHAP, UMAP/PCA): **COMPLETE**

---

## Project Structure

```
BioFusionFormer/
├── PROJECT_GUIDE.md                       # Comprehensive deep-dive explainer for guide and defense
├── README.md                              # Repository overview and setup guide
├── configs/
│   └── default.yaml                       # Frozen hyperparameters & training configurations
├── data/
│   ├── DATA_CARD.json                     # Complete data lineage, dimensions & governance
│   ├── interim/
│   │   ├── cohort_patients.csv            # Aligned 121-patient 3-omics core manifest
│   │   └── priors/                        # Sparse Reactome & miRTarBase masks (.npz, .json)
│   ├── processed/
│   │   ├── mrna_gene.parquet              # (121, 11171) Reactome genes, log2(count+1)
│   │   ├── mirna_unique.parquet           # (121, 1035) deduplicated precursor miRNAs
│   │   └── meth_gene.parquet              # (121, 9778) promoter mean beta values
│   └── splits/
│       └── splits_5x5.json                # Canonical joint repeated stratified 5x5 CV splits
├── src/
│   ├── biofusionformer/
│   │   ├── full_model.py                  # Complete BioFusionFormer PyTorch architecture
│   │   ├── layers.py                      # Biologically-biased transformer encoder layers
│   │   ├── pathway_tokenizer.py           # Sparse pathway token projection module
│   │   └── gating.py                      # Adaptive multi-modal gating network
│   ├── heads/
│   │   └── classification.py              # Classification head with label smoothing
│   ├── explainability/
│   │   └── explain.py                     # Attention rollout, Jaccard stability, IG attribution
│   └── pathways/
│       └── selection.py                   # Unsupervised variance-based pathway selection
├── scripts/
│   ├── run_v1_benchmark_and_stages3_4.py  # Full benchmark suite (E1-E8, E10) & Stage 4
│   ├── check_gene_coverage_a1.py          # Gene pathway coverage verification script
│   └── phase5_baselines.py                # Nested CV baseline benchmark engine
├── results/
│   ├── model/
│   │   ├── summary.md                     # Final benchmark comparison and statistical tests
│   │   └── forking_paths.md               # Scientific log of pre-specified decision rules
│   ├── explainability/
│   │   └── top_pathways.md                # Recurrent Reactome pathways & IG/SHAP overlap
│   └── baselines/
│       └── summary.md                     # Baseline ElasticNet and Random Forest results
├── figures/                               # UMAP, PCA, calibration curves, rollout charts
└── tests/
    ├── test_biofusionformer.py            # Complete unit tests for model components
    └── test_preprocessing_leakage.py      # Unit test verifying zero test-fold leakage
```

---

## Cohort & Clinical Tasks

- **Cohort:** NCI TARGET-WT (Pediatric Wilms Tumor), $N = 121$ primary solid tumor patients with aligned 3-omics measurements.
- **Primary Clinical Task:** Binary Classification of Histological Subtype (`wilms_tumor_histologic_subtype.diagnoses`)
  - **Favorable Histology (FH):** $N = 82$ (67.8%)
  - **Unfavorable Histology / Diffuse Anaplasia (UH):** $N = 39$ (32.2%)
- **Exploratory Secondary Task:** Overall Vital Status Prognostication (`vital_status.demographic`)
  - **Alive (Not Known Dead):** $N = 71$ (58.7%)
  - **Dead:** $N = 50$ (41.3%)

---

## Team Setup & Low-Bandwidth Cloning Guide

> **Note for Team Members with Limited Internet / Mobile Data:**  
> The massive raw datasets (>1.6 GB of raw Xena TSVs, Illumina manifests, and miRTarBase databases) are excluded from the repository.  
> **All essential processed feature matrices and sparse prior masks are already tracked in `data/processed/` and `data/interim/priors/`!**  
> The entire repository download size is **under 35 MB**, so you can clone it in under 1 minute even on a mobile hotspot without using your daily data quota!

### Quick Start:

```bash
# 1. Clone the repository
git clone https://github.com/Dhukka123H/BioFusionFormer-3.git
cd BioFusionFormer-3

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run unit tests to verify setup
python -m unittest tests/test_biofusionformer.py
python -m unittest tests/test_preprocessing_leakage.py

# 5. Run the baseline models
python scripts/phase5_baselines.py

# 6. Run the final BioFusionFormer benchmark and explainability suite
python scripts/run_v1_benchmark_and_stages3_4_resumable.py
```

---

## Baseline Benchmark Results (Nested 5x5 CV)

Evaluated across 25 outer test folds (5 repeats $\times$ 5-fold stratified CV) with inner stratified early stopping:

| Model / Modality | Histology AUROC [95% CI] | Vital Status AUROC [95% CI] |
|---|---|---|
| **Clinical-only (Age + Gender)** | **0.630** [0.583, 0.676] | **0.581** [0.547, 0.615] |
| **mRNA (ElasticNet)** | **0.917** [0.895, 0.939] | **0.502** [0.463, 0.542] |
| **miRNA (ElasticNet)** | **0.848** [0.817, 0.879] | **0.480** [0.438, 0.522] |
| **Methylation (ElasticNet)** | **0.460** [0.424, 0.496] | **0.550** [0.507, 0.593] |
| **Early Fusion (ElasticNet)** | **0.930** [0.912, 0.948] | **0.521** [0.483, 0.558] |
| **BioFusionFormer (Full E1)** | **0.90–0.92** (statistically equivalent to ElasticNet, $p > 0.05$) | **0.50–0.52** |

- **Statistical Significance:** Evaluated using the **Nadeau-Bengio corrected resampled paired t-test** to account for training set overlap across repeated folds.
- **Key Biological Discovery:** DNA methylation alone has *no detectable signal for histology* (AUROC ~0.46), and BioFusionFormer's adaptive gating network automatically learns to downweight methylation (<15%) while prioritizing mRNA (~70%).

---

## References & Documentation
- Comprehensive Defense Guide: [**`PROJECT_GUIDE.md`**](file:///c:/Users/dhukk/Downloads/BioFusionFormer/PROJECT_GUIDE.md)
- Data Governance & Lineage: [**`data/DATA_CARD.json`**](file:///c:/Users/dhukk/Downloads/BioFusionFormer/data/DATA_CARD.json)
- Final Benchmark Report: [**`results/model/summary.md`**](file:///c:/Users/dhukk/Downloads/BioFusionFormer/results/model/summary.md)
- Scientific Forking Paths Log: [**`results/model/forking_paths.md`**](file:///c:/Users/dhukk/Downloads/BioFusionFormer/results/model/forking_paths.md)
- Top Pathway Biological Report: [**`results/explainability/top_pathways.md`**](file:///c:/Users/dhukk/Downloads/BioFusionFormer/results/explainability/top_pathways.md)
