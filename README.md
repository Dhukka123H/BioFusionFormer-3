# BioFusionFormer

**A pathway-aware hierarchical cross-modal Transformer for multi-omics fusion** - final-year project.
Cohort: TARGET-WT (pediatric Wilms tumour), N = 121 primary-tumour patients with mRNA, miRNA, methylation and clinical data.
Task: classify histology, favorable (82) vs unfavorable (39). Pathway prior: Reactome; miRNA prior: miRTarBase.

> Status: **CLI-first**. One headline model (**E1**). A GUI will be added later on top of `predict()`.
> Everything else that was tried is exploratory and is archived in git tag `v1-archive-all-experiments`.

## What is the headline result (read this before quoting any number)

* E1 AUROC over the 25 outer folds: see `results/model/E1_summary.md` (regenerate with `python scripts/run_e1.py --report`).
  Use the **Nadeau-Bengio corrected interval**; the naive interval in older reports is far too narrow.
* E1 is **not distinguishable** from the shallow elastic-net baselines (mRNA elastic net 0.917, early fusion 0.930) with N = 121.
* The pathway prior was **not** shown to help: the real-vs-random prior comparison (old E5) gave p = 0.86.
* Attention rollout is almost uniform over pathways (max/min about 1.06x) and is **not** an explanation.
* Omics do not predict vital status (AUROC about 0.50). Prognosis is therefore not claimed.

## Layout

```
configs/      default.yaml (frozen hyper-parameters, E1 only), clinical_features.yaml
data/         DATA_CARD.json, interim/ (priors, cohort), processed/ (matrices, labels.csv), splits/
docs/         PROJECT_GUIDE.md, AGENT_PROMPT.md
experiments/  final_v1/E1/  (25 per-fold checkpoints + metrics.json)
legacy_xena/  Xena-specific scripts/report (data lineage)
results/      baselines/, sanity_checks.md, leakage_scan.md, confound_report.md, model/ (E1_summary.md, forking_paths.md)
scripts/      run_e1.py (the only model runner), build_labels.py, download_clinical.py, phase_* data steps, phase5_baselines.py
src/          encoders/, transformer/, fusion/, heads/, pathways/, preprocessing/, evaluation/ (cv_stats), explainability/, biofusionformer/full_model.py
tests/        test_e1_model.py, test_cv_stats.py, test_preprocessing_leakage.py
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m pytest tests -q                               # must pass before anything else
python scripts/download_clinical.py                     # GDC open-access clinical (needs the GDC to be up)
python scripts/build_labels.py                          # labels + fold-by-fold count check
python scripts/run_e1.py --status                       # expects 25/25 folds DONE
python scripts/run_e1.py --report                       # regenerate E1_summary.md (no training)
python scripts/run_e1.py --smoke --run                  # 2 folds, 1 seed, 3 epochs: pipeline check
streamlit run app/app.py                                # launch interactive clinical research interface
```

## Results

Evaluated across all 25 outer folds (5 repeats x 5-fold CV, N=121):

| Metric | Cross-Validated Mean | Nadeau-Bengio 95% CI |
|---|:---:|:---:|
| **AUROC** | **0.8716** | **[0.8023, 0.9408]** |
| **AUPRC** (prevalence 0.322) | **0.8332** | [0.7434, 0.9230] |
| **Balanced Accuracy** | **0.7653** | [0.6821, 0.8486] |
| **Macro-F1** | **0.7697** | [0.6839, 0.8554] |
| **Brier Score (fold mean)** | **0.1436** | [0.1089, 0.1783] |
| **Brier Score (pooled)** | **0.1365** | pooled (no CI) |

### Baseline Comparisons (Paired Nadeau-Bengio Test)

| Baseline Model | Baseline AUROC | Difference (E1 − Baseline) | p-value | Conclusion |
|---|:---:|:---:|:---:|---|
| Early Fusion ElasticNet | 0.9295 | −0.0580 | 0.1778 | Not distinguishable with N=121 (p >= 0.05) |
| mRNA ElasticNet | 0.9169 | −0.0453 | 0.2996 | Not distinguishable with N=121 (p >= 0.05) |
| miRNA ElasticNet | 0.8481 | +0.0235 | 0.6209 | Not distinguishable with N=121 (p >= 0.05) |
| Methylation ElasticNet | 0.4601 | +0.4114 | < 0.0001 | Statistically distinguishable |
| Clinical-only (Age + Sex) | 0.6296 | +0.2419 | 0.0022 | Statistically distinguishable |
## Rules of the project

1. No tuning on outer-fold results. Every change after a result was seen goes into `results/model/forking_paths.md`.
2. Fitting steps that use labels run inside CV folds only.
3. Confidence intervals: Nadeau-Bengio corrected. More repeats do not make them much narrower (see `src/evaluation/cv_stats.py`).
4. Data lineage: omics matrices are UCSC Xena (GDC-derived) matrices; clinical labels come from GDC. See `data/DATA_CARD.json`.

## Known limitations

N = 121 from an enriched high-risk cohort; no independent test set; only about 19% of mRNA features lie in the 192 selected
pathways; 726 of 1,035 miRNA features have no edge in the selected pathways and are wired into the last pathway token
(see forking_paths.md, Fork 6); the model needs all three omics at inference (modality dropout only touches the gate).
