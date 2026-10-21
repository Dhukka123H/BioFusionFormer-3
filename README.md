# BioFusionFormer

**A pathway-aware hierarchical cross-modal Transformer for multi-omics fusion** - final-year project.
Cohort: TARGET-WT (pediatric Wilms tumour), N = 121 primary-tumour patients with mRNA, miRNA, methylation and clinical data.
Task: classify histology, favorable (82) vs unfavorable (39). Pathway prior: Reactome; miRNA prior: miRTarBase.

> Status: **CLI-first**. One headline model (**E1**). A GUI will be added later on top of `predict()`.
> Everything else that was tried is exploratory and is archived in git tag `v1-archive-all-experiments`.

## What is the headline result (read this before quoting any number)

* E1 cross-validated AUROC over the 25 outer folds: **0.8716** (Nadeau-Bengio 95% CI: [0.8023, 0.9408]). AUPRC: **0.8332**. Brier score: **0.1436** (fold mean), **0.1365** (pooled).
* Parameters: **47,221** effective free parameters (`count_effective_parameters()`), **4,249,284** dense trainable parameters.
* E1 is **not distinguishable** from the shallow elastic-net baselines (mRNA elastic net 0.9169, early fusion 0.9295) with N = 121.
* The pathway prior was **not** shown to help: the real-vs-random prior comparison (E5 control) gave p = 0.86.
* Attention rollout is almost uniform over pathways (max/min about 1.06x) and is **not** an explanation; Integrated Gradients is used instead.
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
streamlit run app/app.py                                # launch local research prototype GUI
```

## Rules of the project

1. No tuning on outer-fold results. Every change after a result was seen goes into `results/model/forking_paths.md`.
2. Fitting steps that use labels run inside CV folds only.
3. Confidence intervals: Nadeau-Bengio corrected. More repeats do not make them much narrower (see `src/evaluation/cv_stats.py`).
4. Data lineage: omics matrices are UCSC Xena (GDC-derived) matrices; clinical labels come from GDC. See `data/DATA_CARD.json`.

## Known limitations

N = 121 from an enriched high-risk cohort; no independent test set; only about 19% of mRNA features lie in the 192 selected
pathways; 726 of 1,035 miRNA features have no edge in the selected pathways and are wired into the last pathway token
(see forking_paths.md, Fork 6); the model needs all three omics at inference (modality dropout only touches the gate).
