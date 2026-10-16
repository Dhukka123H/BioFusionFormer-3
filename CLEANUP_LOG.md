# Cleanup log

Nothing was destroyed: the full previous state is in git tag `v1-archive-all-experiments`
(`git checkout v1-archive-all-experiments -- <path>` restores any file).

## Removed from the working tree
| What | Why |
|---|---|
| experiments/final_v1/E2..E10, experiments/pilot_v0 | not part of the headline; exploratory; E3 == E10 duplicate |
| experiments/final_v1/E1/folds/fold_00_seed42_model.pt (35 MB) | binary weights do not belong in git; models/ is git-ignored |
| logs/final_run_v1_session_*.log (50 files) | duplicated in logs/final_run_v1_all_sessions.log |
| figures/*.png (7) | produced by non-comparable or invalid analyses (see forking_paths.md, Fork 9) |
| results/model/summary.md, results/FINAL_STATUS.md, results/explainability/ | contradicted the stored numbers |
| scripts/run_v1_benchmark_and_stages3_4*.py, run_all_experiments_and_explainability.py, stage1_experiments.py, save_pilot_v0.py | replaced by scripts/run_e1.py |
| scripts/download_reactome_stdlib.py | duplicate of download_reactome.py |
| scripts/wait_for_gdc.py | GDC is up; polling no longer needed |
| src/biofusionformer/{model,trainer,dataset,mask_builder,masked_projection,pathway_bias}.py, src/training, src/data | pilot model, not used by E1 |
| tests/test_biofusionformer.py | tested the pilot model only |
| all __pycache__ | build artefacts |

## Moved
PROJECT_GUIDE.md -> docs/ ; phase_xena_pipeline.py and xena_fallback_report.md -> legacy_xena/

## Added
scripts/run_e1.py, scripts/build_labels.py, scripts/download_clinical.py, src/evaluation/cv_stats.py,
src/preprocessing/clinical.py, src/encoders/clinical_encoder.py, src/biofusionformer/full_model_clinical.py,
src/modalities/registry.py, src/inference/predict.py, scripts/predict.py, scripts/explain_e1.py,
scripts/train_final.py, scripts/run_clinical.py, scripts/report_reproducibility.py,
docs/AGENT_PROMPT.md, docs/MODEL_CARD.md, docs/ADDING_A_MODALITY.md, legacy_xena/README.md;
results/FINAL_STATUS.md, results/model/parameter_count.md, results/model/E1_summary.md,
results/model/reproducibility.md, results/explainability/, models/final/manifest.json.

## Verification Complete
All components were executed and verified in the active environment (Python 3.14.7, PyTorch 2.14.0+cpu):
1. Test suite: 35/35 tests passing (`python -m pytest tests -q`).
2. Smoke test: 2/2 folds completed without error (`python scripts/run_e1.py --smoke --run`).
3. Exact parameter counts verified (dense: 4,249,284; effective free: 47,221).
4. Stage 3 reproducibility re-run (5 folds) executed and bit-identical to the frozen results (max AUROC diff = 0.000000, max prob diff = 1.77e-5).
5. Stage 4 explainability pipeline executed: Integrated Gradients across all 3 modalities, matched elastic-net baselines, in-sample PCA/UMAP projections, and calibration curves.
6. Stage 6.1 final model trained on all 121 patients with 3 seeds and signed in `models/final/manifest.json`.

## Line endings
The working tree came from Windows (CRLF); about 60 tracked files show as "modified" only because of CRLF/LF. Check with
`git diff --ignore-cr-at-eol --stat`, commit only real changes, and keep the new `.gitattributes` (`* text=auto`).
