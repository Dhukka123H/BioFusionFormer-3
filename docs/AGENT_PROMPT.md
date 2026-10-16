# Agent prompt - BioFusionFormer, E1-only repository (CLI first, GUI later)

Paste everything below the line into the coding agent (Antigravity). Work stage by stage. Stop only at the marked GATES.

---

CONTEXT
You are working in the cleaned BioFusionFormer repository (final-year project: pathway-aware cross-modal Transformer for
TARGET-WT Wilms tumour, N=121, task = histology favorable (82) vs unfavorable (39)). The repository was reduced to ONE model,
E1 (3 omics, pathway attention bias ON, gated fusion ON, strong miRNA mask, frozen config in configs/default.yaml).
All ablations E2-E10 are archived in git tag v1-archive-all-experiments. READ README.md, CLEANUP_LOG.md and
results/model/forking_paths.md (Forks 5-10) FIRST. They list problems found in the previous work; do not repeat them.
Do NOT build a GUI now. Everything is CLI/Python API. A GUI will be added later on top of the predict() function.

HARD RULES
1. Before anything: check that no training process is running (python scripts/run_e1.py --status; task manager). Never write to
   experiments/final_v1/E1/ except via scripts/run_e1.py. Never open or truncate logs/*_all_sessions.log with "w".
2. Work on a new branch: git switch -c cleanup/e1-only. Never use `git add .`; add specific paths. Many files show as modified only because of CRLF/LF: check `git diff --ignore-cr-at-eol --stat` and commit only real changes. Do not merge or force-push.
   .gitignore must keep data/raw/, models/, *.pt, experiments/*_smoke out of git. Do not commit files > 20 MB.
3. No tuning on outer-fold results. Anything you change after seeing a result must be appended to results/model/forking_paths.md
   (what, why, which run). Fitting that uses labels happens INSIDE CV folds only.
4. Confidence intervals: Nadeau-Bengio corrected only (src/evaluation/cv_stats.py). Never print mean +/- 1.96 SD/sqrt(K) as a CI.
5. Do not write claims into reports by hand. Generate reports from JSON/CSV outputs and describe differences with p >= 0.05 as
   "not distinguishable with N=121". Never write "outperforms" or "stable pathways" unless a corrected test supports it.
6. Use at most 2 CPU threads (torch.set_num_threads(2), OMP/MKL_NUM_THREADS=2) unless told otherwise. Check free RAM/disk first.
7. If a command fails, diagnose and retry up to 3 times with a different approach, then report. Never silently skip a step.
8. Open-access GDC data only. Never attempt controlled-access (dbGaP) files.

======================================================================
STAGE 0 - Environment and tests (the tests could NOT be run when the repo was cleaned)
0.1 Create/activate the venv, pip install -r requirements.txt. Report python and torch versions.
0.2 python -m pytest tests -q. Expected: tests/test_cv_stats.py (4) pass; tests/test_e1_model.py and
    tests/test_preprocessing_leakage.py need torch. If a test fails: decide whether the TEST or the CODE is wrong. Fix a wrong test.
    If the code is wrong, do NOT change src/ silently: report it and wait.
    One test intentionally documents a limitation: test_missing_modality_is_not_supported_documented_limitation must PASS
    (KeyError when a modality is missing).
0.3 python scripts/run_e1.py --smoke --run  (2 folds, 1 seed, 3 epochs, goes to experiments/final_v1_smoke). Must finish and write
    fold files. Then python scripts/run_e1.py --smoke --status.
0.4 Print BioFusionFormerModel(...).count_parameters() and .count_effective_parameters() for the real E1 shapes
    (mRNA 11,171; miRNA 1,035; methylation 9,778; 192 pathways) and put both numbers in results/model/parameter_count.md.
    Expected: dense about 4.25M, effective about 50-60k.
Report and continue unless a test failed for a reason you cannot fix.

======================================================================
STAGE 1 - GDC clinical data and labels (GDC is live)
1.1 Check https://api.gdc.cancer.gov/status returns OK. If it does not, retry every 10 minutes for 2 hours, then stop and report.
1.2 python scripts/download_clinical.py (open-access clinical + TARGET supplements; writes data/raw/clinical/ and
    data/processed/clinical/clinical_raw_121.csv, configs/clinical_features.yaml, reports/clinical_data_audit.md).
    The script cross-checks against the Xena clinical table if that file exists; it may not exist now. That is fine, say so.
1.3 python scripts/build_labels.py. It must report: class counts 82/39; histology counts identical to the stored counts in ALL 25
    folds (train and test) of data/splits/splits_5x5.json; supplement agreement. If ANY check fails, STOP and report. Do not use
    --allow-count-change.
1.4 Record provenance (GDC data release, download date, file md5s) in data/DATA_CARD.json under "clinical".
1.5 Confirm in configs/clinical_features.yaml that no histology/outcome/follow-up/administrative column is a candidate feature.
    PRIMARY clinical feature set for models: age at diagnosis (days), sex, stage. Race, ethnicity and the biopsy-QC percentages
    (tumour nuclei, necrosis, stroma) are SENSITIVITY-ONLY (percent_necrosis is 0 in 109/121, stroma may be a soft histology proxy;
    race/ethnicity have no biological rationale here, 18 Black patients, 25% ethnicity not reported).

======================================================================
STAGE 2 - Verify the frozen E1 and regenerate the report (NO retraining)
2.1 python scripts/run_e1.py --status -> must show 25/25 folds DONE with config signature a42642001739. If the runner refuses with
    "different configuration", STOP and report (do not use --force).
2.2 python scripts/run_e1.py --report. Check results/model/E1_summary.md: AUROC mean, Nadeau-Bengio interval (expected about
    [0.80, 0.94]), comparison with baselines, gate table, rollout max/min ratio (expected about 1.06). Copy the main numbers into
    results/model/E1_metrics.json (done by the script).
2.3 Fold-count policy (decided now, before any new result is seen):
    - Headline E1 stays on all 25 folds (already computed).
    - Every NEW exploratory variant uses --n-repeats 3 (15 folds, first 3 repeats of splits_5x5.json) and is compared with E1 on
      exactly the same 15 folds. Reason: with N=121 the corrected standard error factor sqrt(1/K + n_test/n_train) is 0.589 (K=10),
      0.560 (K=15), 0.536 (K=25); going from 15 to 25 folds narrows the interval by about 4%; repeats only reduce partition noise
      (per-repeat mean AUROC of E1 ranges 0.856-0.898). Do not change this after seeing results.

======================================================================
STAGE 3 - Reproducibility re-run (needed anyway: per-fold models were not saved, and E1 was run twice before)
3.1 python scripts/run_e1.py --run --tag final_v1_repro --n-repeats 1  (5 folds; writes experiments/final_v1_repro/ and
    models/e1_folds_final_v1_repro/; the frozen results are untouched).
3.2 Compare per-fold AUROC and per-patient probabilities of the 5 re-run folds with experiments/final_v1/E1 folds 0-4.
    Write results/model/reproducibility.md: max absolute difference in probabilities, AUROC differences. Same machine and torch version
    should give (almost) identical numbers. If they differ by more than 0.01 AUROC on any fold, report it: it measures run-to-run
    variability and must be quoted next to every E1 number. Do NOT replace the frozen results.
ETA: about 12 minutes per fold with 1 thread (3 seeds each).

======================================================================
STAGE 4 - Corrected explanation analysis (scripts/explain_e1.py, CLI)
Write and test scripts/explain_e1.py. It uses the 5 saved fold models of Stage 3 (seed 42 of each fold). Requirements:
4.1 Primary attribution = Integrated Gradients (captum) over ALL THREE modalities, for the fold's TEST patients, aggregated to
    pathways with the same masks as the model (gene/miRNA/CpG -> pathway). Use n_steps >= 50, baseline = zeros in standardised space
    (training-fold mean). Report per-modality and summed pathway attributions. Attention rollout is secondary only; print its max/min ratio.
4.2 Per-fold baseline for comparison: elastic-net logistic regression on mRNA fitted on the TRAINING fold only (fixed documented C and
    l1_ratio), coefficients mapped to pathways with the same mask. Never fit on all 121 patients for a comparison with a fold model.
4.3 Stability: top-20 pathways per fold -> mean pairwise Jaccard across the 5 folds, for (a) IG, (b) elastic net, (c) null.
4.4 MATCHED null: train E1 on shuffled labels with the SAME procedure as E1 (max_epochs 100, patience 20, same inner validation, same
    seed rule), 5 permutations x the same 5 folds, then the same IG pipeline. The old null used 35 epochs / patience 10, which made it
    non-comparable. If compute is too high, use 3 permutations and say so.
4.4b Also report each pathway's stability frequency across folds and the fraction of top-20 pathways that are translation/ribosome/
    splicing (generic high-variance pathways); state clearly that these may reflect proliferation, tumour content or quality rather
    than histology-specific biology.
4.5 IG vs elastic-net agreement per fold (Jaccard of top-20 and Spearman over all 192 pathways); report mean and range.
4.6 Representation: train ONE model with the frozen config on all 121 patients (3 seeds, average embeddings) and project its embeddings
    (PCA, UMAP). Label the figure "in-sample, visualisation only"; no silhouette claims. Never use out-of-fold embeddings from
    different fold models.
4.7 Calibration: pooled out-of-fold probabilities (average over repeats) with (a) Brier, (b) logistic recalibration slope/intercept
    (more stable than 5 bins with N=121), (c) the NESTED-CV elastic-net baseline predictions if per-patient baseline predictions exist,
    otherwise omit the baseline curve instead of refitting a different model.
4.8 Outputs: results/explainability/*.md/json and figures/explain/*.png generated from the data; a text summary that only contains
    numbers and the sentences allowed by Rule 5.

======================================================================
STAGE 5 - Clinical modality (exploratory, additive, E1 untouched)
Create new files only; do not edit full_model.py behaviour.
5.1 src/preprocessing/clinical.py: fold-safe ClinicalPreprocessor (age: log1p + z-score in training fold; sex binary; stage ordinal I<II<III<IV
    with IIIB merged into III, in-fold imputation + missing indicator). Reuse/extend src/preprocessing/transformers.py. Test: changing
    test-fold values must not change the fitted transformer. Test: a column-name blacklist (histology, risk group, vital status, follow-up,
    administrative) fails if any forbidden column appears. Assert patient order equals the omics order.
5.2 src/encoders/clinical_encoder.py + src/biofusionformer/full_model_clinical.py: a 4th modality made of ONE clinical token
    (2-layer MLP, LayerNorm, dropout >= 0.3 -> d_model). The existing AdaptiveModalityGate expects (B, M, P, d_model); handle the clinical
    token by repeating/pooling so that the existing gate code is not modified. Add a regression test: with use_clinical=False the new class
    gives identical parameter names/shapes and identical eval output to BioFusionFormerModel for the same seed.
5.3 Runs (new tags; --n-repeats 3 = 15 folds; same seeds 42/123/999; resumable like run_e1.py; atomic per-fold JSON; config signature):
      EHR2  : omics + clinical (primary set)           EHR4 : clinical-only through the same code (expect AUROC about 0.63)
      EHR2b : omics + clinical (primary + sensitivity set), labelled sensitivity-only
    Compare with E1 (first 15 folds) and with the elastic-net clinical-only baseline using the corrected test, the Wilcoxon test and a
    bootstrap CI on the paired differences. Say plainly if not distinguishable. Expect no accuracy gain; clinical-only is about 0.63.
5.4 Do NOT run a "test-time missing modality" experiment: the current architecture cannot handle missing modalities (Fork 8). Note it as future work.

======================================================================
STAGE 6 - Python/CLI inference and the final model artefact (no GUI)
6.1 scripts/train_final.py: train the frozen-config E1 on all 121 patients with the 3 seeds; save to models/final/ (git-ignored) the weights,
    fitted imputers/scalers, feature name lists, pathway list, masks, bias matrix, config, git commit hash, seeds. Save a small
    models/final/manifest.json (committed) with hashes.
6.2 src/inference/predict.py: predict(mrna_df, mirna_df, meth_df) -> DataFrame(patient_id, probability_unfavorable, gate_mrna, gate_mirna,
    gate_meth, top_pathways_IG, warnings). Strict validation: Ensembl gene IDs without version suffix, log2(count+1) for mRNA, log2(RPM+1)
    for miRNA precursor-level names as in data/processed/mirna_unique.parquet, beta values in [0,1] for methylation gene-level promoter
    means, same feature order and the training-set feature lists; clear error messages; report the fraction of expected features present
    and refuse below 90%. All three omics are REQUIRED (state this in the error). Never store uploaded data.
6.3 scripts/predict.py CLI: python scripts/predict.py --mrna f1.csv --mirna f2.csv --meth f3.csv --out preds.csv. Tests for the validator
    and an end-to-end test on 3 patients of the cohort.
6.4 docs/MODEL_CARD.md: intended use (research prototype; not for clinical decisions), training cohort (N=121, enriched, TARGET-WT), the
    CROSS-VALIDATED performance with the Nadeau-Bengio interval (never in-sample numbers), calibration, limitations, the Fork 6-8 issues.
6.5 Document in docs/ADDING_A_MODALITY.md how a future MRI/CT modality would plug in: pre-extracted image feature vectors or patch embeddings
    (extracted outside this repo with a frozen backbone) -> an ImagingFeatureEncoder returning tokens (B, T, d_model) + an availability mask.
    Implement src/modalities/registry.py with a modality registry (name -> encoder, token count, mask) and an OPT-IN key-padding mask in a NEW
    transformer wrapper for missing modalities (the frozen E1 path stays unchanged). Unit-test with synthetic image tokens that a fifth
    modality plugs in and that the wrapper runs when it is absent. Claim NO imaging performance (TARGET-WT has no matched imaging here).
6.6 GUI is out of scope now. Only keep predict() stable so a Streamlit/web front end can call it later.

======================================================================
STAGE 7 - Data source of record: Xena vs GDC-native  (default = keep; do NOT rebuild unless told "go GDC-native")
7.1 Default: the omics matrices stay the Xena-derived (GDC-hub) matrices used by the frozen E1. Make sure README.md, data/DATA_CARD.json and
    docs/PROJECT_GUIDE.md say exactly that ("GDC-derived data obtained through the UCSC Xena GDC hub; clinical labels from the GDC API"),
    and mention nowhere that Xena was a "fallback" for an outage. Grep all .md/.json for "fallback", "503", "maintenance" and fix.
7.2 Only if I write "go GDC-native": (a) build manifests with scripts/phase_b_gdc_metadata.py for the 121 patients (open access, primary tumour,
    gene-level STAR counts, miRNA expression, methylation beta); (b) download with gdc-client; (c) rebuild the three matrices with the same
    preprocessing; (d) compare with the Xena matrices (per-sample correlations, identical gene sets, value ranges); (e) save as
    data/processed_gdc/ and re-run E1 under tag e1_gdc with --n-repeats 5 (about 5 hours at 1 thread). The numbers WILL change; report both.

======================================================================
STAGE 8 - Wrap-up
8.1 Regenerate README.md "headline result" section values from results/model/E1_metrics.json (no hand-typed numbers).
8.2 Update docs/PROJECT_GUIDE.md: remove or correct every statement contradicted by results/model/forking_paths.md (Forks 5-10): the
    "62,450 < 500k budget" claim, "real prior outperforms random prior", "stable pathways", "robust to missing modalities", "fallback token",
    "adaptive gate" (gate weights are about 0.29-0.39 for every patient), 95% CI values.
8.3 Run python -m pytest tests -q once more. Commit specific paths on cleanup/e1-only in small commits, git push -u origin cleanup/e1-only,
    list the commits. Do not merge.
8.4 Final report (results/FINAL_STATUS.md, generated from files): what was verified, every deviation, every failed check, all new numbers,
    open issues (Fork 6 fallback-token bug, Fork 8), and the exact next step. Then stop.
