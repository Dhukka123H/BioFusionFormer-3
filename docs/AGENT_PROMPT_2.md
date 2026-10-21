# Agent prompt 2 - fix the findings, run the clinical experiments, build the GUI, freeze

Paste everything below the line into the coding agent. Work stage by stage; stop only at the hard-stop conditions.

---

CONTEXT
Stages 0-8 of docs/AGENT_PROMPT.md are finished on branch cleanup/e1-only (frozen E1, reproducibility re-run, explanation
script, clinical code, inference CLI, model card). A review of the produced files found the problems F1-F8 below.
They are facts with evidence; fix them, do not argue with them. Then run the clinical experiments, build the GUI and freeze.

FINDINGS FROM THE REVIEW
F1  The #1 Integrated-Gradients pathway "Regulation of ornithine decarboxylase (ODC)" (R-HSA-350562, score 4.98, about 2x the next one)
    is exactly pathway index P-1 = the 192nd selected pathway. In the frozen E1 the 726 miRNA features without a prior edge are wired
    into this pathway token (Fork 6). Summing |IG| over its members therefore counts about 726 miRNA features plus its own 40 genes.
    This top "finding" is an artefact of the fallback wiring. The other top pathways (mRNA splicing 255 genes, translation initiation
    140 genes) lie at the 97th-99th percentile of pathway size (median pathway = 26 genes), so summed |IG| is also biased toward
    large pathways. The "55% generic pathways" statement is confounded by size.
F2  The matched label-permutation null for pathway stability was NOT run (explain_e1.py was run with --skip-null). It costs about
    1 hour. Without it the stability number (IG 0.3885 vs elastic net 0.5682) has no reference.
F3  results/FINAL_STATUS.md section 3 disagrees with results/model/E1_metrics.json: AUPRC 0.7725 vs 0.8332, balanced accuracy 0.7969 vs
    0.7653, macro-F1 0.7938 vs 0.7697, Brier 0.1365 with a CI vs fold-mean 0.1436 (0.1365 is the POOLED value). MODEL_CARD.md matches the JSON.
F4  The reported recalibration slope 4.207 / intercept 3.080 cannot be right: it implies that a predicted probability of 0.5 is
    recalibrated to 0.956, which contradicts the reliability plot. A rough fit on the plotted bins gives slope about 2.5, intercept about 0.1.
    Slope > 1 means the model is UNDER-confident (probabilities compressed towards the middle).
F5  Parameter documentation: the verified free-parameter count is 47,221. The "62,450" figure cannot be reproduced; parameter_count.md
    wrongly says it is the free-parameter figure. FINAL_STATUS.md per-encoder "effective" numbers (12,059 / 1,223 / 8,151) do not match
    the mask non-zeros (mRNA 7,148; miRNA 5,768 incl. 726 fallback edges; methylation 5,949; total 18,865).
F6  docs/PROJECT_GUIDE.md still says (around lines 318 and 322) that the E5 random-prior control "demonstrates the impact of biological
    topology" (E5 showed no difference, p = 0.86) and that real pathway consistency "significantly outperforms the shuffled-label null,
    p < 0.001" (the old data showed the opposite: 0.110 real vs 0.206 null; the new analysis has no null). README.md still contains
    hand-typed numbers (task 8.1 of the previous prompt was not done).
F7  reproducibility.md labels the folds "Bit-identical" although max |dp| = 1.8e-5; and it proves same-seed determinism only. It does not
    measure seed-to-seed variability (the earlier 0.846 -> 0.872 difference was an initialisation difference).
F8  The in-sample representation plot shows two perfectly separated clusters with PC1 = 99.8% of the variance. That is memorisation of the
    training labels, not evidence of generalisation. The clinical model (EHR2/EHR4/EHR2b) is implemented but has no results yet.

RULES (unchanged, restated)
1. New branch: git switch -c final/e1-gui (from cleanup/e1-only). Never `git add .`; check `git diff --ignore-cr-at-eol --stat`; do not merge.
2. Do not touch experiments/final_v1/ (frozen) or the frozen E1 code path. No tuning on outer-fold results. Every change made after a result was
   seen goes into results/model/forking_paths.md.
3. Nadeau-Bengio corrected intervals only. Reports are GENERATED from JSON/CSV, never typed by hand. Differences with p >= 0.05 are described as
   "not distinguishable with N=121". Do not write "outperforms", "stable", "robust", "biologically validated" unless a corrected test supports it.
4. Use at most 2 CPU threads. The team laptops have 8 GB RAM: log peak memory of every long run, run one training job at a time.
5. Hard stops (report and wait): a test fails for a reason you cannot fix; any frozen file would change; E1 numbers regenerate differently from
   results/model/E1_metrics.json.

======================================================================
STAGE A - Explanation analysis v2  (compute about 1-1.5 h)
A1  Fallback audit: confirm that selected pathway P-1 is R-HSA-350562; count the miRNA features routed to it by the fallback (expected 726) and
    by genuine edges. Write results/explainability/fallback_audit.md.
A2  Re-aggregate IG per pathway in TWO ways, always reported side by side:
      (i)  sum of |IG| over genuine member features, EXCLUDING the 726 fallback-routed miRNA edges;
      (ii) mean |IG| per member feature (size-normalised).
    Report, for both: top-20 list, Spearman correlation between pathway size (member features present in the data) and the attribution
    (this quantifies the size bias), and how many of the top-20 are generic translation/ribosome/splicing pathways.
A3  Stability across the 5 saved fold models: mean pairwise top-20 Jaccard for IG (i), IG (ii) and for the per-fold elastic net.
A4  MATCHED null (this was skipped before): shuffle the TRAINING labels inside each of the 5 folds, train with the SAME procedure as E1
    (max_epochs 100, patience 20, same inner validation rule, same seed rule, seed 42 only), compute the same IG pipeline on the test patients.
    3 permutations x 5 folds = 15 trainings (about 4 min each at 1 thread). Make it resumable per (permutation, fold) with atomic JSON files.
    Report real vs null Jaccard for (i) and (ii). With 3 permutations give NO p-value; say so.
A5  Recalibration: logistic regression WITHOUT penalty on logit(pooled out-of-fold probability) -> slope, intercept, 1000-draw patient bootstrap
    CI. Then 5-fold cross-validation over patients of this 2-parameter calibrator (fit on 4/5, apply to 1/5) and report Brier before and after.
    If the slope is not about 2-3 with intercept near 0, explain why the previous 4.207/3.080 was produced and fix the code. Redraw the figure with
    bin counts printed.
A6  Redraw the representation figure and caption it: "in-sample; separation reflects training labels; not evidence of generalisation".
A7  Write results/explainability/summary.md from the JSON outputs only. State plainly that ODC was a fallback artefact.

======================================================================
STAGE B - Make every document agree with the data
B1  Create scripts/make_reports.py. Source of truth: results/model/E1_metrics.json (+ explainability JSON). It regenerates the numeric tables of
    results/FINAL_STATUS.md, docs/MODEL_CARD.md and the "Results" block of README.md. Fold-mean metrics keep their Nadeau-Bengio CI. A pooled
    metric must be labelled "pooled" and must not carry the NB CI. Add tests/test_reports_match_json.py that parses the three documents and
    compares every number with the JSON.
B2  Parameters: use 47,221 (free) and 4,249,284 (dense) everywhere. Remove every statement that 62,450 is the free-parameter count
    (results/model/parameter_count.md, data/DATA_CARD.json, configs/default.yaml, FINAL_STATUS.md). Write instead: "62,450 appeared in an earlier
    draft and could not be reproduced". Regenerate the per-encoder table from the model: mask non-zeros (7,148 / 5,768 / 5,949) plus per-encoder
    640 (bias 192, LayerNorm 384, token projection 64), plus [CLS] 32 and modality embedding 96.
B3  docs/PROJECT_GUIDE.md: fix the E5 sentence and the permutation-stability sentence (see F6); mark all sections about E2-E10 as
    "archived, exploratory"; grep the file for outperform|significant|robust|stable|adaptive and make each statement match the data
    (gate weights are 0.28-0.40 for every patient, attention rollout is uniform at 1.06x, the prior was not shown to help).
B4  results/model/reproducibility.md: replace "Bit-identical" by "numerically identical (max |dp| 1.8e-5)"; add "this confirms same-seed determinism
    only; seed-to-seed variability is measured in C0".
B5  Remove hand-typed numbers from README.md (they now come from make_reports.py).

======================================================================
STAGE C - Seed sensitivity and clinical experiments (exploratory)
C0  (about 1 h) Add a --seeds option to scripts/run_e1.py (default stays 42,123,999; the signature then differs, so it never touches frozen files).
    Run repeat-0 (5 folds) with seeds 7,8,9 under tag final_v1_seedcheck. Report the per-fold AUROC difference to the frozen run and its mean/SD:
    this is the initialisation variance that the 0.846 -> 0.872 history already hinted at. Quote it next to the E1 AUROC in the model card.
C1  Clinical: python scripts/run_clinical.py --status, then run EHR2 (omics + clinical, primary set: age, sex, stage) and EHR4 (clinical-only)
    with --n-repeats 3 (15 folds, the first 15 folds of E1, decided in advance). EHR2b (sensitivity set) only if time remains; label it
    sensitivity-only. One job at a time. Expected: no accuracy gain over E1; clinical-only about 0.63.
C2  Compare EHR2 with E1 on the same 15 folds and EHR4 with the elastic-net clinical-only baseline: Nadeau-Bengio test, Wilcoxon, bootstrap CI of
    the paired difference. Generate results/clinical/summary.md from JSON. Say plainly if not distinguishable.
C3  The clinical model stays OUT of the GUI and the headline; it appears in the model card only as an exploratory result.

======================================================================
STAGE D - GUI (Streamlit, local, nothing stored)
D1  app/app.py calling src/inference/predict.py. Pages:
      1 About and limitations (research prototype; N=121 enriched cohort; no external validation; all three omics required).
      2 Upload: mRNA, miRNA, methylation CSVs with a schema checker and clear error messages; a "load demo data" button.
      3 Results table per patient: model score; calibrated probability ONLY if the A5 calibrator exists and then labelled "calibrated on
        cross-validated predictions, approximate"; modality gate weights (bar); top pathways from IG (variant (i) and (ii) both shown, fallback
        excluded) with the warning "generic translation/splicing pathways may reflect proliferation or tumour content, not histology".
      4 Model performance: cross-validated AUROC with the Nadeau-Bengio interval, the calibration plot, comparison with the elastic-net
        baselines (not distinguishable) - all read from E1_metrics.json.
      5 Download predictions as CSV.
D2  Safeguards: refuse inputs with less than 90% of the expected features; state that all three omics are required (the model cannot run with a
    missing modality); no network calls; no data written to disk; peak memory < 2 GB (log it).
D3  Demo data: export 3 cohort patients as example CSVs. Label the demo "training patients: predictions here are NOT performance evidence".
D4  Tests: validators, end-to-end predict() on the demo files, and a Streamlit AppTest smoke test. Document `streamlit run app/app.py` in README.md.
D5  If the environment allows, save 4 screenshots to docs/img/ for the project report.

======================================================================
STAGE E - Freeze and hand-off
E1  python -m pytest tests -q (all green). python scripts/make_reports.py. Commit specific paths in small commits on final/e1-gui; push; list commits.
E2  Tag v2-e1-final. Do not merge.
E3  results/FINAL_STATUS.md (generated): what was verified, every deviation, every failed check, open issues (Fork 6 fallback wiring, Fork 8 no
    missing-modality support, no external validation, enriched cohort), and a short list of claims the project IS allowed to make.
E4  Produce BioFusionFormer_final.zip with `git archive` of the tagged commit (code, configs, small data, docs) and list separately the large
    git-ignored artefacts (models/final/, data/raw/) with their sizes. Then stop.
