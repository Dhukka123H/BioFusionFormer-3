# BioFusionFormer E1 — Empirical Claims Registry

> **Status:** Authoritative Guidance for Scientific Reporting, Supervisor Review, and Manuscript Drafts.  
> **Source-of-Truth:** `results/model/E1_metrics.json`, `results/explainability/explainability_results.json`, `results/clinical/clinical_metrics.json`, `results/model/seed_sensitivity_c0.md`.

---

## 1. Permitted Claims (Empirically Supported)

The permitted claims below reflect internal cross-validation and empirical evaluations on the retrospective TARGET-WT cohort, subject to the methodological constraints and evaluation caveats detailed below.

### 1.1 Predictive Discrimination
- **Claim:** BioFusionFormer E1 achieves strong internal cross-validated discrimination for unfavorable vs. favorable histology in pediatric Wilms tumor on the enriched TARGET-WT retrospective cohort ($N=121$, prevalence 32.2%).
  - **Empirical Evidence:**
    - Mean AUROC: **$0.8716$** (95% Nadeau-Bengio corrected CI: `[0.8023, 0.9408]`).
    - Mean AUPRC: **$0.8332$** (95% Nadeau-Bengio corrected CI: `[0.7434, 0.9230]`, baseline cohort prevalence $32.2\%$).
    - Balanced Accuracy: **$0.7653$** (95% Nadeau-Bengio corrected CI: `[0.6821, 0.8486]`, chance level $50.0\%$).
    - Macro-F1: **$0.7697$** (95% Nadeau-Bengio corrected CI: `[0.6839, 0.8554]`).
    - Fold-Mean Brier Score: **$0.1436$** (95% Nadeau-Bengio corrected CI: `[0.1089, 0.1783]`); Pooled Brier Score: **$0.1365$**.

### 1.2 Model Parameters
- **Claim:** BioFusionFormer employs biological prior masking across 192 Reactome pathway tokens.
  - **Empirical Evidence:**
    - Free learnable parameters: **47,221** effective degrees of freedom.
    - Dense representation parameters: **4,249,284** memory footprint.
    - *(Note: The 62,450 figure that appeared in an earlier draft was unverified and could not be reproduced).*

### 1.3 Explainability Audit & Methodological Transparency
- **Claim:** Dual-aggregation Integrated Gradients (genuine member sum vs. size-normalised mean) exposes severe pathway size bias and excludes the fallback edges from the attribution; the frozen E1 still contains the wiring (Fork 6).
  - **Empirical Evidence:**
    - Fallback wiring audit: Regulation of ornithine decarboxylase (ODC, `R-HSA-350562`) was confirmed as an artefact of Fork 6 fallback routing (726 orphaned miRNAs mapped to token $P-1$) and drops out once fallback edges are excluded from the attribution.
    - Pathway size dominance: Genuine member sum attributions correlate strongly with pathway feature count (Spearman $\rho = 0.9099, p = 1.60 \times 10^{-74}$).
    - Generic pathway dominance: 15 of the top-20 pathways (75%) sit in Cluster 1 (translation/ribosome/splicing pathways); 18 of the top-20 mRNA genes (90%) are ribosomal protein genes (`RPL` and `RPS` families).
    - Cluster-level stability result: Real cluster stability is 0.462 (summed) and 0.437 (size-normalised) vs permutation nulls (0.43, 0.26, 0.31) and (0.36, 0.19, 0.23); with 3 permutations, no p-value is possible.
    - Null baseline comparison: After size-normalisation, pathway attribution stability across 5 folds ($0.2765$) falls entirely within the matched label-permutation null distribution ($0.2306\text{--}0.3719$, null mean $0.2870$).
    - IG vs ElasticNet overlap: The 0.01 IG-vs-ElasticNet Jaccard overlap (rank correlation $\rho = 0.02$) is for the size-normalised version; the raw member-sum version has Jaccard 0.49 and rank correlation $\rho = 0.47$.

### 1.4 Clinical Prototype Deployment Safeguards
- **Claim:** The deployment interface implements no network calls by design (not independently verified), input validation (requiring all 3 modalities and $\ge 90\%$ feature coverage), and explicit calibration transparency.
  - **Empirical Evidence:**
    - Predictions display raw model scores with an explicit **"Not Calibrated"** label.
    - Single-repeat out-of-fold calibration slopes ($1.434\text{--}1.974$, with repeats 3, 4, and 5 falling below $1.769$) do not satisfy consistency with the repeat-averaged 95% CI `[1.769, 3.154]`, properly precluding premature calibration claims.

### 1.5 Reproducibility
- **Claim:** BioFusionFormer demonstrates deterministic same-seed reproducibility with run-to-run variation bounded by machine precision ($< 1.8 \times 10^{-5}$ in probabilities, $\Delta \text{AUROC} = 0.000000$).
  - **Empirical Evidence:** Same-seed reproducibility was verified in the same Python/torch environment.

### 1.6 Known Evaluation Caveats
- **Unsupervised Cohort-Wide Filtering:** Pathway selection and some feature variance filters used the full cohort ($N=121$, unsupervised without label information).
- **Preprocessing Partitioning:** Inner validation was split AFTER preprocessing was fit on the whole outer-train partition (not fully isolated); outer test was isolated.
- **Whole-Cohort Deployment Training Budget:** The final deployment model was trained on all 121 cohort samples for a fixed schedule of 100 epochs without early stopping.

---

## 2. Forbidden Claims (Statistically or Methodologically Unsupported)

The following claims are strictly **unsupported** by empirical evidence and must **never** be made in any report, thesis, paper, or presentation.

| # | Forbidden Claim | Reason & Empirical Counter-Evidence |
| :---: | :--- | :--- |
| **F1** | *"BioFusionFormer significantly outperforms shallow or linear baselines."* | **False.** In paired Nadeau-Bengio corrected tests across 25 folds, E1 (0.8716) is **not statistically distinguishable** from linear baselines in `results/model/E1_metrics.json`: mRNA ElasticNet achieves 0.9169 (E1 - baseline: -0.0453, $p = 0.2996$), Early-Fusion ElasticNet achieves 0.9295 (E1 - baseline: -0.0580, $p = 0.1778$), and miRNA ElasticNet achieves 0.8481 (E1 - baseline: +0.0235, $p = 0.6209$). With $N=121$, no complex architecture advantage can be claimed. |
| **F2** | *"BioFusionFormer provides superior value compared to clinical variables."* | **False.** Adding clinical features to omics gave no significant gain (EHR2 vs E1, difference -0.0175, NB 95% CI [-0.1213, +0.0863], $p = 0.7227$; EHR2 AUROC 0.8506 vs E1 0.8681 on the same 15 folds). Furthermore, an omics-vs-clinical-only paired test was NOT run. EHR4 (clinical-only) achieves AUROC 0.6214 (NB 95% CI [0.4639, 0.7789], spanning 0.5); its Brier score of 0.2486 is worse than a constant predictor of 0.2184. |
| **F3** | *"Integrated Gradients identified novel Wilms tumor biological drivers or therapeutic pathway targets."* | **False.** Real size-normalised stability ($0.277$) does not exceed matched label-permutation nulls ($0.23\text{--}0.37$). Top pathways and genes reflect generic translation/ribosome/splicing pathways and pathway size ($\rho = 0.91$). IG and ElasticNet share negligible overlap in size-normalised space (Jaccard $0.01$, rank correlation $\rho = 0.02$). |
| **F4** | *"The model outputs calibrated clinical probabilities of unfavorable histology."* | **False.** Deployment uses uncalibrated model outputs. Cross-validated single-repeat slopes vary widely ($1.434\text{--}1.974$), and the deployment model was trained on the full cohort without cross-validation. GUI strictly displays raw scores with "Not Calibrated". |
| **F5** | *"The model is robust to random initialization seed variations ($r \approx 1.0000$)."* | **False.** The $r \approx 1.0000$ correlation was an artefact of evaluating whole-cohort deployment models in-sample on overfitted training data (numbers from previous deployment checkpoints). Real cross-validation reveals that the per-fold AUROC difference between two seed sets has SD 0.121 (5 folds). Over 25 folds, two re-runs' mean AUROCs would typically differ by about 0.05 (95% range +/-0.047). Differences below about 0.05 are not interpretable. The SD is estimated from 5 folds only (95% interval 0.07-0.35), so the true 25-fold range could be anywhere from about +/-0.03 to +/-0.14. The 5-fold seed-check CI is not meaningful (upper bound above 1). Brier on the same 5 folds changed by +0.036 between seed sets (0.135 vs 0.171). |
| **F6** | *"BioFusionFormer is ready for clinical deployment, patient triage, or therapy selection."* | **False.** Academic research prototype only. Evaluated on an enriched retrospective cohort ($N=121$) with no external validation. Not cleared or approved by any regulatory body. |
| **F7** | *"miRNA fallback routing to ODC represents a biological interaction."* | **False.** Fully confirmed as a technical routing artefact where 726 orphaned miRNAs without Reactome prior edges were wired into token index $P-1=191$. |
| **F8** | *"The final whole-cohort deployment model generalizes without overfitting."* | **False.** The final deployment model trained on all 121 patients is overfit in-sample (AUROC 1.000, Brier 0.0009, numbers from previous deployment checkpoints); in-sample evaluation reflects memorisation of the training cohort rather than generalization. |
| **F9** | *"Modality gate weights represent a stable biological mechanism."* | **False.** Modality gate weights change with the seed (mRNA: 0.32 vs 0.44 vs 0.40; Methylation: 0.33 vs 0.23 vs 0.31, numbers from previous deployment checkpoints). Because gating weights fluctuate depending on random seed initialization, gate weights are not stable, and "adaptive gate" cannot be claimed as a biological mechanism. |
| **F10** | *"Biological pathway prior masking significantly improves representation learning over random wiring."* | **False.** The pathway prior was not shown to help (real vs random prior $p = 0.86$, archived E5). |
| **F11** | *"The model has been externally validated."* | **False.** There is no external validation. The model has only been evaluated on a single retrospective cohort ($N=121$, TARGET-WT). |
| **F12** | *"BioFusionFormer predicts patient prognosis, survival, or disease progression."* | **False.** BioFusionFormer is trained strictly on binary histologic subtype classification (favorable vs. unfavorable histology at resection). It was never evaluated for or shown to predict patient survival, event-free survival, overall mortality, or therapeutic response. Exploratory vital-status models were discarded in early phases due to lack of predictive signal. |
| **F13** | *"BioFusionFormer processes or fuses medical imaging, radiological scans, or histopathology whole-slide images."* | **False.** The model exclusively ingests molecular multi-omics tabular matrices (mRNA, miRNA, DNA methylation). It contains no convolutional, vision, or digital pathology encoders, and does not analyze CT, MRI, ultrasound, or whole-slide microscopy images. The label "histology" refers strictly to pathologist-assigned ground-truth classification labels, not imaging data inputs. |

---

## 3. Summary Compliance Checklist

Before submitting code, reports, or text to supervisors or journals:
- [x] All AUROC comparisons cite paired Nadeau-Bengio corrected $p$-values and intervals.
- [x] Parameter counts state 47,221 free and 4,249,284 dense weights.
- [x] Attributions are described as size-dominated and reflective of generic translation/ribosome/splicing pathways, not novel disease biology.
- [x] Model outputs in the GUI are labeled "Not Calibrated".
- [x] Seed sensitivity reports cross-validated variability ($\text{SD} = 0.1210$, differences below ~0.05 not interpretable) alongside the in-sample memorisation explanation.
- [x] Clinical disclaimer is prominent on all interfaces and documentation.
- [x] No claims of superiority over linear baselines (mRNA ElasticNet 0.9169, Early-Fusion ElasticNet 0.9295) or clinical features (EHR2 0.8506).
- [x] No claims of prognostic prediction, time-to-event survival modeling, or relapse forecasting.
- [x] No claims of imaging, radiology, or histopathological image processing.
- [x] Clear acknowledgement that the pathway prior was not shown to help over a random prior ($p = 0.86$) and the model lacks external validation.

