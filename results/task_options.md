# BioFusionFormer - Clinical Prediction Task Options & Statistical Power Report

**Date**: 2026-10-03  
**Cohort**: `GDC TARGET-WT` Primary Tumour 3-Omics Multi-Modal Core ($N = 121$ Patients)  
**Modalities**: mRNA (STAR counts) + miRNA (RPM) + DNA Methylation (HM450K) + Clinical Metadata  
**Audit Source**: Exhaustive verification across all 53 features in `TARGET-WT.clinical.tsv.gz` and `TARGET-WT.survival.tsv.gz`.

---

## 1. Executive Summary & Recommended Path Forward

After exhaustive auditing of all 53 clinical features and survival records across the 121 core multi-modal Wilms' tumor patients, here are the evidence-based task recommendations:

### Primary Task Recommendation (Top Recommendation: Strongest Biological & Clinical Defensibility)
- **Task Option 1**: **Wilms' Tumor Histologic Subtype Classification** (`wilms_tumor_histologic_subtype.diagnoses`)
  - **Classes**: **Favorable Histology (FH)** ($N = 82$, 67.8%) vs **Unfavorable Histology (UH / Diffuse Anaplasia)** ($N = 39$, 32.2%).
  - **Missingness**: **0.0%** (121 / 121 patients fully annotated).
  - **Clinical & Biological Defensibility**:
    - In pediatric renal oncology (Children's Oncology Group / SIOP protocols), histological subtype is the single most critical prognostic biomarker for Wilms' tumor.
    - Unfavorable histology (diffuse anaplasia) accounts for the vast majority of treatment failures, drug resistance, and deaths.
    - Because histology is directly driven by underlying biological and molecular aberrations (e.g. *TP53* mutations, chromosomal 1q gains, genome-wide DNA hypermethylation, and microRNA deregulation), this task directly tests whether **BioFusionFormer's pathway-aware multi-modal fusion** captures coordinated multi-omics mechanisms distinguishing aggressive anaplasia from favorable histology.
  - **Statistical Adequacy**: With **39 minority cases** ($\ge 20$ requirement easily met), this provides exceptional statistical stability for stratified 5-fold cross-validation (~8 minority cases per validation fold).
  - **Evaluation Metrics**: **AUROC**, **AUPRC** (Area Under Precision-Recall Curve, baseline prevalence = 0.322), **Balanced Accuracy**, and **F1-score**.

### Co-Primary / Alternative Task Recommendation (Clinical Outcome Prediction)
- **Task Option 2**: **Vital Status / Overall Survival Status Classification** (`vital_status.demographic`)
  - **Classes**: **Alive** ($N = 71$, 58.7%) vs **Dead** ($N = 50$, 41.3%).
  - **Missingness**: **0.0%** (121 / 121 patients fully annotated).
  - **Clinical Relevance**: In TARGET-WT (which intentionally enriched for high-risk Wilms' tumor cases), 50 out of 121 patients suffered mortality. Predicting patient survival/mortality from baseline multi-omics directly addresses therapy stratification.
  - **Statistical Adequacy**: Exceptional class balance ($50:71$, 41.3% mortality rate). With 50 events, statistical power is very high for binary classification (~10 events per fold in 5-fold CV).
  - **Evaluation Metrics**: **AUROC**, **AUPRC**, **Balanced Accuracy**, and **F1-score**.

### Secondary / Exploratory Task Options
- **Task Option 3**: **Time-to-Event Survival Analysis (Cox Elastic Net / DeepSurv)**
  - **Caveat & Honest Limitations**:
    - `TARGET-WT.survival.tsv` and `TARGET-WT.clinical.tsv` record exact `days_to_death` for all 50 deceased patients (median time to death = 564 days / 1.54 years).
    - However, follow-up times for the 71 alive patients are not populated in the Xena clinical matrix (censored follow-up times are missing).
    - Therefore, while binary classification of Vital Status (Option 2) is 100% complete and defensible, Cox regression requires imputation or artificial right-censoring assumptions for alive patients.
- **Task Option 4**: **Biological Sex / Gender Prediction** (`gender.demographic`, Female: 69, Male: 52)
  - Serves as an unsupervised / sanity-check control task for multi-modal model training (e.g. verifying that sex-chromosome probes/genes behave as expected).

---

## 2. Complete Candidate Target Screening Table

Every candidate clinical target in the TARGET-WT cohort was screened against strict criteria:
- **`usable-for-classification`**: Minority class $\ge 20$ patients, $< 10\%$ missing rate.
- **`usable-for-survival`**: $\ge 30$ events ideal; low-power flagged if $< 30$.
- **`not usable`**: Degenerate / single class, $< 20$ minority events, or $> 10\%$ missing.
- **Label Leakage**: Post-diagnostic treatments, death dates, or follow-up duration variables must never be included as input features.

| Target Variable | Clinical Meaning | Target Type | Missing (%) | Class Distribution (Counts) | Usability Status | Technical Justification & Leakage Warnings |
|---|---|---|---|---|---|---|
| **`wilms_tumor_histologic_subtype`** | Histological Subtype (FH vs UH) | Classification | **0.0%** | Favorable: 82 (67.8%), Unfavorable: 39 (32.2%) | **USABLE (RECOMMENDED PRIMARY)** | Minority class = 39 ($\ge 20$), 0% missing. Top biological marker in Wilms' tumor. |
| **`vital_status.demographic`** | Patient Mortality Status | Classification | **0.0%** | Alive: 71 (58.7%), Dead: 50 (41.3%) | **USABLE (RECOMMENDED CO-PRIMARY)** | Near-optimal balance ($50:71$). Clinically critical outcome. |
| **`gender.demographic`** | Biological Sex | Classification | **0.0%** | Female: 69 (57.0%), Male: 52 (43.0%) | **USABLE (CONTROL TASK)** | Balanced demographic baseline ($52:69$). |
| **`pediatric_kidney_staging`** | Surgical / COG Stage | Classification | **10.7%** | Margin/LN neg: 49, Margin/LN pos: 40, Confined: 15, Biopsy: 4, Missing: 13 | **BORDERLINE / NOT USABLE** | Missing rate exceeds 10% threshold (10.7%). Multi-class imbalance. |
| **`OS` (survival.tsv)** | Overall Survival Event & Time | Survival | 58.7% | Dead: 50, Alive: 0 (in survival.tsv) | **USABLE WITH CAUTION** | Survival.tsv records all 50 deaths, but follow-up time for alive patients is absent. |
| **`race.demographic`** | Patient Race | Classification | **0.0%** | White: 90, Black: 18, Other: 5, Not reported: 8 | **NOT USABLE** | Minority class (Black) has only 18 patients (< 20 threshold). Severe demographic skew. |
| **`ethnicity.demographic`** | Patient Ethnicity | Classification | **0.0%** | Not Hispanic: 80, Hispanic: 11, Not reported: 30 | **NOT USABLE** | Hispanic minority = 11 (< 20 threshold); 24.8% not reported. |
| **`primary_diagnosis`** | Primary Diagnosis | Classification | **0.0%** | Wilms tumor: 121 (100.0%) | **NOT USABLE** | Degenerate single class. |

---

## 3. Honest Statistical Power & Precision Analysis

In high-dimensional multi-modal bioinformatics ($N = 121$, $P > 30,000$ across mRNA genes, miRNAs, and CpGs), rigorous evaluation is essential to avoid reporting overfitted artifacts.

### A. Classification Confidence Interval Width (Hanley-McNeil Formulation)
- For the primary task (**Histologic Subtype**, $N = 121$, $n_1 = 39$ Unfavorable, $n_2 = 82$ Favorable):
  - For a baseline model achieving an **AUROC of 0.75**, the asymptotic standard error (SE) is:
    $$\text{SE}(\text{AUC}) \approx \sqrt{\frac{\text{AUC}(1-\text{AUC}) + (n_1 - 1)(Q_1 - \text{AUC}^2) + (n_2 - 1)(Q_2 - \text{AUC}^2)}{n_1 n_2}} \approx 0.046$$
  - The **95% Confidence Interval** for a true AUC of 0.75 is:
    $$0.75 \pm 1.96 \times 0.046 = [0.660, \; 0.840] \quad (\text{CI width} \approx 0.18)$$
  - **Implication**: Any model claiming an improvement from 0.74 to 0.77 without cross-validation fold variance is statistically indistinguishable.
  - To ensure scientific validity, our Phase 5 baseline protocol utilizes **repeated stratified 5-fold cross-validation (5 repeats = 25 validation folds)** with fixed random seeds.

- For the co-primary task (**Vital Status**, $N = 121$, $n_1 = 50$ Dead, $n_2 = 71$ Alive):
  - With $n_1 = 50$, class balance is even stronger, yielding $\text{SE}(\text{AUC}) \approx 0.043$ and a 95% CI of $[0.666, \; 0.834]$ ($\text{CI width} \approx 0.168$).

### B. Precision-Recall Curve (AUPRC) Necessity
- For Histologic Subtype, positive class prevalence is 32.2%. A random dummy classifier achieves an AUPRC of **0.322**.
- For Vital Status, positive class prevalence is 41.3%. A random dummy classifier achieves an AUPRC of **0.413**.
- Reporting AUPRC alongside AUROC is mandatory because AUPRC penalizes false positives much more strictly under class imbalance.

---

## 4. Label Leakage Prevention Protocol

To ensure valid pair programming and machine learning integrity, the following features are permanently blacklisted from all model input matrices:
1. `days_to_death.demographic` (Directly discloses patient death)
2. `vital_status.demographic` (Outcome label for survival classification)
3. `cause_of_death.demographic` (Direct outcome indicator)
4. `updated_datetime.treatments.diagnoses`, `created_datetime.treatments.diagnoses` (Post-diagnosis treatment timestamps)
5. `treatment_or_therapy.treatments.diagnoses` (Second-line / salvage therapies administered after relapse)

Only baseline transcriptomic (mRNA, miRNA), epigenomic (DNA methylation), and pre-treatment diagnostic covariates will be provided to the feature extractors.
