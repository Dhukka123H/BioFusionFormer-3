# TARGET-WT Clinical Data & EHR Feature Audit

**Generated:** 2026-10-05T06:37:17.832363+00:00 UTC  
**GDC API Data Release:** Data Release 46.0 - August 10, 2026  
**GDC API Commit:** `8f7c2a51ab0084b216ad1b62a3fae8b945439c53`  
**Cohort Size:** 121 Primary Solid Tumor (-01A) Patients Aligned to 3-Omics Core  

---

## 1. Executive Summary

- **Primary Data Source:** GDC `/cases` API endpoint (Open-Access, No Token Required).
- **Clinical Supplements:** GDC `/files` and `/data` open-access clinical supplements:
  1. `TARGET_WT_ClinicalData_Discovery_20230322.xlsx` (Discovery cohort metadata)
  2. `TARGET_WT_ClinicalData_Validation_20230322.xlsx` (Validation cohort metadata)
  3. `TARGET_WT_ClinicalData_Discovery_and_Validation_Percent_Tumor_Nuclei_and_Necrosis_Supplement_20230322.xlsx` (Biopsy composition metrics)
  4. `TARGET_WT_CDE_20230322.xlsx` (Common Data Elements dictionary)
- **Fallback Systems:** Fallback A (UCSC Xena) and Fallback B (TARGET Data Matrix) were **not required** because the primary GDC endpoints operated with 100% availability.
- **Cohort Matching:** Exactly **121 of 121 (100.0%)** core primary-tumour patients matched across GDC cases, Discovery/Validation sheets, and Tumor Nuclei sheets. **0 missing IDs, 0 duplicate rows.**
- **Cross-Check with Existing Clinical Table:** 100% exact agreement observed for Gender (69 female, 52 male) and Age at Diagnosis in days against the UCSC Xena baseline clinical table.
- **Laterality Status:** Laterality was extensively searched across GDC cases schema (`diagnoses.laterality`, `samples.biospecimen_laterality`), supplement tables, and UCSC Xena. In the public TARGET-WT open-access dataset, laterality is unrecorded (100% missing / not collected).

---

## 2. Kept Baseline Candidate Features (Pre-Modeling)

These features represent valid pre-treatment baseline patient characteristics. No data leakage is present, and values remain unscaled and unimputed (imputation and standardization must be applied strictly within CV folds).

| Feature Name | Source Column | Type | % Missing | Unique | Example Values |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `age_at_diagnosis_days` | `supp_Age at Diagnosis in Days` | continuous | 0.0% | 117 | 1757, 2244, 1710 |
| `gender` | `supp_Gender` | categorical_binary | 0.0% | 2 | Female, Male |
| `race` | `supp_Race` | categorical | 0.0% | 4 | White, Black or African American, Not Reported |
| `ethnicity` | `supp_Ethnicity` | categorical | 0.0% | 3 | Not reported, Not Hispanic or Latino, Hispanic or Latino |
| `stage` | `supp_Stage` | categorical_ordinal | 0.0% | 5 | III, IV, II |
| `percent_tumor_nuclei` | `nuclei_% Tumor Nuclei: Top` | continuous_percentage | 0.8% | 8 | 90.0, 98.0, 95.0 |
| `percent_necrosis` | `nuclei_% Necrosis: Top` | continuous_percentage | 0.8% | 6 | 0.0, 20.0, 5.0 |
| `percent_stroma` | `nuclei_% Stroma: Top` | continuous_percentage | 0.8% | 15 | 60.0, 8.0, 5.0 |

### Baseline Candidate Distributions

#### `age_at_diagnosis_days`
- Min: 156.0, Median: 1546.0, Mean: 1703.1, Max: 5698.0, Std: 1031.0

#### `gender`
- Counts: **Female**: 69, **Male**: 52

#### `race`
- Counts: **White**: 90, **Black or African American**: 18, **Not Reported**: 8, **Other**: 5

#### `ethnicity`
- Counts: **Not Hispanic or Latino**: 80, **Not reported**: 30, **Hispanic or Latino**: 11

#### `stage`
- Counts: **II**: 49, **III**: 40, **I**: 15, **IV**: 13, **IIIB**: 4

#### `percent_tumor_nuclei`
- Counts: **98.0**: 50, **95.0**: 48, **90.0**: 10, **99.0**: 6, **85.0**: 3, **88.0**: 1, **nan**: 1, **93.0**: 1, **75.0**: 1

#### `percent_necrosis`
- Counts: **0.0**: 109, **5.0**: 5, **1.0**: 3, **20.0**: 1, **2.0**: 1, **nan**: 1, **3.0**: 1

#### `percent_stroma`
- Min: 0.0, Median: 2.0, Mean: 5.9, Max: 60.0, Std: 10.4

#### `stage_normalized` (Clean Roman Numerals)
- Counts: {'Stage II': 49, 'Stage III': 44, 'Stage I': 15, 'Stage IV': 13}

---

## 3. Data Leakage & Feature Exclusion Audit

To strictly prevent data leakage with respect to the primary classification task (**Histological Subtype: Favorable [0] vs Unfavorable / Anaplasia [1]**), features were audited against 4 exclusion categories:

1. **Histology / Target Leakage:** Any column defining tumor histology, anaplasia subtype, morphology codes, or COG risk stage that correlates directly with unfavorable histology.
2. **Outcome / Survival Leakage:** Follow-up status, vital status, days to death, survival times, recurrence, or adjuvant therapies occurring after baseline.
3. **Administrative / IDs:** Sample barcodes, UUIDs, system timestamps, and data submission metadata.
4. **Low Variance / High Missingness / Redundant Biopsy:** Columns with zero variance (single constant value), >50% missing values, or collinear redundant biopsy metrics (e.g. bottom section or complements).

### Comprehensive Column Audit Table

| Column | Dtype | % Missing | Unique | Status | Exclusion Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `gdc_case_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_submitter_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_project_id` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_demographic.demographic_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.race` | `str` | 0.0% | 4 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.ethnicity` | `str` | 0.0% | 3 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.vital_status` | `str` | 0.0% | 2 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_demographic.updated_datetime` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_demographic.age_is_obfuscated` | `bool` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_demographic.age_at_index` | `float64` | 0.0% | 14 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.submitter_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.state` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_demographic.days_to_birth` | `float64` | 0.0% | 117 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.sex_at_birth` | `str` | 0.0% | 2 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_demographic.created_datetime` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.pediatric_kidney_staging` | `str` | 10.7% | 4 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.days_to_diagnosis` | `int64` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.created_datetime` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.last_known_disease_status` | `object` | 100.0% | 0 | ❌ EXCLUDE | HIGH MISSINGNESS: 100.0% missing (> 50% threshold) |
| `gdc_diagnoses.tissue_or_organ_of_origin` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.days_to_last_follow_up` | `float64` | 0.0% | 118 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_diagnoses.age_at_diagnosis` | `float64` | 0.0% | 117 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.primary_diagnosis` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.updated_datetime` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.year_of_diagnosis` | `float64` | 0.0% | 9 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.state` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.diagnosis_is_primary_disease` | `bool` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.days_to_last_known_disease_status` | `object` | 100.0% | 0 | ❌ EXCLUDE | HIGH MISSINGNESS: 100.0% missing (> 50% threshold) |
| `gdc_diagnoses.days_to_recurrence` | `object` | 100.0% | 0 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_diagnoses.morphology` | `str` | 0.0% | 1 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `gdc_diagnoses.submitter_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.wilms_tumor_histologic_subtype` | `str` | 0.0% | 2 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `gdc_diagnoses.classification_of_tumor` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.diagnosis_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `gdc_diagnoses.icd_10_code` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.cog_renal_stage` | `object` | 100.0% | 0 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `gdc_diagnoses.site_of_resection_or_biopsy` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `gdc_diagnoses.tumor_grade` | `object` | 100.0% | 0 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `gdc_diagnoses.progression_or_recurrence` | `object` | 100.0% | 0 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.timepoint_category` | `str` | 0.0% | 2 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.follow_up_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.updated_datetime` | `str` | 0.0% | 1 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.submitter_id` | `str` | 0.0% | 121 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.year_of_follow_up` | `float64` | 48.8% | 16 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.state` | `str` | 0.0% | 1 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.days_to_follow_up` | `float64` | 0.0% | 113 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.created_datetime` | `str` | 0.0% | 121 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_demographic.cause_of_death` | `str` | 58.7% | 3 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_demographic.days_to_death` | `float64` | 58.7% | 49 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.days_to_first_event` | `float64` | 51.2% | 53 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `gdc_follow_ups.first_event` | `str` | 51.2% | 3 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `supp_TARGET USI` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `supp_Gender` | `str` | 0.0% | 2 | ✅ KEEP | Clean baseline feature |
| `supp_Race` | `str` | 0.0% | 4 | ✅ KEEP | Clean baseline feature |
| `supp_Ethnicity` | `str` | 0.0% | 3 | ✅ KEEP | Clean baseline feature |
| `supp_Age at Diagnosis in Days` | `int64` | 0.0% | 117 | ✅ KEEP | Clean baseline feature |
| `supp_First Event` | `str` | 20.7% | 2 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `supp_Event Free Survival Time in Days` | `int64` | 0.0% | 109 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `supp_Vital Status` | `str` | 0.0% | 2 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `supp_Overall Survival Time in Days` | `int64` | 0.0% | 118 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `supp_Year of Diagnosis` | `int64` | 0.0% | 9 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `supp_Year of Last Follow Up` | `int64` | 0.0% | 19 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `supp_Protocol` | `str` | 0.0% | 2 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `supp_Stage` | `str` | 0.0% | 5 | ✅ KEEP | Clean baseline feature |
| `supp_ICD-O-3-M` | `str` | 0.0% | 1 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `supp_ICD-O-3-T` | `str` | 0.0% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `supp_Histologic Classification of Primary Tumor` | `str` | 0.0% | 2 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `supp_Reason for Death` | `str` | 58.7% | 4 | ❌ EXCLUDE | LEAKAGE: Survival/Outcome/Follow-up (post-baseline event, not baseline clinical EHR) |
| `nuclei_TARGET USI` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `nuclei_TARGET - Sample ID` | `str` | 0.0% | 121 | ❌ EXCLUDE | ADMINISTRATIVE: Identifier, timestamp, study year, or constant/uninformative metadata |
| `nuclei_Histological Subtype` | `str` | 1.7% | 2 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `nuclei_Specimen Type` | `str` | 0.8% | 1 | ❌ EXCLUDE | LOW VARIANCE: Constant column (unique values = 1) |
| `nuclei_Specimen Sub type` | `str` | 100.0% | 0 | ❌ EXCLUDE | HIGH MISSINGNESS: 100.0% missing (> 50% threshold) |
| `nuclei_Adequate Tissue (Y/N)` | `str` | 100.0% | 0 | ❌ EXCLUDE | HIGH MISSINGNESS: 100.0% missing (> 50% threshold) |
| `nuclei_Pathology: Diagnosis Confirmed` | `str` | 1.7% | 2 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |
| `nuclei_% Tumor Nuclei: Top` | `float64` | 0.8% | 8 | ✅ KEEP | Clean baseline feature |
| `nuclei_% Non tumor Nuclei: Top` | `float64` | 0.8% | 8 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Cellular Tumor: Top` | `float64` | 0.8% | 13 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Normal: Top` | `float64` | 0.8% | 4 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Stroma: Top` | `float64` | 0.8% | 15 | ✅ KEEP | Clean baseline feature |
| `nuclei_% Necrosis: Top` | `float64` | 0.8% | 6 | ✅ KEEP | Clean baseline feature |
| `nuclei_% Tumor Nuclei: Bottom` | `float64` | 0.8% | 8 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Non tumor Nuclei: Bottom` | `float64` | 0.8% | 9 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Cellular Tumor: Bottom` | `float64` | 0.8% | 14 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Normal: Bottom` | `float64` | 0.8% | 4 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Stroma: Bottom` | `float64` | 0.8% | 13 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_% Necrosis: Bottom` | `float64` | 0.8% | 7 | ❌ EXCLUDE | REDUNDANT: Collinear biopsy metric (complement or duplicate of Top section) |
| `nuclei_Pathology Pass/Fail` | `str` | 0.8% | 2 | ❌ EXCLUDE | LEAKAGE: Histology/Subtype/Morphology/COG risk (correlates directly with target) |

---

## 4. Output Artifacts

- `data/raw/clinical/cases_mapping.json`: Full GDC cases schema mapping.
- `data/raw/clinical/cases_raw.json`: Complete JSON response from GDC `/cases` API.
- `data/raw/clinical/cases_flat.csv`: Flattened 1-row-per-case dataset across all 652 TARGET-WT cases.
- `data/raw/clinical/supplements/`: Downloaded open-access XLSX clinical discovery, validation, and pathology supplements.
- `data/raw/clinical/PROVENANCE.json`: Detailed cryptographic hashes, endpoints, and metadata.
- `data/processed/clinical/clinical_raw_121.csv`: All merged clinical columns for the 121 core patients in exact fold order.
- `data/processed/clinical/clinical_candidates.csv`: Cleaned baseline candidate features for the 121 core patients.
- `configs/clinical_features.yaml`: Declarative candidate feature configuration for model experimentation.
