# Clinical Feature Leakage Scan Report

Generated: 2026-10-03 06:55:03 UTC
Cohort: GDC TARGET-WT Core ($N = 121$ Patients)

## 1. Top 10 Clinical Features Associated with Primary Label: Histologic Subtype

| Rank | Clinical Column | Metric | Association Score | Missing (%) | Unique Values | Leakage Status |
|---|---|---|---|---|---|---|
| 1 | `days_to_birth.demographic` | AUROC | **0.6488** | 0.0% | 117 | `CLEAN` |
| 2 | `age_at_diagnosis.diagnoses` | AUROC | **0.6488** | 0.0% | 117 | `CLEAN` |
| 3 | `age_at_earliest_diagnosis.diagnoses.xena_derived` | AUROC | **0.6488** | 0.0% | 117 | `CLEAN` |
| 4 | `age_at_earliest_diagnosis_in_years.diagnoses.xena_derived` | AUROC | **0.6488** | 0.0% | 117 | `CLEAN` |
| 5 | `age_at_index.demographic` | AUROC | **0.6418** | 0.0% | 14 | `CLEAN` |
| 6 | `year_of_diagnosis.diagnoses` | AUROC | **0.5464** | 0.0% | 9 | `CLEAN` |
| 7 | `sample_type_id.samples` | Cramer's V | **0.0622** | 0.0% | 3 | `CLEAN` |
| 8 | `tumor_descriptor.samples` | Cramer's V | **0.0622** | 0.0% | 3 | `CLEAN` |
| 9 | `sample_type.samples` | Cramer's V | **0.0622** | 0.0% | 3 | `CLEAN` |
| 10 | `gender.demographic` | Cramer's V | **0.0** | 0.0% | 2 | `CLEAN` |

## 2. Top 10 Clinical Features Associated with Secondary Label: Vital Status

| Rank | Clinical Column | Metric | Association Score | Missing (%) | Unique Values | Leakage Status |
|---|---|---|---|---|---|---|
| 1 | `age_at_index.demographic` | AUROC | **0.5375** | 0.0% | 14 | `CLEAN` |
| 2 | `days_to_birth.demographic` | AUROC | **0.5356** | 0.0% | 117 | `CLEAN` |
| 3 | `age_at_earliest_diagnosis.diagnoses.xena_derived` | AUROC | **0.5356** | 0.0% | 117 | `CLEAN` |
| 4 | `age_at_earliest_diagnosis_in_years.diagnoses.xena_derived` | AUROC | **0.5356** | 0.0% | 117 | `CLEAN` |
| 5 | `age_at_diagnosis.diagnoses` | AUROC | **0.5356** | 0.0% | 117 | `CLEAN` |
| 6 | `year_of_diagnosis.diagnoses` | AUROC | **0.5046** | 0.0% | 9 | `CLEAN` |
| 7 | `pediatric_kidney_staging.diagnoses` | Cramer's V | **0.3811** | 10.7% | 4 | `CLEAN` |
| 8 | `sample_type_id.samples` | Cramer's V | **0.1564** | 0.0% | 3 | `CLEAN` |
| 9 | `sample_type.samples` | Cramer's V | **0.1564** | 0.0% | 3 | `CLEAN` |
| 10 | `tumor_descriptor.samples` | Cramer's V | **0.1564** | 0.0% | 3 | `CLEAN` |

## 3. Permanently Blacklisted / Excluded Variables
The following variables are permanently blacklisted to guarantee 0% data leakage:
- `case_id`
- `cause_of_death.demographic`
- `created_datetime.treatments.diagnoses`
- `days_to_death.demographic`
- `id`
- `patient_id`
- `pid`
- `protocol_identifier.treatments.diagnoses`
- `sample`
- `sample_id.samples`
- `state.treatments.diagnoses`
- `submitter_id`
- `submitter_id.treatments.diagnoses`
- `treatment_id.treatments.diagnoses`
- `treatment_or_therapy.treatments.diagnoses`
- `updated_datetime.treatments.diagnoses`
- `vital_status.demographic`
- `wilms_tumor_histologic_subtype.diagnoses`
