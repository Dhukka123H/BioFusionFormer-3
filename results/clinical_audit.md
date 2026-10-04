# Clinical Feature Audit: TARGET-WT 3-Omics Cohort (N = 122)

Generated: 2026-10-03 06:45:49 UTC

Total features audited: 53

| Feature Name | Data Type | Missing Count | Missing (%) | Unique Values | Distribution / Sample Values |
|---|---|---|---|---|---|
| `sample` | object | 0 | 0.0% | 121 | 121 unique values (e.g. TARGET-50-CAAAAQ-01A) |
| `id` | object | 0 | 0.0% | 121 | 121 unique values (e.g. 0b372c9b-2d5a-5483-8c96-7f829d962a99) |
| `disease_type` | object | 0 | 0.0% | 1 | Complex Mixed and Stromal Neoplasms: 121 |
| `case_id` | object | 0 | 0.0% | 121 | 121 unique values (e.g. 0b372c9b-2d5a-5483-8c96-7f829d962a99) |
| `primary_site` | object | 0 | 0.0% | 1 | Kidney: 121 |
| `submitter_id` | object | 0 | 0.0% | 121 | 121 unique values (e.g. TARGET-50-CAAAAQ) |
| `race.demographic` | object | 0 | 0.0% | 4 | white: 90, black or african american: 18, not reported: 8, other: 5 |
| `gender.demographic` | object | 0 | 0.0% | 2 | female: 69, male: 52 |
| `ethnicity.demographic` | object | 0 | 0.0% | 3 | not hispanic or latino: 80, not reported: 30, hispanic or latino: 11 |
| `vital_status.demographic` | object | 0 | 0.0% | 2 | Alive: 71, Dead: 50 |
| `age_is_obfuscated.demographic` | bool | 0 | 0.0% | 1 | False: 121 |
| `age_at_index.demographic` | float64 | 0 | 0.0% | 14 | 14 unique values (e.g. 3.0) |
| `days_to_birth.demographic` | float64 | 0 | 0.0% | 117 | 117 unique values (e.g. -1287.0) |
| `primary_site.project` | object | 0 | 0.0% | 1 | Kidney: 121 |
| `project_id.project` | object | 0 | 0.0% | 1 | TARGET-WT: 121 |
| `disease_type.project` | object | 0 | 0.0% | 1 | Complex Mixed and Stromal Neoplasms: 121 |
| `name.project` | object | 0 | 0.0% | 1 | High-Risk Wilms Tumor: 121 |
| `name.program.project` | object | 0 | 0.0% | 1 | TARGET: 121 |
| `cause_of_death.demographic` | object | 71 | 58.7% | 3 | Cancer Related: 47, Infection: 2, Toxicity: 1 |
| `days_to_death.demographic` | float64 | 71 | 58.7% | 49 | 49 unique values (e.g. 702.0) |
| `pediatric_kidney_staging.diagnoses` | object | 13 | 10.7% | 4 | Nephrectomy specimen with tumor that penetrates the renal capsule or involves the renal sinus with negative margins and negative lymph nodes; no distant metastasis: 49, Nephrectomy specimen with tumor that is present at the surgical margin of resection or within regional lymph nodes; no distant metastasis: 40, Nephrectomy specimen with tumor confined to the kidney; no distant metastasis: 15, Tumor biopsied only at diagnosis; no distant metastasis: 4 |
| `days_to_diagnosis.diagnoses` | int64 | 0 | 0.0% | 1 | 0: 121 |
| `tissue_or_organ_of_origin.diagnoses` | object | 0 | 0.0% | 1 | Kidney, NOS: 121 |
| `age_at_diagnosis.diagnoses` | float64 | 0 | 0.0% | 117 | 117 unique values (e.g. 1287.0) |
| `primary_diagnosis.diagnoses` | object | 0 | 0.0% | 1 | Wilms tumor: 121 |
| `year_of_diagnosis.diagnoses` | float64 | 0 | 0.0% | 9 | 2001.0: 28, 2000.0: 25, 1998.0: 21, 1999.0: 17, 1996.0: 13, 1997.0: 10, 2002.0: 4, 1995.0: 2, 1993.0: 1 |
| `diagnosis_is_primary_disease.diagnoses` | bool | 0 | 0.0% | 1 | True: 121 |
| `morphology.diagnoses` | object | 0 | 0.0% | 1 | 8960/3: 121 |
| `wilms_tumor_histologic_subtype.diagnoses` | object | 0 | 0.0% | 2 | Favorable: 82, Unfavorable: 39 |
| `classification_of_tumor.diagnoses` | object | 0 | 0.0% | 1 | primary: 121 |
| `icd_10_code.diagnoses` | object | 0 | 0.0% | 1 | C64: 121 |
| `site_of_resection_or_biopsy.diagnoses` | object | 0 | 0.0% | 1 | Not Reported: 121 |
| `age_at_earliest_diagnosis.diagnoses.xena_derived` | float64 | 0 | 0.0% | 117 | 117 unique values (e.g. 1287.0) |
| `age_at_earliest_diagnosis_in_years.diagnoses.xena_derived` | float64 | 0 | 0.0% | 117 | 117 unique values (e.g. 3.526027397260274) |
| `protocol_identifier.treatments.diagnoses` | object | 0 | 0.0% | 2 | NWTS-5: 119, NWTS-4: 2 |
| `updated_datetime.treatments.diagnoses` | object | 0 | 0.0% | 1 | 2024-03-14T11:20:38.074146-05:00: 121 |
| `treatment_id.treatments.diagnoses` | object | 0 | 0.0% | 121 | 121 unique values (e.g. 9845e5f2-999d-4f87-93bd-9fb606ffc89f) |
| `submitter_id.treatments.diagnoses` | object | 0 | 0.0% | 121 | 121 unique values (e.g. TARGET-50-CAAAAQ_treatment) |
| `state.treatments.diagnoses` | object | 0 | 0.0% | 1 | released: 121 |
| `treatment_or_therapy.treatments.diagnoses` | object | 0 | 0.0% | 1 | yes: 121 |
| `created_datetime.treatments.diagnoses` | object | 0 | 0.0% | 121 | 121 unique values (e.g. 2023-03-28T12:36:14.710723-05:00) |
| `sample_type_id.samples` | int64 | 0 | 0.0% | 1 | 1: 121 |
| `tumor_descriptor.samples` | object | 0 | 0.0% | 1 | Primary: 121 |
| `sample_id.samples` | object | 0 | 0.0% | 121 | 121 unique values (e.g. e09b9e9e-8df9-580c-be3f-49c0d98324e9) |
| `freezing_method.samples` | float64 | 121 | 100.0% | 0 |  |
| `tumor_code_id.samples` | int64 | 0 | 0.0% | 1 | 50: 121 |
| `sample_type.samples` | object | 0 | 0.0% | 1 | Primary Tumor: 121 |
| `tumor_code.samples` | object | 0 | 0.0% | 1 | Wilms tumor (WT): 121 |
| `oct_embedded.samples` | float64 | 121 | 100.0% | 0 |  |
| `specimen_type.samples` | object | 0 | 0.0% | 1 | Solid Tissue: 121 |
| `tissue_type.samples` | object | 0 | 0.0% | 1 | Tumor: 121 |
| `preservation_method.samples` | object | 0 | 0.0% | 1 | Unknown: 121 |
| `patient_id` | object | 0 | 0.0% | 121 | 121 unique values (e.g. TARGET-50-CAAAAQ) |
