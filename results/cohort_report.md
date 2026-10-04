# TARGET-WT Cohort Report
Generated: 2026-10-04T15:56:58.223973+00:00
Project: TARGET-WT  |  Access: open

## 1. Files per Modality (from GDC API)

| Modality        | File count (API) |
|-----------------|-----------------|
| mRNA             |             136 |
| miRNA            |             138 |
| Methylation      |             131 |

## 2. Patients per Modality

| Group                              | N patients |
|------------------------------------|-----------|
| mRNA                               |       125 |
| miRNA                              |       127 |
| Methylation                        |       126 |
| Clinical                           |       652 |
| Survival data                      |       651 |

## 3. Pairwise and Full Intersections

| Intersection                       | N patients |
|------------------------------------|-----------|
| mRNA ∩ miRNA                       |       124 |
| mRNA ∩ Methylation                 |       123 |
| miRNA ∩ Methylation                |       125 |
| mRNA ∩ miRNA ∩ Methylation         |       122 |
| + Clinical (full cohort)           |       122 |
| + Survival (vital + time)          |       122 |

## 4. Candidate Classification Labels

  vital_status    : Alive:72; Dead:50  [missing 0%]
  tumor_stage     :   [missing 100%]
  tumor_grade     :   [missing 100%]
  morphology      : 8960/3:122  [missing 0%]
  progression     :   [missing 100%]
  prior_malignancy:   [missing 100%]

## 5. Sample Type Breakdown

| Sample Type                        | Count |
|------------------------------------|-------|
| Primary Tumor                      |   187 |
| Blood Derived Normal               |    73 |
| Solid Tissue Normal                |    57 |
| Recurrent Tumor                    |     5 |
| Metastatic                         |     1 |

Patients with >1 sample per modality: 126
(Strategy: prefer 'Primary Tumor' sample; flag multi-sample patients for sensitivity analysis.)

## 6. Estimated Download Size (Full Cohort Files Only)

  ~2.3 GB  (sum of file_size for 122 patients across 3 modalities)

## 7. Recommendation

N is marginal (~100-200). A survival task is defensible; classification may be under-powered unless classes are balanced. Consider cross-validation with careful stratification.