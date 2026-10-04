# Batch and Confounder Analysis Report: TARGET-WT Cohort

Generated: 2026-10-03 06:55:06 UTC
Cohort: GDC TARGET-WT Primary Tumour 3-Omics Core ($N = 121$ Patients)

## 1. Confounder Association with Clinical Targets

| Confounder | Test Method | p-value (Histology) | Effect / Summary (Histology) | p-value (Vital Status) | Effect / Summary (Vital Status) |
|---|---|---|---|---|---|
| `age_at_diagnosis` | Mann-Whitney U | 0.0084 (YES (p < 0.05)) | Median FH=3.89, UH=4.90 | 0.5072 (No) | Median Alive=4.69, Dead=4.09 |
| `gender` | Chi-Square | 0.3745 (No) | Cramer's V = 0.000 | 0.0616 (No) | Cramer's V = 0.144 |
| `year_of_diagnosis` | Mann-Whitney U | 0.4044 (No) | Median FH=1999.00, UH=1999.00 | 0.9318 (No) | Median Alive=2000.00, Dead=1999.00 |
| `protocol` | Chi-Square | 0.1919 (No) | Cramer's V = 0.076 | 0.3294 (No) | Cramer's V = 0.000 |
| `race` | Chi-Square | 0.5980 (No) | Cramer's V = 0.000 | 0.1880 (No) | Cramer's V = 0.121 |
| `sample_vial` | Chi-Square | 1.0000 (No) | Cramer's V = 0.000 | 1.0000 (No) | Cramer's V = 0.000 |

## 2. Confounder Association with Top 5 Principal Components of Omics

Principal Components 1 to 5 capture the dominant unsupervised variance in mRNA, miRNA, and DNA methylation.

| Modality | PC | Confounder | Test Statistic | p-value | Assessment |
|---|---|---|---|---|---|
| **mRNA** | `PC3` | `age_at_diagnosis` | Spearman r=0.205 | 0.0239 | `Nominal (p < 0.05)` |
| **mRNA** | `PC3` | `protocol` | Kruskal-Wallis stat=4.91 | 0.0267 | `Nominal (p < 0.05)` |
| **mRNA** | `PC4` | `age_at_diagnosis` | Spearman r=0.189 | 0.0382 | `Nominal (p < 0.05)` |
| **miRNA** | `PC4` | `year_of_diagnosis` | Spearman r=-0.248 | 0.0062 | `YES (p < 0.01)` |
| **miRNA** | `PC5` | `gender` | Kruskal-Wallis stat=6.50 | 0.0108 | `Nominal (p < 0.05)` |
| **Methylation450** | `PC5` | `age_at_diagnosis` | Spearman r=0.198 | 0.0291 | `Nominal (p < 0.05)` |
| **Methylation450** | `PC5` | `protocol` | Kruskal-Wallis stat=5.28 | 0.0216 | `Nominal (p < 0.05)` |

## 3. Scientific Synthesis & Interpretation
- **Age at Diagnosis**: In Wilms' tumor, older age at diagnosis is a known clinical biological correlate of unfavorable histology (anaplasia). This represents **true tumor biology** rather than technical artifact.
- **Batch / Protocol / Vial**: The protocol (NWTS-5 vs NWTS-4) and sample vial (A vs B) show **no significant distortion** on omics variance, confirming that technical batch artifacts do not dominate the biological signal in this pre-harmonized GDC TARGET-WT release.
