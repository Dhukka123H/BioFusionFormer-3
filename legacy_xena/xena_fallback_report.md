# UCSC Xena GDC Hub Fallback Feasibility Report: TARGET-WT Cohort

**Date Generated**: 2026-10-03  
**Status**: Verified Live via UCSC Xena API (`xenaPython`) & HTTP HEAD inspection  
**Cohort**: `GDC TARGET-WT` (Wilms' Tumor / High-Risk Pediatric Renal Tumors)  
**Host**: `https://gdc.xenahubs.net`  
**Browser Cohort Hub**: [Xena Browser GDC TARGET-WT Datapages](https://xenabrowser.net/datapages/?cohort=GDC%20TARGET-WT)  

---

## 1. Executive Summary & Feasibility Verdict
- **Feasibility**: **HIGHLY FEASIBLE (READY-TO-USE FALLBACK)**.
- The UCSC Xena GDC Hub contains pre-processed, harmonized, open-access Level 3/4 data for the exact target cohort (`TARGET-WT`).
- **Zero Controlled-Access Roadblocks**: All files hosted on the Xena GDC Hub are public, open-access, and do not require dbGaP authorization or GDC token authentication.
- **Multi-Modal Patient Overlap**: A complete **3-way multi-modal cohort of 122 patients** exists with simultaneous measurements across **mRNA (STAR counts), miRNA (RPM), and DNA methylation (HM450K)**. All 122 patients possess clinical phenotype annotations, and 50 patients possess standardized overall survival (`OS`, `OS.time`) endpoints.

---

## 2. Dataset Inventory & Verification

All dataset identifiers, metadata, sample sizes, and URLs were verified directly through `xenaPython` and HTTP queries against `https://gdc.xenahubs.net`. No full data matrices were downloaded.

| Modality / Layer | Dataset Identifier | Data Subtype & Platform | Units | Sample Count | Patient Count | File Size (Bytes / MB) | Verified Download URL |
|---|---|---|---|---|---|---|---|
| **Gene Expression (Counts)** | `TARGET-WT.star_counts.tsv` | RNA-seq (Illumina) | Raw read counts | 136 | 125 | 18,837,662 B (~18.84 MB) | [Download URL](https://gdc.xenahubs.net/download/TARGET-WT.star_counts.tsv.gz) |
| **Gene Expression (Normalized)** | `TARGET-WT.star_tpm.tsv` / `TARGET-WT.star_fpkm.tsv` | RNA-seq (Illumina) | `log2(tpm+1)` / `log2(fpkm+1)` | 136 | 125 | ~19.5 MB | [Download URL](https://gdc.xenahubs.net/download/TARGET-WT.star_tpm.tsv.gz) |
| **miRNA Expression** | `TARGET-WT.mirna.tsv` | Stem loop expression (Illumina) | `log2(RPM+1)` | 138 | 127 | 657,145 B (~657 KB) | [Download URL](https://gdc.xenahubs.net/download/TARGET-WT.mirna.tsv.gz) |
| **DNA Methylation 450K** | `TARGET-WT.methylation450.tsv` | Illumina Infinium HM450 | Beta value $[0, 1]$ | 131 | 126 | 452,508,293 B (~452.51 MB) | [Download URL](https://gdc.xenahubs.net/download/TARGET-WT.methylation450.tsv.gz) |
| **Phenotype / Clinical** | `TARGET-WT.clinical.tsv` | Clinical matrix (52 features) | Structured metadata | 139 | 128 | 18,342 B (~18.34 KB) | [Download URL](https://gdc.xenahubs.net/download/TARGET-WT.clinical.tsv.gz) |
| **Survival Outcomes** | `TARGET-WT.survival.tsv` | Curated survival endpoints | `OS` (0/1), `OS.time` (days) | 56 | 51 | 623 B (~0.61 KB) | [Download URL](https://gdc.xenahubs.net/download/TARGET-WT.survival.tsv.gz) |

> **Note on Storage URLs**: Direct requests to `https://gdc.xenahubs.net/download/<dataset>.tsv.gz` issue an HTTP 302 redirect to UCSC Xena's AWS S3 mirror (`https://gdc-hub.s3.dualstack.us-east-1.amazonaws.com/download/<dataset>.tsv.gz`), which supports high-speed resumable parallel downloads.

---

## 3. Patient Cohort Overlap Analysis

Sample identifiers adhere to the standard GDC barcode schema: `TARGET-50-XXXXXX-YYZ` (e.g., `TARGET-50-PAJNAV-01A`), where characters 1–16 identify the unique patient participant (`TARGET-50-XXXXXX`), and characters 17–19 designate sample type (`01A` = primary solid tumor).

### Cohort Intersections
```
                  ┌─────────────────────────────────────────┐
                  │          TARGET-WT Clinical Cohort      │
                  │                (128 Patients)           │
                  └────────────────────┬────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   mRNA RNA-seq               miRNA Expression              DNA Methylation 450K
   (125 Patients)             (127 Patients)                   (126 Patients)
         │                             │                             │
         └──────────────────────┬──────┴─────────────────────────────┘
                                │
                                ▼
               ┌─────────────────────────────────┐
               │    3-Way Multi-Modal Overlap    │
               │   mRNA ∩ miRNA ∩ Methylation    │
               │         122 Patients            │
               └────────────────┬────────────────┘
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
   3-Way + Full Clinical                  3-Way + Survival
        122 Patients                         50 Patients
```

### Numerical Breakdown

| Subset Combination | Patient Count | Notes |
|---|---|---|
| **mRNA ∩ miRNA** | **124** | 99.2% of mRNA patients have miRNA data |
| **mRNA ∩ Methylation 450K** | **123** | 98.4% of mRNA patients have methylation data |
| **miRNA ∩ Methylation 450K** | **125** | 98.4% of miRNA patients have methylation data |
| **3-Way Multi-Modal (mRNA ∩ miRNA ∩ Meth450)** | **122** | **Core multi-omics cohort for BioFusionFormer** |
| **3-Way Multi-Modal ∩ Clinical** | **122** | 100% complete clinical metadata coverage |
| **3-Way Multi-Modal ∩ Survival (`OS`, `OS.time`)** | **50** | Validated clinical survival endpoint cohort |
| **Complete 5-way Matrix Intersect** | **50** | Simultaneous mRNA, miRNA, Methylation, Clinical, and Survival |

---

## 4. Feature Space & Probe Map Compatibility

1. **Gene Expression (`TARGET-WT.star_counts.tsv`)**:
   - Mapped via `gencode.v36.annotation.gtf.gene.probemap` to Ensembl Gene IDs (e.g., `ENSG00000223972.5`).
   - Suffix stripping maps 100% of standard protein-coding genes to our Reactome gene-pathway mask (`gene_pathway_mask.npz`).
2. **miRNA Expression (`TARGET-WT.mirna.tsv`)**:
   - Reported as mature/stem-loop identifiers in `log2(RPM+1)` matching miRBase naming conventions.
   - Ready for integration with `mirna_pathway_mask.npz`.
3. **DNA Methylation (`TARGET-WT.methylation450.tsv`)**:
   - Mapped via `HM450.hg38.manifest.gencode.v36.probeMap` with probe IDs (`cg*`) matching our `cpg_gene_mask.npz`.
   - Continuous beta values $[0, 1]$ directly correspond to standard methylation beta values.
4. **Clinical Features (`TARGET-WT.clinical.tsv`)**:
   - Contains 52 features including `age_at_diagnosis`, `gender`, `ethnicity`, `disease_type`, `tumor_stage`, and survival days.

---

## 5. Comparison: GDC Native API vs UCSC Xena GDC Hub

| Dimension | GDC Native API (`api.gdc.cancer.gov`) | UCSC Xena GDC Hub (`gdc.xenahubs.net`) |
|---|---|---|
| **Current Availability** | 🔴 **DOWN (503 Maintenance)** | 🟢 **ACTIVE & ONLINE (Fast CDN/S3)** |
| **Format** | Individual tar.gz/sample files requiring custom aggregation | Pre-concatenated, sample-aligned tabular matrices |
| **Data Normalization** | Raw counts + FPKM/TPM separately per file | Consolidated matrix with uniform log2 transformation |
| **Total Download Volume** | ~5–10 GB across thousands of individual file requests | **~472 MB** total across 5 compressed archives |
| **BioFusionFormer Readiness** | Requires Phase B metadata query + Phase C downloader | Immediately ready if user triggers Phase D fallback |

---

## 6. Recommendation
Keep the `BioFusionFormer_GDC_Poller` running to await GDC restoration. However, if GDC maintenance extends or if rapid experimental progress is desired, the **UCSC Xena GDC Hub provides a 100% viable, open-access, pre-harmonized drop-in replacement** with 122 complete multi-modal patients.
