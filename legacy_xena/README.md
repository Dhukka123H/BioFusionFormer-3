# legacy_xena

The omics matrices used by the frozen E1 run (`data/processed/mrna_gene.parquet`, `mirna.parquet`,
`mirna_unique.parquet`, `meth_gene.parquet`) were built from the **UCSC Xena GDC hub** (GDC-derived data,
hg38, transformed by Xena).


This folder keeps the two Xena-specific artefacts so the lineage stays reproducible:

* `phase_xena_pipeline.py` - downloads the Xena files and builds `cohort_patients.csv` (paths assume it is run from
  `scripts/`; move it back or fix `ROOT` before running).
* `xena_fallback_report.md` - the feasibility check that justified using Xena.

`scripts/phase4_feature_construction.py` still reads the raw Xena files (`data/raw/target_wt_xena/`) to build the
matrices. A GDC-native rebuild would replace that step and requires re-running E1 (numbers will change).
