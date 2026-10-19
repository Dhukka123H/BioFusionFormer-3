"""
scripts/export_demo_data.py
Exports 3 cohort patients as example CSVs for GUI and CLI testing.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
demo_pats = core_pats[:3]

print(f"Exporting 3 demo patients: {demo_pats}")

mrna = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[demo_pats]
mirna = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet").loc[demo_pats]
meth = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[demo_pats]

mrna.to_csv(OUT_DIR / "demo_mrna.csv", index=True, index_label="patient_id")
mirna.to_csv(OUT_DIR / "demo_mirna.csv", index=True, index_label="patient_id")
meth.to_csv(OUT_DIR / "demo_meth.csv", index=True, index_label="patient_id")

readme_text = """# Demo Data for BioFusionFormer

> **Notice:** These demo CSVs contain 3 training cohort patients (`TARGET-30-PAAPDU`, `TARGET-30-PAAPEB`, `TARGET-30-PAAPFE`).
> Predictions here are NOT performance evidence and reflect in-sample inference only.
"""
(OUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")
print(f"Saved demo CSVs to {OUT_DIR}")
