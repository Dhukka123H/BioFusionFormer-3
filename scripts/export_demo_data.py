"""
scripts/export_demo_data.py
===========================
Exports 3 cohort patients as example CSVs for GUI and CLI testing.
Generates data/demo/README.md dynamically from the actual CSV patient IDs,
and synchronizes app/demo_data to be identical to data/demo.
"""

from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "demo"
APP_DEMO_DIR = ROOT / "app" / "demo_data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
APP_DEMO_DIR.mkdir(parents=True, exist_ok=True)

cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
demo_pats = core_pats[:3]

print(f"Exporting 3 demo patients: {demo_pats}")

mrna = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[demo_pats]
mirna = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet").loc[demo_pats]
meth = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[demo_pats]

# Write CSVs to data/demo
mrna.to_csv(OUT_DIR / "demo_mrna.csv", index=True, index_label="patient_id")
mirna.to_csv(OUT_DIR / "demo_mirna.csv", index=True, index_label="patient_id")
meth.to_csv(OUT_DIR / "demo_meth.csv", index=True, index_label="patient_id")

# Dynamically generate README.md using the actual patient IDs present in the CSVs
actual_pats = mrna.index.tolist()
formatted_pats = ", ".join(f"`{pid}`" for pid in actual_pats)

readme_text = f"""# Demo Data for BioFusionFormer

> **Notice:** These demo CSVs contain 3 training cohort patients ({formatted_pats}).
> Predictions here are NOT performance evidence and reflect in-sample inference only.
"""
(OUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")
print(f"Saved demo CSVs and dynamically generated README.md to {OUT_DIR}")

# Synchronize app/demo_data to be identical to data/demo
for fname in ["demo_mrna.csv", "demo_mirna.csv", "demo_meth.csv", "README.md"]:
    shutil.copy2(OUT_DIR / fname, APP_DEMO_DIR / fname)

print(f"Synchronized app/demo_data to be identical to {OUT_DIR}")
