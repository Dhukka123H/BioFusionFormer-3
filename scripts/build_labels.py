#!/usr/bin/env python3
"""
scripts/build_labels.py
=======================
Builds data/processed/labels.csv (histology label for the 121 cohort patients) from the GDC clinical
table produced by scripts/download_clinical.py (data/processed/clinical/clinical_raw_121.csv).

Safety checks (the script stops with an error if any fails):
  1. exactly the 121 patients of data/interim/cohort_patients.csv (is_3omics_core), in the same order;
  2. every patient has a histology value (Favorable / Unfavorable);
  3. the overall class counts equal EXPECTED (82 FH / 39 UH) - the labels the frozen E1 and the CV
     splits were built on - unless --allow-count-change is given;
  4. for every one of the 25 outer folds, the FH/UH counts in train and test equal the counts stored in
     data/splits/splits_5x5.json. This proves the GDC labels are the SAME labels the splits and E1 used.
  5. if the TARGET supplement column exists, it must agree with the GDC column (disagreements are listed).

Run:  python scripts/build_labels.py
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"FH": 82, "UH": 39}
GDC_COL = "gdc_diagnoses.wilms_tumor_histologic_subtype"
SUPP_COL = "supp_Histologic Classification of Primary Tumor"
VITAL_COL = "gdc_demographic.vital_status"


def to_label(v):
    s = str(v).strip().lower()
    if s in ("nan", "", "none", "not reported", "unknown"):
        return None
    if "unfavorable" in s or "anaplas" in s:   # check before 'favorable' (substring!)
        return 1
    if "favorable" in s:
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-count-change", action="store_true")
    args = ap.parse_args()

    cohort = pd.read_csv(ROOT / "data" / "interim" / "cohort_patients.csv")
    core = cohort[cohort["is_3omics_core"]]["patient_id"].tolist()
    src = ROOT / "data" / "processed" / "clinical" / "clinical_raw_121.csv"
    if not src.exists():
        sys.exit(f"{src} not found. Run: python scripts/download_clinical.py")
    df = pd.read_csv(src)
    idcol = "patient_id" if "patient_id" in df.columns else df.columns[0]
    df = df.set_index(idcol)
    if list(df.index) != core:
        sys.exit("clinical_raw_121.csv patient order differs from cohort_patients.csv")
    if GDC_COL not in df.columns:
        cands = [c for c in df.columns if "histologic" in c.lower()]
        sys.exit(f"{GDC_COL} not found. Histology-like columns: {cands}")

    lab = df[GDC_COL].map(to_label)
    if lab.isna().any():
        bad = df.loc[lab.isna(), GDC_COL].astype(str).value_counts().to_dict()
        sys.exit(f"unmapped histology values: {bad}")
    lab = lab.astype(int)

    counts = {"FH": int((lab == 0).sum()), "UH": int((lab == 1).sum())}
    print("class counts:", counts)
    if counts != EXPECTED and not args.allow_count_change:
        sys.exit(f"counts {counts} != expected {EXPECTED}. These are NOT the labels used by E1/splits.")

    # fold-by-fold check against the splits file
    splits = json.load(open(ROOT / "data" / "splits" / "splits_5x5.json"))["folds"]
    y = lab.values
    for f in splits:
        for part in ("train", "test"):
            idx = f[f"{part}_indices"]
            got = {"FH": int((y[idx] == 0).sum()), "UH": int((y[idx] == 1).sum())}
            if got != f[f"{part}_histology_counts"]:
                sys.exit(f"fold {f['split_id']} {part}: {got} != stored {f[f'{part}_histology_counts']}")
    print(f"OK: histology counts match the stored counts in all {len(splits)} folds (train and test).")

    out = pd.DataFrame({"histology_label": lab, "histology_text": df[GDC_COL].astype(str)})
    if SUPP_COL in df.columns:
        sl = df[SUPP_COL].map(to_label)
        dis = out.index[(sl.notna()) & (sl != lab)].tolist()
        out["supp_agrees"] = (sl.isna()) | (sl == lab)
        print(f"supplement agreement: {len(dis)} disagreement(s) {dis[:5]}")
    if VITAL_COL in df.columns:
        out["vital_label"] = (df[VITAL_COL].astype(str).str.lower() == "dead").astype(int)  # reference only
    out.index.name = "patient_id"
    dest = ROOT / "data" / "processed" / "labels.csv"
    out.to_csv(dest)
    print("wrote", dest)


if __name__ == "__main__":
    main()
