"""
BioFusionFormer - Repeated Stratified 5-Fold Cross-Validation Split Generator
Generates reproducible 5 repeats x 5 folds = 25 test folds.
Uses fixed seeds (42, 43, 44, 45, 46).
Supports:
  1. Primary stratification: wilms_tumor_histologic_subtype.diagnoses
  2. Exploratory stratification: vital_status.demographic
  3. Joint stratification: histology_x_vital_status (4 balanced strata)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load aligned cohort
cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
print(f"Total 3-omics core patients: {len(core_pats)}")

# Load clinical annotations
clin_df = pd.read_csv(DATA_DIR / "raw" / "target_wt_xena" / "TARGET-WT.clinical.tsv.gz", sep="\t")
clin_df["patient_id"] = clin_df["sample"].str.slice(0, 16)
core_clin = clin_df[clin_df["patient_id"].isin(core_pats)].drop_duplicates(subset=["patient_id"]).copy()
core_clin = core_clin.set_index("patient_id").loc[core_pats].reset_index()

# Extract labels
histology = (core_clin["wilms_tumor_histologic_subtype.diagnoses"] == "Unfavorable").astype(int).values
vital_status = (core_clin["vital_status.demographic"] == "Dead").astype(int).values

# Create joint stratum: 0: FH+Alive, 1: FH+Dead, 2: UH+Alive, 3: UH+Dead
joint_strata = histology * 2 + vital_status
print("Histology counts (0=FH, 1=UH):", np.bincount(histology))
print("Vital status counts (0=Alive, 1=Dead):", np.bincount(vital_status))
print("Joint strata counts (0=FH/Alive, 1=FH/Dead, 2=UH/Alive, 3=UH/Dead):", np.bincount(joint_strata))

SEEDS = [42, 43, 44, 45, 46]


def create_splits(strata_vector, strata_name):
    splits_data = {
        "metadata": {
            "n_repeats": 5,
            "n_splits": 5,
            "total_folds": 25,
            "n_samples": len(core_pats),
            "seeds": SEEDS,
            "strata_name": strata_name,
            "patient_ids": core_pats,
        },
        "folds": []
    }

    for repeat_idx, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(core_pats)), strata_vector)):
            train_pats = [core_pats[i] for i in train_idx]
            test_pats = [core_pats[i] for i in test_idx]

            fold_info = {
                "repeat": repeat_idx,
                "fold": fold_idx,
                "split_id": f"repeat_{repeat_idx}_fold_{fold_idx}",
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_patient_ids": train_pats,
                "test_patient_ids": test_pats,
                "train_histology_counts": {
                    "FH": int((histology[train_idx] == 0).sum()),
                    "UH": int((histology[train_idx] == 1).sum()),
                },
                "test_histology_counts": {
                    "FH": int((histology[test_idx] == 0).sum()),
                    "UH": int((histology[test_idx] == 1).sum()),
                },
                "train_vital_counts": {
                    "Alive": int((vital_status[train_idx] == 0).sum()),
                    "Dead": int((vital_status[train_idx] == 1).sum()),
                },
                "test_vital_counts": {
                    "Alive": int((vital_status[test_idx] == 0).sum()),
                    "Dead": int((vital_status[test_idx] == 1).sum()),
                },
            }
            splits_data["folds"].append(fold_info)

    return splits_data


# 1. Primary splits (stratified strictly on Histology)
splits_hist = create_splits(histology, "histology_primary")
with open(SPLITS_DIR / "splits_histology_5x5.json", "w") as f:
    json.dump(splits_hist, f, indent=2)
print("Saved data/splits/splits_histology_5x5.json")

# 2. Exploratory splits (stratified strictly on Vital Status)
splits_vital = create_splits(vital_status, "vital_status_exploratory")
with open(SPLITS_DIR / "splits_vital_status_5x5.json", "w") as f:
    json.dump(splits_vital, f, indent=2)
print("Saved data/splits/splits_vital_status_5x5.json")

# 3. Joint splits (stratified on both simultaneously)
splits_joint = create_splits(joint_strata, "joint_histology_x_vital_status")
with open(SPLITS_DIR / "splits_joint_5x5.json", "w") as f:
    json.dump(splits_joint, f, indent=2)
print("Saved data/splits/splits_joint_5x5.json")

# Standard canonical splits_5x5.json (pointing to joint stratification so both labels use identical folds)
with open(SPLITS_DIR / "splits_5x5.json", "w") as f:
    json.dump(splits_joint, f, indent=2)
print("Saved data/splits/splits_5x5.json (canonical joint stratification)")
