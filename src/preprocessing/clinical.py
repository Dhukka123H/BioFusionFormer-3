"""
src/preprocessing/clinical.py
=============================
Fold-safe ClinicalPreprocessor for BioFusionFormer (Stage 5.1).

Design rules per AGENT_PROMPT Stage 5.1:
  1. Age: log1p(age_days) + z-score fitted strictly on the training fold.
  2. Sex: binary (0 = Female, 1 = Male).
  3. Stage: ordinal I < II < III < IV with IIIB merged into III (1, 2, 3, 4).
     In-fold imputation (mode) + missing indicator column.
  4. Column-name blacklist: fails immediately if any histology, risk group,
     vital status, follow-up, or administrative column appears.
  5. Patient order assertion: asserts patient order equals the omics cohort order.
  6. Primary feature set: [age, sex, stage] (plus stage missing indicator if missingness exists).
  7. Optional sensitivity set: [race, ethnicity, percent_tumor_nuclei, percent_necrosis, percent_stroma].
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Blacklist keywords that indicate label leakage, outcome leakage, or administrative noise
FORBIDDEN_KEYWORDS = [
    "histol", "morphol", "anaplas", "subtype", "cog_risk", "cog_renal",
    "tumor_grade", "pathology", "diagnosis_confirmed",
    "vital_status", "days_to_death", "cause_of_death", "death",
    "follow_up", "first_event", "event_free", "overall_survival",
    "recurrence", "progression",
    "case_id", "submitter_id", "project_id", "protocol", "year_of",
]

STAGE_MAPPING = {
    "I": 1.0,
    "Stage I": 1.0,
    "1": 1.0,
    "II": 2.0,
    "Stage II": 2.0,
    "2": 2.0,
    "III": 3.0,
    "Stage III": 3.0,
    "3": 3.0,
    "IIIB": 3.0,  # Merged into III per Stage 5.1 requirement
    "Stage IIIB": 3.0,
    "IV": 4.0,
    "Stage IV": 4.0,
    "4": 4.0,
}


def check_column_blacklist(columns: List[str]) -> None:
    """Raises ValueError if any forbidden keyword is found in any column name."""
    forbidden_found = []
    for col in columns:
        col_lower = str(col).lower()
        for kw in FORBIDDEN_KEYWORDS:
            if kw in col_lower:
                forbidden_found.append((col, kw))
                break
    if forbidden_found:
        details = ", ".join(f"'{c}' (matches '{kw}')" for c, kw in forbidden_found)
        raise ValueError(
            f"Blacklist violation: Forbidden clinical column(s) detected: {details}. "
            f"Ensure no histology, outcome, follow-up, or administrative columns are passed."
        )


class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Fold-safe clinical transformer fitting all scalers and imputers strictly inside the training fold.
    """

    def __init__(self, include_sensitivity: bool = False):
        self.include_sensitivity = include_sensitivity
        self.age_mean_ = None
        self.age_std_ = None
        self.stage_impute_val_ = None
        self.gender_impute_val_ = None
        self.sensitivity_imputers_ = {}
        self.sensitivity_scalers_ = {}
        self.feature_names_out_ = []

    def fit(self, df: pd.DataFrame, y=None, cohort_order: Optional[List[str]] = None):
        """Fit strictly on training fold samples."""
        check_column_blacklist(df.columns.tolist())

        if cohort_order is not None and "patient_id" in df.columns:
            # If a subset is passed for training, check that order is consistent with cohort_order
            pats = df["patient_id"].tolist()
            filtered_cohort = [p for p in cohort_order if p in pats]
            if pats != filtered_cohort:
                raise ValueError("Patient order does not match omics cohort order.")

        # 1. Age: log1p + z-score
        age_col = None
        for c in ["age_at_diagnosis_days", "supp_Age at Diagnosis in Days", "age_days", "age"]:
            if c in df.columns:
                age_col = c
                break
        if age_col is None:
            raise KeyError("Age column not found in DataFrame.")

        age_vals = pd.to_numeric(df[age_col], errors="coerce").values.astype(float)
        log_age = np.log1p(np.maximum(0.0, age_vals[~np.isnan(age_vals)]))
        self.age_mean_ = float(np.mean(log_age)) if len(log_age) > 0 else 0.0
        self.age_std_ = float(np.std(log_age)) if len(log_age) > 0 else 1.0
        if self.age_std_ < 1e-6:
            self.age_std_ = 1.0

        # 2. Gender / Sex: binary (0 = Female, 1 = Male)
        gender_col = None
        for c in ["gender", "supp_Gender", "sex"]:
            if c in df.columns:
                gender_col = c
                break
        if gender_col is not None:
            g_series = df[gender_col].astype(str).str.lower().map({"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0})
            valid_g = g_series.dropna()
            self.gender_impute_val_ = float(valid_g.mode().iloc[0]) if not valid_g.empty else 0.0

        # 3. Stage: ordinal I < II < III < IV with IIIB -> III
        stage_col = None
        for c in ["stage", "supp_Stage", "stage_clean"]:
            if c in df.columns:
                stage_col = c
                break
        if stage_col is not None:
            st_mapped = df[stage_col].map(STAGE_MAPPING)
            valid_st = st_mapped.dropna()
            self.stage_impute_val_ = float(valid_st.mode().iloc[0]) if not valid_st.empty else 2.0

        # 4. Optional sensitivity features
        if self.include_sensitivity:
            sens_cols = ["percent_tumor_nuclei", "percent_necrosis", "percent_stroma"]
            for col in sens_cols:
                matching = [c for c in df.columns if col in c.lower()]
                if matching:
                    mcol = matching[0]
                    vals = pd.to_numeric(df[mcol], errors="coerce").values.astype(float)
                    valid_vals = vals[~np.isnan(vals)]
                    med = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
                    std = float(np.std(valid_vals)) if len(valid_vals) > 0 else 1.0
                    self.sensitivity_imputers_[col] = med
                    self.sensitivity_scalers_[col] = std if std > 1e-6 else 1.0

        return self

    def transform(self, df: pd.DataFrame, cohort_order: Optional[List[str]] = None) -> pd.DataFrame:
        """Transform test (or train) fold using strictly fitted parameters."""
        check_column_blacklist(df.columns.tolist())

        if cohort_order is not None and "patient_id" in df.columns:
            pats = df["patient_id"].tolist()
            filtered_cohort = [p for p in cohort_order if p in pats]
            if pats != filtered_cohort:
                raise ValueError("Patient order does not match omics cohort order.")

        features = {}

        # 1. Age: log1p + z-score + missing indicator
        age_col = None
        for c in ["age_at_diagnosis_days", "supp_Age at Diagnosis in Days", "age_days", "age"]:
            if c in df.columns:
                age_col = c
                break
        age_raw = pd.to_numeric(df[age_col], errors="coerce").values.astype(float)
        age_missing = np.isnan(age_raw)
        log_age = np.log1p(np.maximum(0.0, np.nan_to_num(age_raw, nan=0.0)))
        age_norm = (log_age - self.age_mean_) / self.age_std_
        age_norm[age_missing] = 0.0  # imputed to train mean in z-score space
        features["age_scaled"] = age_norm
        if np.any(age_missing):
            features["age_missing"] = age_missing.astype(float)

        # 2. Gender / Sex: binary
        gender_col = None
        for c in ["gender", "supp_Gender", "sex"]:
            if c in df.columns:
                gender_col = c
                break
        if gender_col is not None:
            g_series = df[gender_col].astype(str).str.lower().map({"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0})
            g_missing = g_series.isna()
            features["sex_binary"] = g_series.fillna(self.gender_impute_val_).values.astype(float)
            if np.any(g_missing):
                features["sex_missing"] = g_missing.values.astype(float)

        # 3. Stage: ordinal (1.0 - 4.0) + missing indicator
        stage_col = None
        for c in ["stage", "supp_Stage", "stage_clean"]:
            if c in df.columns:
                stage_col = c
                break
        if stage_col is not None:
            st_mapped = df[stage_col].map(STAGE_MAPPING)
            st_missing = st_mapped.isna()
            features["stage_ordinal"] = st_mapped.fillna(self.stage_impute_val_).values.astype(float)
            if np.any(st_missing):
                features["stage_missing"] = st_missing.values.astype(float)

        # 4. Sensitivity features (if requested)
        if self.include_sensitivity:
            sens_cols = ["percent_tumor_nuclei", "percent_necrosis", "percent_stroma"]
            for col in sens_cols:
                matching = [c for c in df.columns if col in c.lower()]
                if matching and col in self.sensitivity_imputers_:
                    mcol = matching[0]
                    vals = pd.to_numeric(df[mcol], errors="coerce").values.astype(float)
                    missing = np.isnan(vals)
                    imputed = np.nan_to_num(vals, nan=self.sensitivity_imputers_[col])
                    scaled = (imputed - self.sensitivity_imputers_[col]) / self.sensitivity_scalers_[col]
                    features[f"{col}_scaled"] = scaled

        out_df = pd.DataFrame(features, index=df.index)
        self.feature_names_out_ = out_df.columns.tolist()
        return out_df
