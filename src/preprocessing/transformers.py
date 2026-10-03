"""
BioFusionFormer - Preprocessing Transformers
Implements scikit-learn compatible fit/transform classes to ensure strict in-fold preprocessing
without test set leakage.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OmicsStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardizes high-dimensional omics features (zero mean, unit variance)
    computed strictly on the training fold.
    """
    def __init__(self, with_mean=True, with_std=True, eps=1e-8):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.mean_ = None
        self.scale_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if self.with_mean:
            self.mean_ = np.nanmean(X_arr, axis=0)
        else:
            self.mean_ = np.zeros(X_arr.shape[1], dtype=np.float32)

        if self.with_std:
            std = np.nanstd(X_arr, axis=0)
            std[std < self.eps] = 1.0  # prevent division by zero for constant features
            self.scale_ = std
        else:
            self.scale_ = np.ones(X_arr.shape[1], dtype=np.float32)

        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        cols = X.columns if is_df else None
        idx = X.index if is_df else None
        X_arr = X.values if is_df else np.asarray(X)

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("OmicsStandardizer must be fitted before transform.")

        X_norm = (X_arr - self.mean_) / self.scale_

        if is_df:
            return pd.DataFrame(X_norm, index=idx, columns=cols)
        return X_norm


class InFoldImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using median/mean computed strictly on training folds.
    """
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self.strategy == "median":
            self.statistics_ = np.nanmedian(X_arr, axis=0)
        elif self.strategy == "mean":
            self.statistics_ = np.nanmean(X_arr, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        # If any feature was 100% NaN in train fold, default to 0.0
        self.statistics_ = np.nan_to_num(self.statistics_, nan=0.0)
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        cols = X.columns if is_df else None
        idx = X.index if is_df else None
        X_arr = np.array(X.values if is_df else X, copy=True)

        nan_mask = np.isnan(X_arr)
        if np.any(nan_mask):
            # Broadcast replacement per column
            for col_idx in range(X_arr.shape[1]):
                m = nan_mask[:, col_idx]
                if np.any(m):
                    X_arr[m, col_idx] = self.statistics_[col_idx]

        if is_df:
            return pd.DataFrame(X_arr, index=idx, columns=cols)
        return X_arr


class ClinicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes clinical baseline variables (age at diagnosis, gender, optional stage)
    strictly fitted on training fold.
    """
    def __init__(self, include_stage=False):
        self.include_stage = include_stage
        self.age_mean_ = None
        self.age_std_ = None
        self.stage_categories_ = None
        self.stage_impute_mode_ = None

    def fit(self, df: pd.DataFrame, y=None):
        # Age
        age = pd.to_numeric(df["age_at_earliest_diagnosis_in_years.diagnoses.xena_derived"], errors="coerce").values
        self.age_mean_ = np.nanmean(age)
        self.age_std_ = np.nanstd(age)
        if self.age_std_ < 1e-6:
            self.age_std_ = 1.0

        if self.include_stage and "stage_clean" in df.columns:
            st = df["stage_clean"].dropna()
            self.stage_impute_mode_ = st.mode().iloc[0] if not st.empty else "Stage II"
            self.stage_categories_ = ["Stage I", "Stage II", "Stage III", "Stage IV"]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}
        # 1. Standardized age
        age = pd.to_numeric(df["age_at_earliest_diagnosis_in_years.diagnoses.xena_derived"], errors="coerce").values
        age_clean = np.nan_to_num(age, nan=self.age_mean_)
        features["age_scaled"] = (age_clean - self.age_mean_) / self.age_std_

        # 2. Gender binary (male = 1, female = 0)
        gender = df["gender.demographic"].astype(str).str.lower()
        features["gender_male"] = (gender == "male").astype(float).values

        out_df = pd.DataFrame(features, index=df.index)

        # 3. Optional Stage encoding
        if self.include_stage and "stage_clean" in df.columns:
            st = df["stage_clean"].fillna(self.stage_impute_mode_)
            for cat in self.stage_categories_:
                out_df[f"stage_{cat.replace(' ', '_')}"] = (st == cat).astype(float).values

        return out_df
