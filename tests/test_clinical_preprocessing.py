"""
tests/test_clinical_preprocessing.py
====================================
Tests for Stage 5.1 fold-safe ClinicalPreprocessor:
1. Changing test-fold values must not change fitted transformer.
2. Blacklist fails if any forbidden column appears.
3. Ordinal stage mapping with IIIB merged into III.
4. Patient order assertion.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.clinical import ClinicalPreprocessor, check_column_blacklist


def test_blacklist_rejects_forbidden_columns():
    forbidden_cols = [
        ["age_at_diagnosis_days", "gender", "stage", "histologic_type"],
        ["age_at_diagnosis_days", "vital_status"],
        ["age_at_diagnosis_days", "days_to_death"],
        ["age_at_diagnosis_days", "cog_renal_stage"],
        ["age_at_diagnosis_days", "case_id"],
        ["age_at_diagnosis_days", "submitter_id"],
    ]
    for cols in forbidden_cols:
        with pytest.raises(ValueError, match="Blacklist violation"):
            check_column_blacklist(cols)


def test_blacklist_accepts_clean_columns():
    clean_cols = ["patient_id", "age_at_diagnosis_days", "gender", "stage", "percent_tumor_nuclei", "percent_necrosis", "percent_stroma"]
    check_column_blacklist(clean_cols)


def test_changing_test_fold_does_not_affect_fitted_transformer():
    train_df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "age_at_diagnosis_days": [1000, 2000, 3000, 4000],
        "gender": ["Male", "Female", "Male", "Female"],
        "stage": ["I", "II", "III", "IV"],
    })
    test_df_1 = pd.DataFrame({
        "patient_id": ["P5", "P6"],
        "age_at_diagnosis_days": [1500, 2500],
        "gender": ["Female", "Male"],
        "stage": ["II", "III"],
    })
    test_df_2 = pd.DataFrame({
        "patient_id": ["P5", "P6"],
        "age_at_diagnosis_days": [99999, 99999],  # radically changed test data
        "gender": ["Male", "Male"],
        "stage": ["IV", "IV"],
    })

    prep = ClinicalPreprocessor()
    prep.fit(train_df)

    fitted_age_mean = prep.age_mean_
    fitted_age_std = prep.age_std_
    fitted_stage = prep.stage_impute_val_

    # Transform test_df_1
    out_1 = prep.transform(test_df_1)

    # Transform test_df_2
    out_2 = prep.transform(test_df_2)

    # Fitted parameters must not change
    assert prep.age_mean_ == fitted_age_mean
    assert prep.age_std_ == fitted_age_std
    assert prep.stage_impute_val_ == fitted_stage


def test_stage_ordinal_mapping_and_iiib_merge():
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4", "P5"],
        "age_at_diagnosis_days": [1000, 1000, 1000, 1000, 1000],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "stage": ["I", "II", "III", "IIIB", "IV"],
    })
    prep = ClinicalPreprocessor()
    prep.fit(df)
    out = prep.transform(df)

    # Stage values must be 1.0, 2.0, 3.0, 3.0, 4.0
    expected_stages = [1.0, 2.0, 3.0, 3.0, 4.0]
    np.testing.assert_array_equal(out["stage_ordinal"].values, expected_stages)


def test_patient_order_assertion():
    cohort_order = ["P1", "P2", "P3"]
    wrong_order_df = pd.DataFrame({
        "patient_id": ["P2", "P1", "P3"],
        "age_at_diagnosis_days": [1000, 2000, 3000],
        "gender": ["Male", "Female", "Male"],
        "stage": ["I", "II", "III"],
    })
    prep = ClinicalPreprocessor()
    with pytest.raises(ValueError, match="Patient order does not match omics cohort order"):
        prep.fit(wrong_order_df, cohort_order=cohort_order)
