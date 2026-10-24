"""Unit tests for privacy validation and memorization risk."""

import numpy as np
import pandas as pd
import pytest

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.validation.privacy_validator import (
    PrivacyValidator,
    validate_privacy,
)
from component3.src.validation.validation_models import ValidationStatus


@pytest.fixture
def reference_data():
    return pd.DataFrame({
        "age": [25.0, 35.0, 45.0, 55.0, 65.0],
        "income": [30000.0, 50000.0, 70000.0, 90000.0, 110000.0],
        "education": ["Bachelors", "Masters", "Doctorate", "Bachelors", "Masters"],
        "occupation": ["Tech", "Finance", "Healthcare", "Tech", "Finance"],
    })


@pytest.fixture
def prepared_reference(reference_data):
    return PreparedGeneratorData(
        training_data=reference_data.copy(),
        feature_types={
            "age": "continuous",
            "income": "continuous",
            "education": "categorical",
            "occupation": "categorical",
        },
        discrete_columns=("education", "occupation"),
        continuous_columns=("age", "income"),
        n_rows=len(reference_data),
        source="affected_slice",
        target_included=False,
        slice_support=0.06,
        generation_mode="unsupervised_exploratory",
    )


def test_distinct_records_pass(reference_data, prepared_reference):
    # Clearly distinct, perturbed records
    synthetic = pd.DataFrame({
        "age": [28.0, 38.0, 48.0, 58.0, 68.0],
        "income": [34000.0, 54000.0, 74000.0, 94000.0, 114000.0],
        "education": ["Masters", "Doctorate", "Bachelors", "Masters", "Doctorate"],
        "occupation": ["Finance", "Healthcare", "Tech", "Finance", "Healthcare"],
    })

    result = validate_privacy(synthetic, prepared_reference, min_dcr_p05=0.01)
    assert result.status == ValidationStatus.PASS
    assert len(result.reasons) == 0
    assert result.metrics["exact_matches"]["exact_match_count"] == 0
    assert result.metrics["dcr_metrics"]["dcr_p05"] > 0.01


def test_exact_copies_fail(reference_data, prepared_reference):
    # Contains exact copy of reference rows
    synthetic = reference_data.copy()

    result = validate_privacy(synthetic, prepared_reference, max_exact_match_ratio=0.0)
    assert result.status == ValidationStatus.FAIL
    assert any("Exact match ratio" in r for r in result.reasons)
    assert result.metrics["exact_matches"]["exact_match_count"] == len(reference_data)


def test_near_duplicate_memorization_fails(reference_data, prepared_reference):
    # Microscopic perturbation (e.g. 0.00001 diff in continuous)
    synthetic = reference_data.copy()
    synthetic["age"] = synthetic["age"] + 0.00001

    result = validate_privacy(
        synthetic,
        prepared_reference,
        max_exact_match_ratio=1.0,
        min_dcr_p05=0.05,
    )
    assert result.status == ValidationStatus.FAIL
    assert any("5th percentile DCR" in r for r in result.reasons)


def test_holdout_comparison(reference_data, prepared_reference):
    synthetic = pd.DataFrame({
        "age": [27.0, 37.0, 47.0],
        "income": [33000.0, 53000.0, 73000.0],
        "education": ["Bachelors", "Masters", "Doctorate"],
        "occupation": ["Tech", "Finance", "Healthcare"],
    })
    holdout = pd.DataFrame({
        "age": [26.0, 36.0, 46.0],
        "income": [31000.0, 51000.0, 71000.0],
        "education": ["Bachelors", "Masters", "Doctorate"],
        "occupation": ["Tech", "Finance", "Healthcare"],
    })

    result = validate_privacy(synthetic, prepared_reference, holdout_data=holdout, min_dcr_p05=0.001)
    assert result.status == ValidationStatus.PASS
    assert "holdout_comparison" in result.metrics
    assert "nndr" in result.metrics["holdout_comparison"]


def test_empty_datasets(reference_data, prepared_reference):
    empty_df = pd.DataFrame(columns=reference_data.columns)

    res_syn_empty = validate_privacy(empty_df, prepared_reference)
    assert res_syn_empty.status == ValidationStatus.FAIL
    assert "Synthetic data is empty" in res_syn_empty.reasons[0]

    res_ref_empty = validate_privacy(reference_data, empty_df)
    assert res_ref_empty.status == ValidationStatus.FAIL
    assert "Reference data is empty" in res_ref_empty.reasons[0]


def test_input_immutability(reference_data, prepared_reference):
    syn_original = reference_data.copy()
    syn_copy = syn_original.copy()

    _ = validate_privacy(syn_original, prepared_reference)
    pd.testing.assert_frame_equal(syn_original, syn_copy)


def test_validator_class_interface(reference_data, prepared_reference):
    validator = PrivacyValidator(
        max_exact_match_ratio=0.05,
        min_dcr_p05=0.01,
        epsilon_near_duplicate=0.005,
    )
    res = validator.validate(reference_data, prepared_reference)
    assert isinstance(res.to_dict(), dict)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        PrivacyValidator(max_exact_match_ratio=-0.1)
    with pytest.raises(ValueError):
        PrivacyValidator(min_dcr_p05=1.5)
    with pytest.raises(ValueError):
        PrivacyValidator(epsilon_near_duplicate=True)


def test_sub_five_percent_near_duplicate_fails(reference_data, prepared_reference):
    # 2 rows with microscopic perturbations (near duplicates), remaining rows distant
    synthetic = pd.concat([reference_data.iloc[0:2].copy(), reference_data.iloc[0:2].copy() * 10], ignore_index=True)
    # Give distant values to remaining rows
    synthetic.iloc[0, 0] = synthetic.iloc[0, 0] + 0.00001
    synthetic.iloc[1, 0] = synthetic.iloc[1, 0] + 0.00001

    result = validate_privacy(
        synthetic,
        prepared_reference,
        max_exact_match_ratio=1.0,
        min_dcr_p05=0.0,  # Disable p05 check to isolate near duplicate ratio check
        max_near_duplicate_ratio=0.0,
        epsilon_near_duplicate=0.01,
    )
    assert result.status == ValidationStatus.FAIL
    assert any("Near-duplicate ratio" in r for r in result.reasons)

