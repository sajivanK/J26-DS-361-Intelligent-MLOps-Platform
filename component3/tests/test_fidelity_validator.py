"""Unit tests for fidelity and diversity validation."""

import numpy as np
import pandas as pd
import pytest

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.validation.fidelity_validator import (
    FidelityValidator,
    validate_fidelity,
)
from component3.src.validation.validation_models import ValidationStatus


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(35, 10, 200),
        "income": np.random.normal(50000, 15000, 200),
        "education": np.random.choice(["Bachelors", "Masters", "Doctorate"], size=200, p=[0.6, 0.3, 0.1]),
        "occupation": np.random.choice(["Tech", "Finance", "Healthcare"], size=200, p=[0.4, 0.4, 0.2]),
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


def test_identical_distributions_pass(reference_data, prepared_reference):
    # Synthetic data sampled from same distribution
    np.random.seed(101)
    synthetic = pd.DataFrame({
        "age": np.random.normal(35, 10, 200),
        "income": np.random.normal(50000, 15000, 200),
        "education": np.random.choice(["Bachelors", "Masters", "Doctorate"], size=200, p=[0.6, 0.3, 0.1]),
        "occupation": np.random.choice(["Tech", "Finance", "Healthcare"], size=200, p=[0.4, 0.4, 0.2]),
    })

    result = validate_fidelity(synthetic, prepared_reference)
    assert result.status == ValidationStatus.PASS
    assert len(result.reasons) == 0
    assert result.metrics["fidelity_score"] > 0.70
    assert "mean_ks_statistic" in result.metrics["continuous_metrics_summary"]
    assert "mean_js_divergence" in result.metrics["categorical_metrics_summary"]


def test_strong_distribution_shift_fails(reference_data, prepared_reference):
    # Massive distribution shift
    synthetic = pd.DataFrame({
        "age": np.random.normal(85, 2, 200),  # Much older
        "income": np.random.normal(5000, 100, 200),  # Much lower income
        "education": ["Doctorate"] * 200,  # Extreme shift
        "occupation": ["Healthcare"] * 200,
    })

    result = validate_fidelity(
        synthetic,
        prepared_reference,
        max_mean_ks_statistic=0.30,
        max_mean_js_divergence=0.30,
    )
    assert result.status == ValidationStatus.FAIL
    assert len(result.reasons) > 0
    assert result.metrics["continuous_metrics_summary"]["mean_ks_statistic"] > 0.5


def test_duplicate_heavy_synthetic_data_fails(reference_data, prepared_reference):
    # 90% duplicated records
    base_row = reference_data.iloc[0:1]
    synthetic = pd.concat([base_row] * 100, ignore_index=True)

    result = validate_fidelity(
        synthetic,
        prepared_reference,
        max_duplicate_ratio=0.10,
    )
    assert result.status == ValidationStatus.FAIL
    assert any("duplicate ratio" in r for r in result.reasons)
    assert result.metrics["diversity_metrics"]["duplicate_ratio"] >= 0.90


def test_empty_datasets(reference_data, prepared_reference):
    empty_df = pd.DataFrame(columns=reference_data.columns)

    res_syn_empty = validate_fidelity(empty_df, prepared_reference)
    assert res_syn_empty.status == ValidationStatus.FAIL
    assert "Synthetic data is empty" in res_syn_empty.reasons[0]

    res_ref_empty = validate_fidelity(reference_data, empty_df)
    assert res_ref_empty.status == ValidationStatus.FAIL
    assert "Reference data is empty" in res_ref_empty.reasons[0]


def test_missing_columns(reference_data, prepared_reference):
    partial_synthetic = reference_data[["age", "income"]].copy()

    result = validate_fidelity(partial_synthetic, prepared_reference)
    assert result.status == ValidationStatus.FAIL
    assert "Missing columns" in result.reasons[0]


def test_constant_columns(reference_data):
    # Constant columns edge case
    ref = reference_data.copy()
    ref["constant_col"] = 1.0
    syn = reference_data.copy()
    syn["constant_col"] = 1.0

    result = validate_fidelity(
        synthetic_data=syn,
        reference_data=ref,
        continuous_columns=["age", "income", "constant_col"],
        discrete_columns=["education", "occupation"],
    )
    assert result.status == ValidationStatus.PASS
    assert "constant_col" in result.metrics["continuous_metrics"]


def test_input_immutability(reference_data, prepared_reference):
    syn_original = reference_data.copy()
    syn_copy = syn_original.copy()

    _ = validate_fidelity(syn_original, prepared_reference)
    pd.testing.assert_frame_equal(syn_original, syn_copy)


def test_validator_class_interface(reference_data, prepared_reference):
    validator = FidelityValidator(
        max_mean_ks_statistic=0.40,
        max_mean_js_divergence=0.40,
        max_duplicate_ratio=0.20,
    )
    res = validator.validate(reference_data, prepared_reference)
    assert res.status == ValidationStatus.PASS
    assert isinstance(res.to_dict(), dict)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        FidelityValidator(max_mean_ks_statistic=-0.1)
    with pytest.raises(ValueError):
        FidelityValidator(max_mean_js_divergence=1.5)
    with pytest.raises(ValueError):
        FidelityValidator(max_duplicate_ratio=True)


def test_correlation_matrix_excludes_diagonal_zeros():
    # 2 features where correlation changed from +1.0 to -1.0 (difference = 2.0)
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    syn = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})

    result = validate_fidelity(
        synthetic_data=syn,
        reference_data=ref,
        continuous_columns=["a", "b"],
        discrete_columns=[],
    )
    # The true difference between feature a and b is 2.0. Excluding diagonal zeros guarantees 2.0.
    corr_diff = result.metrics["correlation_difference"]["mean_absolute_difference"]
    assert np.isclose(corr_diff, 2.0)


def test_continuous_all_nan_fails():
    ref = pd.DataFrame({"age": [20.0, 30.0, 40.0], "edu": ["B", "M", "D"]})
    syn = pd.DataFrame({"age": [np.nan, np.nan, np.nan], "edu": ["B", "M", "D"]})

    result = validate_fidelity(
        synthetic_data=syn,
        reference_data=ref,
        continuous_columns=["age"],
        discrete_columns=["edu"],
    )
    assert result.status == ValidationStatus.FAIL
    assert any("no valid numeric values" in r for r in result.reasons)

