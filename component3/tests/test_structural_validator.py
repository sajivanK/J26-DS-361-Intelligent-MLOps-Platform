"""Unit tests for the Component 3 structural and target coverage validator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.validation.structural_validator import (
    StructuralValidator,
    validate_structural,
)
from component3.src.validation.validation_models import ValidationStatus


@pytest.fixture
def training_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "race": [1, 2, 2, 3],
        }
    )


@pytest.fixture
def prepared_data(training_reference: pd.DataFrame) -> PreparedGeneratorData:
    return PreparedGeneratorData(
        training_data=training_reference,
        discrete_columns=("race",),
        continuous_columns=("age",),
        n_rows=len(training_reference),
        source="affected_slice",
        target_included=False,
        slice_support=1.0,
        generation_mode="unsupervised_exploratory",
        feature_types={"age": "continuous", "race": "categorical"},
    )


@pytest.fixture
def valid_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25.0, 35.0, 45.0, 30.0, 40.0],
            "race": [2, 2, 2, 2, 1],
        }
    )


def test_valid_synthetic_data_passes(prepared_data, valid_synthetic):
    result = validate_structural(
        valid_synthetic,
        prepared_data,
        slice_rule="race == 2",
    )
    assert result.status == ValidationStatus.PASS
    assert result.stage == "structural"
    assert result.reasons == ()


def test_empty_dataset_fails(prepared_data):
    synthetic = pd.DataFrame(columns=["age", "race"])
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert any("empty" in reason.lower() for reason in result.reasons)


def test_missing_column_fails(prepared_data):
    synthetic = pd.DataFrame({"age": [25.0, 35.0]})
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL


def test_extra_column_fails(prepared_data, valid_synthetic):
    synthetic = valid_synthetic.assign(unexpected=1)
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL


def test_duplicate_columns_fail(prepared_data):
    synthetic = pd.DataFrame(
        [[25.0, 2, 2]],
        columns=["age", "race", "race"],
    )
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL


def test_missing_values_fail(prepared_data, valid_synthetic):
    synthetic = valid_synthetic.copy()
    synthetic.loc[0, "age"] = np.nan
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert result.metrics["missing_values"] == 1


def test_missing_categorical_values_fail(prepared_data, valid_synthetic):
    synthetic = valid_synthetic.copy()
    synthetic.loc[0, "race"] = np.nan
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert result.metrics["missing_values"] == 1


@pytest.mark.parametrize("val", [np.inf, -np.inf])
def test_non_finite_numerical_values_fail(
    prepared_data, valid_synthetic, val
):
    synthetic = valid_synthetic.copy()
    synthetic.loc[0, "age"] = val
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert any("non-finite" in r.lower() for r in result.reasons)


def test_non_numeric_continuous_column_fails(prepared_data, valid_synthetic):
    synthetic = valid_synthetic.copy()
    synthetic["age"] = ["25", "35", "45", "30", "40"]
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert any("not numeric" in r.lower() for r in result.reasons)


def test_unseen_categorical_value_fails(prepared_data, valid_synthetic):
    synthetic = valid_synthetic.copy()
    synthetic.loc[0, "race"] = 99
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert "race" in result.metrics["unseen_categories"]


def test_low_target_coverage_fails(prepared_data, valid_synthetic):
    result = validate_structural(
        valid_synthetic,
        prepared_data,
        slice_rule="race == 1",
        min_target_coverage=0.8,
    )
    assert result.status == ValidationStatus.FAIL
    assert any("coverage" in r.lower() for r in result.reasons)


@pytest.mark.parametrize(
    "threshold,expected_status",
    [
        (0.80, ValidationStatus.PASS),
        (0.85, ValidationStatus.FAIL),
    ],
)
def test_target_coverage_boundary(
    prepared_data, valid_synthetic, threshold, expected_status
):
    result = validate_structural(
        valid_synthetic,
        prepared_data,
        slice_rule="race == 2",
        min_target_coverage=threshold,
    )
    assert result.status == expected_status


def test_observed_range_violations_diagnostic_only(
    prepared_data, valid_synthetic
):
    synthetic = valid_synthetic.copy()
    synthetic.loc[0, "age"] = 99.0  # Training max is 50.0
    result = validate_structural(
        synthetic, prepared_data, slice_rule="race == 2"
    )
    assert result.status == ValidationStatus.PASS
    assert (
        result.metrics["numerical_range_violations"]["age"][
            "out_of_range_count"
        ]
        == 1
    )


def test_input_data_immutability(prepared_data, valid_synthetic):
    syn_copy = valid_synthetic.copy(deep=True)
    ref_copy = prepared_data.training_data.copy(deep=True)
    validate_structural(valid_synthetic, prepared_data, slice_rule="race == 2")
    pd.testing.assert_frame_equal(valid_synthetic, syn_copy)
    pd.testing.assert_frame_equal(prepared_data.training_data, ref_copy)


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.5])
def test_invalid_target_coverage_threshold_raises(
    prepared_data, valid_synthetic, invalid_threshold
):
    with pytest.raises(ValueError, match="min_target_coverage"):
        validate_structural(
            valid_synthetic,
            prepared_data,
            min_target_coverage=invalid_threshold,
        )


def test_invalid_synthetic_data_type_raises(prepared_data):
    with pytest.raises(TypeError):
        validate_structural("not a dataframe", prepared_data)


def test_invalid_prepared_data_type_raises(valid_synthetic):
    with pytest.raises(TypeError):
        validate_structural(valid_synthetic, "not a prepared_data")


def test_inconsistent_feature_types_fails(
    training_reference, valid_synthetic
):
    prep = PreparedGeneratorData(
        training_data=training_reference,
        discrete_columns=("race",),
        continuous_columns=("age",),
        n_rows=len(training_reference),
        source="affected_slice",
        target_included=False,
        slice_support=1.0,
        generation_mode="unsupervised_exploratory",
        feature_types={"age": "continuous"},  # Inconsistent: missing 'race'
    )
    result = validate_structural(valid_synthetic, prep)
    assert result.status == ValidationStatus.FAIL
    assert any("feature_types" in r.lower() for r in result.reasons)


def test_structural_validator_class(prepared_data, valid_synthetic):
    validator = StructuralValidator(min_target_coverage=0.8)
    result = validator.validate(
        valid_synthetic, prepared_data, slice_rule="race == 2"
    )
    assert result.status == ValidationStatus.PASS


def test_result_to_dict_serialization(prepared_data, valid_synthetic):
    result = validate_structural(
        valid_synthetic, prepared_data, slice_rule="race == 2"
    )
    d = result.to_dict()
    assert d["stage"] == "structural"
    assert d["status"] == "PASS"
    assert isinstance(d["metrics"], dict)
    assert isinstance(d["reasons"], list)


def test_boolean_continuous_column_fails(prepared_data):
    synthetic = pd.DataFrame(
        {
            "age": [True, False, True, False],
            "race": [2, 2, 2, 2],
        }
    )
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert any("not numeric: age" in r for r in result.reasons)


def test_mixed_type_unseen_categories_handled_safely(prepared_data):
    synthetic = pd.DataFrame(
        {
            "age": [25.0, 35.0],
            "race": ["unknown", 99],
        }
    )
    result = validate_structural(synthetic, prepared_data)
    assert result.status == ValidationStatus.FAIL
    assert any("unseen categories" in r for r in result.reasons)

