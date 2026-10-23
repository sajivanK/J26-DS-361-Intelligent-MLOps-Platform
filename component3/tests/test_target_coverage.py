"""Automated unit tests for target-slice coverage evaluation."""

import pandas as pd
import pytest

from component3.src.contracts.safe_slice_evaluator import UnsafeSliceRuleError
from component3.src.generators.target_coverage import (
    TargetCoverageResult,
    evaluate_target_coverage,
)


@pytest.fixture
def sample_data():
    """Sample dataset with 4 rows of categorical and continuous features."""
    return pd.DataFrame({
        "age": [0.2, 0.5, 0.8, 0.9],
        "education_num": [-0.5, -0.5, -0.5, -0.5],
        "race": [1, 2, 2, 3],
        "workclass": [1, 1, 2, 2],
    })


def test_categorical_condition(sample_data):
    """Categorical equality slice rule evaluation."""
    result = evaluate_target_coverage(sample_data, "race == 2")
    assert result.total_rows == 4
    assert result.matching_rows == 2
    assert result.non_matching_rows == 2
    assert result.coverage == 0.5
    assert result.slice_rule == "race == 2"


def test_continuous_condition(sample_data):
    """Continuous range slice rule evaluation."""
    result = evaluate_target_coverage(
        sample_data,
        "age > 0.69 and education_num < -0.03",
    )
    assert result.total_rows == 4
    assert result.matching_rows == 2
    assert result.non_matching_rows == 2
    assert result.coverage == 0.5


def test_full_coverage(sample_data):
    """Slice rule matching all records yields 1.0 coverage."""
    result = evaluate_target_coverage(sample_data, "age > 0.0")
    assert result.total_rows == 4
    assert result.matching_rows == 4
    assert result.non_matching_rows == 0
    assert result.coverage == 1.0


def test_zero_coverage(sample_data):
    """Slice rule matching no records yields 0.0 coverage."""
    result = evaluate_target_coverage(sample_data, "race == 99")
    assert result.total_rows == 4
    assert result.matching_rows == 0
    assert result.non_matching_rows == 4
    assert result.coverage == 0.0


def test_or_condition(sample_data):
    """Disjunctive slice rule evaluation."""
    result = evaluate_target_coverage(sample_data, "race == 2 or age > 0.85")
    assert result.total_rows == 4
    assert result.matching_rows == 3
    assert result.non_matching_rows == 1
    assert result.coverage == 0.75


def test_result_serialization(sample_data):
    result = evaluate_target_coverage(sample_data, "race == 2")
    assert isinstance(result, TargetCoverageResult)
    assert result.to_dict() == {
        "total_rows": 4,
        "matching_rows": 2,
        "non_matching_rows": 2,
        "coverage": 0.5,
        "slice_rule": "race == 2",
    }


def test_input_dataframe_not_modified(sample_data):
    original = sample_data.copy(deep=True)
    evaluate_target_coverage(sample_data, "race == 2")
    pd.testing.assert_frame_equal(sample_data, original)


def test_empty_dataframe_rejected():
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_target_coverage(pd.DataFrame(), "race == 2")


def test_unknown_column_rejected(sample_data):
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_target_coverage(sample_data, "unknown_feature > 0")


def test_unsafe_expression_rejected(sample_data):
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_target_coverage(
            sample_data,
            "__import__('os').system('echo unsafe')",
        )
