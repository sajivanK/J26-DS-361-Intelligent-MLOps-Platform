"""Automated unit tests for generator training data and metadata preparer."""

import pandas as pd
import pytest

from component3.src.generators.data_preparer import (
    DataPreparationError,
    PreparedGeneratorData,
    prepare_generator_data,
)
from component3.src.readiness.readiness_models import (
    ReadinessDecision,
    ReadinessStatus,
)


@pytest.fixture
def data():
    """Small sample dataset with 3 rows and 4 features plus target."""
    return pd.DataFrame({
        "relationship": ["Husband", "Not-in-family", "Own-child"],
        "workclass": ["Private", "Self-emp", "Private"],
        "hours_per_week": [40, 50, 20],
        "age": [35, 45, 25],
        "target": [0, 1, 0],
    })


@pytest.fixture
def config():
    """Authoritative test configuration."""
    return {
        "dataset": {
            "name": "uci_adult_sample",
            "target_column": "target",
            "feature_types": {
                "age": "continuous",
                "hours_per_week": "continuous",
                "workclass": "categorical",
                "relationship": "categorical",
            },
        }
    }


def decision(
    generation_ready: bool = True,
    supervised_ready: bool = False,
    status: ReadinessStatus = ReadinessStatus.REVIEW,
    current_rows: int = 10,
) -> ReadinessDecision:
    """Helper to create test readiness decisions."""
    return ReadinessDecision(
        status=status,
        generation_ready=generation_ready,
        supervised_retraining_ready=supervised_ready,
        current_rows=current_rows,
        reference_rows=100,
        affected_slice_rows=3,
    )


def test_not_a_dataframe(config):
    """Passing a non-DataFrame must raise DataPreparationError."""
    with pytest.raises(DataPreparationError):
        prepare_generator_data("not_df", decision(), config)


def test_empty_dataframe(config):
    """Empty training data must raise DataPreparationError."""
    with pytest.raises(DataPreparationError):
        prepare_generator_data(pd.DataFrame(), decision(), config)


def test_duplicate_columns(config):
    """Duplicate column names must raise DataPreparationError."""
    df = pd.DataFrame([[1, 2]], columns=["age", "age"])
    with pytest.raises(DataPreparationError):
        prepare_generator_data(df, decision(), config)


def test_missing_feature(data, config):
    """Missing required feature column must raise DataPreparationError."""
    df = data.drop(columns=["age"])
    with pytest.raises(DataPreparationError):
        prepare_generator_data(df, decision(), config)


def test_missing_values(data, config):
    """Missing values in required features must raise DataPreparationError."""
    df = data.copy()
    df.loc[0, "age"] = None
    with pytest.raises(DataPreparationError):
        prepare_generator_data(df, decision(), config)


def test_non_numeric_continuous(data, config):
    """Non-numeric values in continuous columns must raise DataPreparationError."""
    df = data.copy()
    df["age"] = ["young", "middle", "old"]
    with pytest.raises(DataPreparationError):
        prepare_generator_data(df, decision(), config)


def test_boolean_continuous(data, config):
    """Boolean values in continuous columns must raise DataPreparationError."""
    df = data.copy()
    df["age"] = [True, False, True]
    with pytest.raises(DataPreparationError):
        prepare_generator_data(df, decision(), config)


def test_invalid_source(data, config):
    """Empty or blank source string must raise DataPreparationError."""
    with pytest.raises(DataPreparationError):
        prepare_generator_data(data, decision(), config, source="   ")


def test_rows_exceed_current_rows(data, config):
    """Training rows exceeding assessed current dataset rows must raise DataPreparationError."""
    with pytest.raises(DataPreparationError):
        prepare_generator_data(data, decision(current_rows=2), config)


def test_generation_not_ready_blocked(data, config):
    """Readiness decision with generation_ready=False must be rejected."""
    with pytest.raises(DataPreparationError):
        prepare_generator_data(
            data,
            decision(generation_ready=False, status=ReadinessStatus.BLOCKED),
            config,
        )


def test_unsupervised_mode_excludes_target(data, config):
    """When supervised_ready=False, target column must be excluded."""
    result = prepare_generator_data(
        data,
        decision(supervised_ready=False),
        config,
    )
    assert result.target_included is False
    assert result.generation_mode == "unsupervised_exploratory"
    assert "target" not in result.training_data.columns


def test_target_included_when_supervised_ready(data, config):
    """When supervised_ready=True, target column must be included."""
    result = prepare_generator_data(
        data,
        decision(
            supervised_ready=True,
            status=ReadinessStatus.READY,
        ),
        config,
    )
    assert result.target_included is True
    assert result.generation_mode == "supervised_repair"
    assert "target" in result.training_data.columns


def test_feature_order_matches_configuration(data, config):
    """Training data columns must strictly follow the configured feature order."""
    result = prepare_generator_data(data, decision(), config)
    assert result.training_data.columns.tolist() == [
        "age",
        "hours_per_week",
        "workclass",
        "relationship",
    ]


def test_feature_type_metadata(data, config):
    """Continuous and discrete feature lists must match configuration."""
    result = prepare_generator_data(data, decision(), config)
    assert result.continuous_columns == (
        "age",
        "hours_per_week",
    )
    assert result.discrete_columns == (
        "workclass",
        "relationship",
    )


def test_slice_support(data, config):
    """Slice support must be accurately computed relative to assessed current rows."""
    result = prepare_generator_data(data, decision(), config)
    assert result.slice_support == 0.3
    assert result.source == "affected_slice"


def test_input_dataframe_not_modified(data, config):
    """Input DataFrame must not be mutated during preparation."""
    original = data.copy(deep=True)
    result = prepare_generator_data(data, decision(), config)
    result.training_data.loc[0, "age"] = 99
    pd.testing.assert_frame_equal(data, original)


def test_affected_slice_row_count_mismatch_rejected(data, config):
    """Reject inconsistent affected-slice row counts."""
    readiness = ReadinessDecision(
        status=ReadinessStatus.REVIEW,
        generation_ready=True,
        supervised_retraining_ready=False,
        current_rows=10,
        affected_slice_rows=len(data) + 1,
    )
    with pytest.raises(
        DataPreparationError,
        match="affected-slice count",
    ):
        prepare_generator_data(
            data,
            readiness,
            config,
            source="affected_slice",
        )

