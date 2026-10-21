"""Focused tests for CTGAN categorical metadata handling."""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from component3.src.generators.ctgan_generator import CTGANGenerator


@pytest.fixture
def encoded_data():
    return pd.DataFrame({
        "age": [25.0, 35.0, 45.0],
        "workclass": [1, 2, 1],
        "race": [2, 3, 2],
    })


def test_explicit_categorical_columns_passed_to_ctgan(encoded_data):
    """Explicit discrete_columns list must be forwarded to CTGAN.fit."""
    generator = CTGANGenerator(
        epochs=1,
        discrete_columns=["workclass", "race"],
    )
    generator.model = MagicMock()
    generator.fit(encoded_data)
    generator.model.fit.assert_called_once()
    args, kwargs = generator.model.fit.call_args
    pd.testing.assert_frame_equal(args[0], encoded_data)
    assert kwargs["discrete_columns"] == ["workclass", "race"]
    assert generator.is_fitted is True


def test_tuple_discrete_columns_supported(encoded_data):
    """Tuple discrete_columns must be accepted and forwarded to CTGAN.fit."""
    generator = CTGANGenerator(
        epochs=1,
        discrete_columns=("workclass", "race"),
    )
    generator.model = MagicMock()
    generator.fit(encoded_data)
    generator.model.fit.assert_called_once()
    _, kwargs = generator.model.fit.call_args
    assert kwargs["discrete_columns"] == ["workclass", "race"]
    assert generator.is_fitted is True


def test_default_fallback_without_discrete_columns():
    """When discrete_columns is None, infer object/category/bool columns."""
    df = pd.DataFrame({
        "age": [25.0, 35.0],
        "workclass": ["Private", "Local-gov"],
        "is_citizen": [True, False],
    })
    generator = CTGANGenerator(epochs=1)
    generator.model = MagicMock()
    generator.fit(df)
    generator.model.fit.assert_called_once()
    _, kwargs = generator.model.fit.call_args
    assert sorted(kwargs["discrete_columns"]) == ["is_citizen", "workclass"]
    assert generator.is_fitted is True


def test_invalid_discrete_columns_rejected():
    """Invalid types, empty strings, and duplicate column names must raise ValueError."""
    with pytest.raises(ValueError, match="list or tuple"):
        CTGANGenerator(epochs=1, discrete_columns="not_a_list")

    with pytest.raises(ValueError, match="unique"):
        CTGANGenerator(epochs=1, discrete_columns=["workclass", "workclass"])

    with pytest.raises(ValueError, match="unique"):
        CTGANGenerator(epochs=1, discrete_columns=["workclass", ""])

    with pytest.raises(ValueError, match="unique"):
        CTGANGenerator(epochs=1, discrete_columns=[123])


def test_missing_discrete_column_raises_value_error(encoded_data):
    """Configured discrete column not present in DataFrame must raise ValueError."""
    generator = CTGANGenerator(
        epochs=1,
        discrete_columns=["workclass", "non_existent"],
    )
    generator.model = MagicMock()
    with pytest.raises(ValueError, match="missing from data"):
        generator.fit(encoded_data)
    assert generator.is_fitted is False


def test_training_data_not_modified(encoded_data):
    """The input DataFrame must not be mutated during training."""
    original = encoded_data.copy(deep=True)
    generator = CTGANGenerator(
        epochs=1,
        discrete_columns=["workclass", "race"],
    )
    generator.model = MagicMock()
    generator.fit(encoded_data)
    pd.testing.assert_frame_equal(encoded_data, original)


def test_failed_retraining_clears_fitted_state(encoded_data):
    """A failed retraining attempt must leave is_fitted as False."""
    generator = CTGANGenerator(
        epochs=1,
        discrete_columns=["workclass", "race"],
    )
    generator.model = MagicMock()
    generator.fit(encoded_data)
    assert generator.is_fitted is True

    generator.discrete_columns = ["missing_column"]
    with pytest.raises(ValueError):
        generator.fit(encoded_data)
    assert generator.is_fitted is False


def test_sampling_calls_underlying_model(encoded_data):
    """Sampling after fit must delegate to underlying CTGAN model.sample."""
    generator = CTGANGenerator(
        epochs=1,
        discrete_columns=["workclass", "race"],
    )
    generator.model = MagicMock()
    generator.fit(encoded_data)

    mock_sample = pd.DataFrame({"age": [20.0], "workclass": [1], "race": [2]})
    generator.model.sample.return_value = mock_sample

    sampled = generator.sample(1)
    generator.model.sample.assert_called_once_with(1)
    pd.testing.assert_frame_equal(sampled, mock_sample)
