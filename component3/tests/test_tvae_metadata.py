"""Unit tests for TVAE generator metadata handling and input validation."""

from unittest.mock import patch
import pandas as pd
import pytest

from component3.src.generators.tvae_generator import TVAEGenerator


@pytest.fixture
def training_data():
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0],
            "race": [1, 2, 1],
            "sex": [0, 1, 0],
        }
    )


def test_explicit_discrete_columns(training_data):
    with patch("component3.src.generators.tvae_generator.TVAE") as mock_tvae:
        generator = TVAEGenerator(
            epochs=1,
            discrete_columns=["race", "sex"],
        )
        generator.fit(training_data)
        assert (
            mock_tvae.return_value.fit.call_args.kwargs["discrete_columns"]
            == ["race", "sex"]
        )


def test_tuple_discrete_columns(training_data):
    with patch("component3.src.generators.tvae_generator.TVAE") as mock_tvae:
        generator = TVAEGenerator(
            epochs=1,
            discrete_columns=("race", "sex"),
        )
        generator.fit(training_data)
        assert (
            mock_tvae.return_value.fit.call_args.kwargs["discrete_columns"]
            == ["race", "sex"]
        )


def test_dtype_fallback():
    data = pd.DataFrame(
        {
            "age": [25.0, 30.0],
            "category": ["A", "B"],
        }
    )
    with patch("component3.src.generators.tvae_generator.TVAE") as mock_tvae:
        generator = TVAEGenerator(epochs=1)
        generator.fit(data)
        assert (
            mock_tvae.return_value.fit.call_args.kwargs["discrete_columns"]
            == ["category"]
        )


def test_invalid_discrete_columns():
    with pytest.raises(ValueError):
        TVAEGenerator(epochs=1, discrete_columns="race")
    with pytest.raises(ValueError):
        TVAEGenerator(epochs=1, discrete_columns=["race", "race"])
    with pytest.raises(ValueError):
        TVAEGenerator(epochs=1, discrete_columns=["race", 123])


def test_missing_discrete_column(training_data):
    with patch("component3.src.generators.tvae_generator.TVAE"):
        generator = TVAEGenerator(
            epochs=1,
            discrete_columns=["nonexistent"],
        )
        with pytest.raises(ValueError, match="not found"):
            generator.fit(training_data)


def test_training_data_is_deep_copied(training_data):
    with patch("component3.src.generators.tvae_generator.TVAE") as mock_tvae:
        generator = TVAEGenerator(
            epochs=1,
            discrete_columns=["race", "sex"],
        )
        original = training_data.copy(deep=True)
        generator.fit(training_data)
        passed_data = mock_tvae.return_value.fit.call_args.args[0]
        assert passed_data is not training_data
        assert training_data.equals(original)


def test_failed_retraining_clears_fitted_state(training_data):
    with patch("component3.src.generators.tvae_generator.TVAE"):
        generator = TVAEGenerator(epochs=1)
        generator.fit(training_data)
        assert generator.is_fitted is True

        generator.discrete_columns = ["nonexistent"]
        with pytest.raises(ValueError):
            generator.fit(training_data)
        assert generator.is_fitted is False


def test_sample_delegates_to_model(training_data):
    with patch("component3.src.generators.tvae_generator.TVAE") as mock_tvae:
        generator = TVAEGenerator(epochs=1)
        generator.fit(training_data)
        mock_tvae.return_value.sample.return_value = pd.DataFrame(
            {"age": [25.0], "race": [1], "sex": [0]}
        )
        sampled = generator.sample(5)
        mock_tvae.return_value.sample.assert_called_once_with(5)
        assert len(sampled) == 1
