"""Unit tests for TabDDPM generator metadata handling and input validation."""

import numpy as np
import pandas as pd
import pytest

from component3.src.generators.tabddpm_generator import TabDDPMGenerator


def make_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0],
            "race": [1, 2, 1],
            "sex": [0, 1, 0],
        }
    )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("epochs", 0),
        ("epochs", -1),
        ("epochs", 1.5),
        ("epochs", "10"),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 128.5),
        ("batch_size", "256"),
        ("num_timesteps", 0),
        ("num_timesteps", -1),
        ("num_timesteps", 50.5),
        ("num_timesteps", "100"),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
    ],
)
def test_invalid_hyperparameters(parameter, value):
    with pytest.raises(ValueError):
        TabDDPMGenerator(**{parameter: value})


@pytest.mark.parametrize(
    "metadata",
    [
        "race",
        ["race", "race"],
        ["race", ""],
        ["race", 123],
    ],
)
def test_invalid_discrete_metadata(metadata):
    with pytest.raises((TypeError, ValueError)):
        TabDDPMGenerator(
            epochs=1,
            discrete_columns=metadata,
        )


def test_default_metadata_is_none():
    generator = TabDDPMGenerator(epochs=1)
    assert generator.discrete_columns is None


def test_invalid_training_data_rejected():
    generator = TabDDPMGenerator(epochs=1)
    with pytest.raises(TypeError):
        generator.fit([1, 2, 3])


def test_missing_discrete_column_rejected():
    generator = TabDDPMGenerator(
        epochs=1,
        discrete_columns=["unknown_column"],
    )
    with pytest.raises(ValueError, match="Unknown discrete columns"):
        generator.fit(make_training_data())


def test_missing_values_rejected():
    data = make_training_data()
    data.loc[0, "age"] = np.nan
    generator = TabDDPMGenerator(
        epochs=1,
        discrete_columns=["race", "sex"],
    )
    with pytest.raises(ValueError):
        generator.fit(data)


def test_failed_fit_resets_previous_model_state():
    generator = TabDDPMGenerator(
        epochs=1,
        discrete_columns=["unknown_column"],
    )
    generator.is_fitted = True
    generator.model = object()
    generator.diffusion = object()
    with pytest.raises(ValueError, match="Unknown discrete columns"):
        generator.fit(make_training_data())
    assert generator.is_fitted is False
    assert generator.model is None
    assert generator.diffusion is None
