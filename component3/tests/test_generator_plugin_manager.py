"""Unit tests for the Component 3 Generator Plugin Manager."""

from __future__ import annotations

import pandas as pd
import pytest

from component3.src.generators.base import BaseGenerator
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.plugin_manager import (
    GeneratorPluginError,
    GeneratorPluginManager,
)
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.tvae_generator import TVAEGenerator


class DummyCustomGenerator(BaseGenerator):
    """Custom dummy generator for testing plugin registration."""

    def __init__(self, custom_param: int = 42):
        self.custom_param = custom_param
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        self.is_fitted = True

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        return pd.DataFrame({"dummy": [self.custom_param] * n_rows})


@pytest.fixture
def manager():
    return GeneratorPluginManager()


def test_default_generator_registry(manager):
    assert manager.available_generators() == (
        "ctgan",
        "tabddpm",
        "tvae",
    )


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("ctgan", CTGANGenerator),
        ("tvae", TVAEGenerator),
        ("tabddpm", TabDDPMGenerator),
    ],
)
def test_get_generator_class(manager, name, expected_class):
    assert manager.get_generator_class(name) is expected_class


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("ctgan", CTGANGenerator),
        ("tvae", TVAEGenerator),
        ("tabddpm", TabDDPMGenerator),
    ],
)
def test_create_generator_default(manager, name, expected_class):
    gen = manager.create(name)
    assert isinstance(gen, expected_class)
    assert isinstance(gen, BaseGenerator)


@pytest.mark.parametrize(
    "name,expected_class",
    [
        (" CTGAN ", CTGANGenerator),
        ("tVaE", TVAEGenerator),
        (" TabDDPM ", TabDDPMGenerator),
    ],
)
def test_create_case_insensitive(manager, name, expected_class):
    gen = manager.create(name)
    assert isinstance(gen, expected_class)


@pytest.mark.parametrize(
    "name",
    ["ctgan", "tvae", "tabddpm"],
)
def test_create_with_discrete_columns(manager, name):
    gen = manager.create(name, epochs=1, discrete_columns=["workclass", "race"])
    assert list(gen.discrete_columns) == ["workclass", "race"]


def test_create_tabddpm_custom_parameters(manager):
    gen = manager.create(
        "tabddpm",
        epochs=2,
        batch_size=128,
        num_timesteps=20,
        learning_rate=0.005,
    )
    assert isinstance(gen, TabDDPMGenerator)
    assert gen.epochs == 2
    assert gen.batch_size == 128
    assert gen.num_timesteps == 20
    assert gen.learning_rate == 0.005


@pytest.mark.parametrize(
    "name,invalid_kwargs",
    [
        ("ctgan", {"num_timesteps": 10}),
        ("ctgan", {"batch_size": 128}),
        ("tvae", {"num_timesteps": 10}),
        ("tvae", {"learning_rate": 0.01}),
        ("tabddpm", {"unknown_param": 10}),
    ],
)
def test_unsupported_parameters_rejected(manager, name, invalid_kwargs):
    with pytest.raises(GeneratorPluginError):
        manager.create(name, **invalid_kwargs)


@pytest.mark.parametrize(
    "bad_name",
    ["unknown", "diffusion", "", "   ", None],
)
def test_unknown_or_invalid_name_rejected(manager, bad_name):
    with pytest.raises(GeneratorPluginError):
        manager.get_generator_class(bad_name)


def test_register_custom_generator(manager):
    manager.register("dummy", DummyCustomGenerator)
    assert "dummy" in manager.available_generators()
    gen = manager.create("dummy", custom_param=99)
    assert isinstance(gen, DummyCustomGenerator)
    assert gen.custom_param == 99


def test_register_duplicate_name_raises(manager):
    with pytest.raises(GeneratorPluginError):
        manager.register("ctgan", DummyCustomGenerator)


def test_register_with_override_succeeds(manager):
    manager.register("ctgan", DummyCustomGenerator, override=True)
    gen = manager.create("ctgan", custom_param=77)
    assert isinstance(gen, DummyCustomGenerator)
    assert gen.custom_param == 77


def test_register_invalid_class_raises(manager):
    class NotAGenerator:
        pass

    with pytest.raises(GeneratorPluginError):
        manager.register("invalid", NotAGenerator)
