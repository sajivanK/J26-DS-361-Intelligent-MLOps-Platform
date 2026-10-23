"""Lightweight common-interface tests for C3 generators."""

import inspect
from unittest.mock import MagicMock
import pandas as pd
import pytest

from component3.src.generators.base import BaseGenerator
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tvae_generator import TVAEGenerator
from component3.src.generators.tabddpm_generator import TabDDPMGenerator

GENERATORS = [
    CTGANGenerator,
    TVAEGenerator,
    TabDDPMGenerator,
]


@pytest.mark.parametrize("generator_class", GENERATORS)
def test_inherits_common_interface(generator_class):
    """All generators must inherit from BaseGenerator and be concrete."""
    assert issubclass(generator_class, BaseGenerator)
    assert not inspect.isabstract(generator_class)


@pytest.mark.parametrize("generator_class", GENERATORS)
def test_fit_signature(generator_class):
    """All generators must expose fit(self, data)."""
    signature = inspect.signature(generator_class.fit)
    assert list(signature.parameters) == ["self", "data"]


@pytest.mark.parametrize("generator_class", GENERATORS)
def test_sample_signature(generator_class):
    """All generators must expose sample(self, n_rows)."""
    signature = inspect.signature(generator_class.sample)
    assert list(signature.parameters) == ["self", "n_rows"]


@pytest.mark.parametrize("generator_class", GENERATORS)
def test_sample_before_fit_raises_error(generator_class):
    """Attempting to sample before fitting must fail safely with RuntimeError."""
    generator = generator_class()
    with pytest.raises(RuntimeError):
        generator.sample(10)


@pytest.mark.parametrize("generator_class", GENERATORS)
def test_fit_empty_data_raises_error(generator_class):
    """Fitting on an empty DataFrame must raise ValueError."""
    generator = generator_class()
    with pytest.raises(ValueError):
        generator.fit(pd.DataFrame())


@pytest.mark.parametrize("generator_class", GENERATORS)
def test_sample_invalid_row_counts(generator_class):
    """Zero or negative sample counts must raise ValueError."""
    generator = generator_class()
    generator.is_fitted = True
    if hasattr(generator, "diffusion"):
        generator.diffusion = MagicMock()
    with pytest.raises(ValueError):
        generator.sample(0)
    with pytest.raises(ValueError):
        generator.sample(-10)


def test_base_generator_cannot_be_instantiated():
    """BaseGenerator is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseGenerator()


def test_base_generator_abstract_methods():
    """BaseGenerator must declare fit and sample as abstract methods."""
    assert BaseGenerator.__abstractmethods__ == frozenset({"fit", "sample"})
