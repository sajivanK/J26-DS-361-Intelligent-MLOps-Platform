"""Tests for validated offline scenario specifications."""

import pytest
from pydantic import ValidationError

from c4.data import FEATURE_COLUMNS
from c4.scenarios import ScenarioConfig


@pytest.mark.parametrize(
    "settings",
    [
        {
            "mechanism": "none",
            "magnitude": 0.0,
        },
        {
            "mechanism": "marginal_single",
            "magnitude": 0.3,
            "features": ("age",),
        },
        {
            "mechanism": "marginal_multi",
            "magnitude": 0.3,
            "features": ("age", "education_num"),
        },
        {
            "mechanism": "interaction",
            "magnitude": 0.3,
            "interaction_pairs": (("education_num", "occupation"),),
        },
        {
            "mechanism": "combined",
            "magnitude": 0.3,
            "features": ("age",),
            "interaction_pairs": (("education_num", "occupation"),),
        },
    ],
)
def test_valid_mechanisms_are_accepted(settings):
    config = ScenarioConfig(seed=42, **settings)

    assert config.dataset == "uci_adult"
    assert config.seed == 42


@pytest.mark.parametrize(
    "settings",
    [
        {"mechanism": "none", "magnitude": 0.3},
        {
            "mechanism": "none",
            "magnitude": 0.0,
            "features": ("age",),
        },
        {
            "mechanism": "marginal_single",
            "magnitude": 0.0,
            "features": ("age",),
        },
        {
            "mechanism": "marginal_single",
            "magnitude": 0.3,
            "features": ("target",),
        },
        {
            "mechanism": "marginal_single",
            "magnitude": 0.3,
            "features": ("age", "education_num"),
        },
        {
            "mechanism": "marginal_multi",
            "magnitude": 0.3,
            "features": ("age", "age"),
        },
        {
            "mechanism": "interaction",
            "magnitude": 0.3,
            "interaction_pairs": (("age", "age"),),
        },
        {
            "mechanism": "interaction",
            "magnitude": 0.3,
            "interaction_pairs": (
                ("age", "occupation"),
                ("occupation", "age"),
            ),
        },
        {
            "mechanism": "combined",
            "magnitude": 0.3,
            "features": ("age",),
            "interaction_pairs": (("age", "occupation"),),
        },
    ],
)
def test_invalid_injection_settings_are_rejected(settings):
    with pytest.raises(ValidationError):
        ScenarioConfig(seed=42, **settings)


@pytest.mark.parametrize("magnitude", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_magnitudes_are_rejected(magnitude):
    with pytest.raises(ValidationError):
        ScenarioConfig(
            mechanism="marginal_single",
            magnitude=magnitude,
            features=("age",),
            seed=42,
        )


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValidationError):
        ScenarioConfig(mechanism="none", magnitude=0.0, seed=seed)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ScenarioConfig(
            mechanism="none",
            magnitude=0.0,
            seed=42,
            future_margin=0.1,
        )


def test_configuration_is_frozen():
    config = ScenarioConfig(mechanism="none", magnitude=0.0, seed=42)

    with pytest.raises(ValidationError):
        config.seed = 99


def test_hash_is_stable_after_json_round_trip():
    config = ScenarioConfig(
        mechanism="interaction",
        magnitude=0.3,
        interaction_pairs=(("education_num", "occupation"),),
        seed=42,
    )
    restored = ScenarioConfig.model_validate_json(config.model_dump_json())

    assert config.configuration_sha256() == restored.configuration_sha256()
    assert len(config.configuration_sha256()) == 64


def test_changing_seed_changes_configuration_hash():
    first = ScenarioConfig(mechanism="none", magnitude=0.0, seed=42)
    second = ScenarioConfig(mechanism="none", magnitude=0.0, seed=43)

    assert first.configuration_sha256() != second.configuration_sha256()


def test_feature_types_match_processed_adult_schema():
    config = ScenarioConfig(mechanism="none", magnitude=0.0, seed=42)
    types = config.feature_types()

    assert set(types) == set(FEATURE_COLUMNS)
    assert types["age"] == "continuous"
    assert types["marital_status"] == "categorical"
    assert types["occupation"] == "categorical"
    assert "target" not in types