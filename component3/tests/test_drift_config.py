"""Unit tests for Component 3 Drift Experiment Configuration."""

import pytest

from component3.src.drift.drift_config import (
    DriftCategory,
    DriftScenarioConfig,
    DriftSeverityTier,
    get_drift_registry,
)


def test_drift_registry_default_scenarios():
    reg = get_drift_registry()
    scenarios = reg.list_scenarios()
    for s_id in ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08_CONCEPT_ADULT"]:
        assert s_id in scenarios


def test_all_eight_drift_categories_defined():
    expected_categories = {
        "numerical_distribution_shift",
        "categorical_distribution_shift",
        "mixed_feature_shift",
        "missingness_shift",
        "feature_correlation_shift",
        "class_prior_shift",
        "covariate_shift",
        "genuine_concept_drift",
    }
    actual_categories = {c.value for c in DriftCategory}
    assert actual_categories == expected_categories


def test_prohibition_of_labeling_feature_shift_as_concept_drift():
    # Scientific guardrail: A marginal feature shift P(X) cannot be labeled as concept drift P(Y|X)
    with pytest.raises(ValueError) as exc_info:
        DriftScenarioConfig(
            scenario_id="INVALID_DRIFT_01",
            dataset_id="uci_adult",
            category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
            target_features=["age"],
            injected_strength=0.5,
            slice_rule="age > 0.69",
            is_concept_drift=True,  # Illegal!
        )
    assert "cannot be flagged as is_concept_drift=True" in str(exc_info.value)


def test_valid_genuine_concept_drift_scenario():
    cfg = DriftScenarioConfig(
        scenario_id="VALID_CONCEPT_DRIFT",
        dataset_id="uci_adult",
        category=DriftCategory.GENUINE_CONCEPT_DRIFT,
        target_features=["education_num"],
        injected_strength=0.6,
        slice_rule="education_num > 1.0",
        is_concept_drift=True,
        severity_tier=DriftSeverityTier.CRITICAL,
        measured_severity_metric="conditional_label_shift",
        measured_severity_value=0.32,
    )
    assert cfg.is_concept_drift is True
    assert cfg.category == DriftCategory.GENUINE_CONCEPT_DRIFT


def test_injected_strength_bounds():
    with pytest.raises(ValueError):
        DriftScenarioConfig(
            scenario_id="OUT_OF_BOUNDS_STRENGTH",
            dataset_id="uci_adult",
            category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
            target_features=["race"],
            injected_strength=1.5,  # > 1.0
            slice_rule="race == 2",
        )


def test_drift_config_serialization(tmp_path):
    reg = get_drift_registry()
    s04 = reg.get("S04")

    save_path = tmp_path / "s04_drift.json"
    s04.save(save_path)
    assert save_path.is_file()

    loaded_s04 = DriftScenarioConfig.load(save_path)
    assert loaded_s04.scenario_id == "S04"
    assert loaded_s04.category == DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT
    assert loaded_s04.target_features == ["race"]
