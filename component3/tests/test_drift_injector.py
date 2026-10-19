"""Tests for Controlled Drift Injection, Statistical Divergence, and Degradation Measurement."""

import pytest
import pandas as pd
import numpy as np

from component3.src.datasets.dataset_registry import DatasetMetadata, DatasetStatus
from component3.src.drift.drift_config import DriftCategory, DriftScenarioConfig, DriftSeverityTier
from component3.src.drift.drift_injector import DriftInjector, DriftInjectionResult


@pytest.fixture
def sample_dataset():
    """Create reproducible synthetic tabular dataset."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "num1": rng.normal(50, 10, n),
        "num2": rng.normal(100, 20, n),
        "cat1": rng.choice(["A", "B", "C"], n),
        "cat2": rng.choice(["Low", "Med", "High"], n),
        "target": rng.choice([0, 1], p=[0.7, 0.3], size=n),
    })
    meta = DatasetMetadata(
        dataset_id="mock_data",
        display_name="Mock Tabular",
        version="1.0.0",
        source="unit_test",
        license_info="CC0",
        status=DatasetStatus.VERIFIED,
        target_column="target",
        target_mapping={0: 0, 1: 1},
        numerical_features=["num1", "num2"],
        categorical_features=["cat1", "cat2"],
        class_distribution={0: 0.7, 1: 0.3},
    )
    return df, meta


def test_numerical_distribution_shift(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_NUM",
        dataset_id="mock_data",
        category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
        target_features=["num1"],
        injected_strength=0.8,
        slice_rule="num1 > 50",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)

    assert res.scenario_id == "TEST_NUM"
    assert res.measured_drift_score > 0.3
    assert "num1" in res.feature_ks_statistics
    assert len(res.drifted_df) == len(df)


def test_categorical_distribution_shift(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_CAT",
        dataset_id="mock_data",
        category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
        target_features=["cat1"],
        injected_strength=0.7,
        slice_rule="cat1 == 'A'",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)

    assert res.measured_drift_score > 0.1
    assert "cat1" in res.feature_tvd_statistics


def test_missingness_shift(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_MISS",
        dataset_id="mock_data",
        category=DriftCategory.MISSINGNESS_SHIFT,
        target_features=["num2"],
        injected_strength=0.5,
        slice_rule="num1 > 50",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)

    assert res.drifted_df["num2"].isna().sum() > df["num2"].isna().sum()


def test_feature_correlation_shift(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_CORR",
        dataset_id="mock_data",
        category=DriftCategory.FEATURE_CORRELATION_SHIFT,
        target_features=["num1", "num2"],
        injected_strength=0.7,
        slice_rule="num1 > 50",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)

    assert res.measured_drift_score > 0.05


def test_class_prior_shift(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_PRIOR",
        dataset_id="mock_data",
        category=DriftCategory.CLASS_PRIOR_SHIFT,
        target_features=["target"],
        injected_strength=0.5,
        slice_rule="target == 1",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)

    drifted_ratio = res.drifted_df["target"].mean()
    assert drifted_ratio > 0.35


def test_genuine_concept_drift(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_CONCEPT",
        dataset_id="mock_data",
        category=DriftCategory.GENUINE_CONCEPT_DRIFT,
        target_features=["target"],
        injected_strength=0.6,
        slice_rule="num1 > 60",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)

    assert res.manifest["is_concept_drift"] is True
    assert res.manifest["label_provenance"] == "ground_truth_retained"


def test_diagnostic_manifest_structure(sample_dataset):
    df, meta = sample_dataset
    config = DriftScenarioConfig(
        scenario_id="TEST_MANIFEST",
        dataset_id="mock_data",
        category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
        target_features=["num1"],
        injected_strength=0.5,
        slice_rule="num1 > 50",
    )
    injector = DriftInjector(seed=42)
    res = injector.inject(df, config, metadata=meta)
    manifest = res.to_diagnostic_manifest()

    assert manifest["scenario_id"] == "TEST_MANIFEST"
    assert manifest["category"] == DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT.value
    assert "feature_ks" in manifest
    assert "slice_rows" in manifest
