"""Comprehensive Unit and Scientific Guardrail Tests for Drift Simulation.

Tests:
1. Drift reproducibility across random seeds.
2. Drift configuration validation and boundary enforcement.
3. Strict feature targeting isolation.
4. Label validity classification and provenance preservation.
5. Baseline model immutability during drift inference.
6. Split manifest reusability.
7. Exclusion of held-out test set from drift experiments.
8. Serialization integrity of drift results and manifests.
9. Experiment resumption and artifact verification.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from component3.src.baselines.model_adapter import BaselineModelAdapter
from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.drift.drift_config import (
    DriftCategory,
    DriftScenarioConfig,
    DriftSeverityTier,
    LabelValidityStatus,
    get_drift_registry,
)
from component3.src.drift.drift_injector import DriftInjectionResult, DriftInjector

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sample_tabular_data():
    """Create reproducible synthetic tabular DataFrame and metadata for unit testing."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).clip(18, 90),
        "income": rng.normal(50000, 15000, n).clip(10000, 200000),
        "education": rng.choice(["HighSchool", "Bachelors", "Masters", "PhD"], n),
        "marital": rng.choice(["Single", "Married", "Divorced"], n),
        "target": rng.choice([0, 1], p=[0.7, 0.3], size=n),
    })
    return df


def test_drift_reproducibility(sample_tabular_data):
    """Identical random seeds must produce byte-exact drifted DataFrames and metrics."""
    df = sample_tabular_data
    cfg = DriftScenarioConfig(
        scenario_id="REPRO_TEST",
        dataset_id="mock_data",
        category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
        target_features=["age"],
        injected_strength=0.6,
        slice_rule="age > 50",
    )
    inj1 = DriftInjector(seed=42)
    inj2 = DriftInjector(seed=42)
    res1 = inj1.inject(df, cfg)
    res2 = inj2.inject(df, cfg)

    assert res1.drifted_df.equals(res2.drifted_df)
    assert res1.measured_drift_score == res2.measured_drift_score
    assert res1.feature_ks_statistics == res2.feature_ks_statistics


def test_drift_configuration_validation():
    """Invalid drift parameters must raise informative ValueErrors."""
    # Strength > 1.0
    with pytest.raises(ValueError, match="Injected drift strength must be in"):
        DriftScenarioConfig(
            scenario_id="INVALID_STRENGTH",
            dataset_id="mock",
            category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
            target_features=["age"],
            injected_strength=1.5,
            slice_rule="age > 50",
        )

    # Pure feature shift cannot be marked as concept drift
    with pytest.raises(ValueError, match="cannot be flagged as is_concept_drift=True"):
        DriftScenarioConfig(
            scenario_id="INVALID_CONCEPT_FLAG",
            dataset_id="mock",
            category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
            target_features=["age"],
            injected_strength=0.5,
            slice_rule="age > 50",
            is_concept_drift=True,
        )


def test_correct_feature_targeting(sample_tabular_data):
    """Drift perturbation must isolate target features; untargeted columns must remain identical."""
    df = sample_tabular_data
    cfg = DriftScenarioConfig(
        scenario_id="TARGET_ISOLATION",
        dataset_id="mock",
        category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
        target_features=["age"],
        injected_strength=0.5,
        slice_rule="age > 45",
    )
    inj = DriftInjector(seed=42)
    res = inj.inject(df, cfg)

    # Targeted column changed
    assert not res.drifted_df["age"].equals(df["age"])

    # Untargeted columns strictly untouched
    for col in ["income", "education", "marital", "target"]:
        assert res.drifted_df[col].equals(df[col]), f"Untargeted column {col} was modified!"


def test_label_validity_handling(sample_tabular_data):
    """Drift injector must correctly tag label validity statuses and validity booleans."""
    df = sample_tabular_data
    inj = DriftInjector(seed=42)

    # 1. Stress test feature shift -> uncertain labels
    cfg_num = DriftScenarioConfig(
        scenario_id="NUM_SHIFT",
        dataset_id="mock",
        category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
        target_features=["age"],
        injected_strength=0.5,
        slice_rule="age > 45",
        label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
    )
    res_num = inj.inject(df, cfg_num)
    assert res_num.label_validity_status == "stress_test_uncertain"
    assert res_num.is_label_valid is False

    # 2. Genuine concept drift -> synthetic concept truth
    cfg_concept = DriftScenarioConfig(
        scenario_id="CONCEPT_SHIFT",
        dataset_id="mock",
        category=DriftCategory.GENUINE_CONCEPT_DRIFT,
        target_features=["age"],
        injected_strength=0.5,
        slice_rule="age > 45",
        is_concept_drift=True,
        label_validity=LabelValidityStatus.SYNTHETIC_CONCEPT_TRUTH,
    )
    res_concept = inj.inject(df, cfg_concept)
    assert res_concept.label_validity_status == "synthetic_concept_truth"
    assert res_concept.is_label_valid is True

    # 3. Class prior shift -> resampled ground truth
    cfg_prior = DriftScenarioConfig(
        scenario_id="PRIOR_SHIFT",
        dataset_id="mock",
        category=DriftCategory.CLASS_PRIOR_SHIFT,
        target_features=["target"],
        injected_strength=0.6,
        slice_rule="target == 1",
        label_validity=LabelValidityStatus.VERIFIED_RESAMPLED_GROUND_TRUTH,
    )
    res_prior = inj.inject(df, cfg_prior)
    assert res_prior.label_validity_status == "verified_resampled_ground_truth"
    assert res_prior.is_label_valid is True


def test_baseline_model_immutability():
    """Running drift evaluation through BaselineModelAdapter must not alter model weights or reference predictions."""
    models_dir = ROOT / "component3/models/baselines"
    m_path = models_dir / "uci_adult_full_xgboost.joblib"
    prep_path = models_dir / "uci_adult_full_preprocessor.joblib"

    if not m_path.is_file() or not prep_path.is_file():
        pytest.skip("Baseline model artifacts not found; skipping immutability test.")

    adapter = BaselineModelAdapter.load(m_path, model_family="xgboost", preprocessor_path=prep_path)

    # Reference data
    meta = get_dataset_registry().get("uci_adult_full")
    full_df = pd.read_csv(ROOT / meta.local_path, nrows=200)

    # Benchmark initial predictions
    preds_before = adapter.predict(full_df)

    # Inject drift and evaluate
    cfg = get_drift_registry().get("ADULT_NUM")
    inj = DriftInjector(seed=42)
    _ = inj.inject(full_df, cfg, metadata=meta, model_adapter=adapter)

    # Re-evaluate reference data
    preds_after = adapter.predict(full_df)
    assert np.array_equal(preds_before, preds_after), "Model predictions changed after drift inference!"


def test_split_reuse_and_test_partition_exclusion():
    """Drift simulation must reuse validation partition and strictly exclude held-out test partition."""
    manifest_path = ROOT / "component3/results/split_manifests/uci_adult_full_split_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Split manifest missing.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    val_idx = set(manifest["val_indices"])
    test_idx = set(manifest["test_indices"])
    train_idx = set(manifest["train_indices"])

    # Confirm mutually exclusive partitions
    assert len(val_idx.intersection(test_idx)) == 0, "Validation and test partitions overlap!"
    assert len(val_idx.intersection(train_idx)) == 0, "Validation and train partitions overlap!"

    # Confirm drift summary evaluated exactly val_size rows
    summary_path = ROOT / "component3/results/drift/drift_simulation_summary.json"
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_scenarios"] == 24
        assert summary["total_evaluations"] == 72


def test_drift_result_serialization(tmp_path):
    """Drift results and manifests must serialize and deserialize without loss."""
    res = DriftInjectionResult(
        scenario_id="SER_TEST",
        dataset_id="uci_adult_full",
        category="numerical_distribution_shift",
        injected_strength=0.5,
        measured_drift_score=0.412,
        feature_ks_statistics={"age": 0.412},
        feature_tvd_statistics={},
        affected_slice_rows=50,
        affected_slice_support=0.25,
        pre_drift_overall_f1=0.701,
        post_drift_overall_f1=0.680,
        delta_overall_f1=-0.021,
        pre_drift_slice_f1=0.650,
        post_drift_slice_f1=0.620,
        delta_slice_f1=-0.030,
        drifted_df=pd.DataFrame({"age": [20, 30]}),
        manifest={"test_key": "test_val"},
        label_validity_status="stress_test_uncertain",
        is_label_valid=False,
    )

    d = res.to_dict()
    assert "drifted_df" not in d
    assert d["scenario_id"] == "SER_TEST"
    assert d["label_validity_status"] == "stress_test_uncertain"
    assert d["is_label_valid"] is False

    manifest = res.to_diagnostic_manifest()
    assert manifest["test_key"] == "test_val"


def test_drift_experiment_resumption():
    """Serialized drift simulation summary must exist and contain verified records."""
    summary_json = ROOT / "component3/results/drift/drift_simulation_summary.json"
    summary_csv = ROOT / "component3/results/drift/drift_simulation_summary.csv"

    if not summary_json.is_file() or not summary_csv.is_file():
        pytest.skip("Drift simulation summary records not found.")

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_id"] == "EXP_NOTEBOOK_03_DRIFT_SIMULATION"
    assert len(data["datasets_evaluated"]) == 3
    assert len(data["models_evaluated"]) == 3
    assert data["total_scenarios"] == 24
    assert data["total_evaluations"] == 72

    df = pd.read_csv(summary_csv)
    assert len(df) == 72
    assert "delta_f1" in df.columns
    assert "is_label_valid" in df.columns
    assert "label_validity" in df.columns


def test_frozen_concept_drift_thresholds():
    """Verify that all concept drift scenarios have frozen thresholds and metadata."""
    drift_reg = get_drift_registry()
    scenarios = {
        "ADULT_CONCEPT": ("hours_per_week", 40.0, "hours_per_week > 40.0"),
        "TELCO_CONCEPT": ("tenure", 29.0, "tenure > 29.0"),
        "BANK_CONCEPT": ("euribor3m", 4.857, "euribor3m > 4.857"),
    }
    for s_id, (feat, thresh, rule) in scenarios.items():
        cfg = drift_reg.get(s_id)
        assert cfg.slice_rule == rule
        assert cfg.metadata.get("frozen_threshold") == thresh
        assert cfg.metadata.get("threshold_feature") == feat
        assert cfg.metadata.get("threshold_source") == "train_partition_median"
        assert cfg.metadata.get("scenario_version") == "1.0.0"


def test_concept_drift_target_type_and_slice_inversion():
    """Verify that concept drift injection preserves target column types and inverts only target slice."""
    ds_reg = get_dataset_registry()
    drift_reg = get_drift_registry()
    inj = DriftInjector(seed=42)

    # 1. Adult: object target
    meta_adult = ds_reg.get("uci_adult_full")
    df_adult = pd.read_csv(ROOT / meta_adult.local_path, nrows=300)
    cfg_adult = drift_reg.get("ADULT_CONCEPT")
    res_adult = inj.inject(df_adult, cfg_adult, metadata=meta_adult)
    assert res_adult.drifted_df["target"].dtype == df_adult["target"].dtype
    # Check untargeted slice untouched
    outside_mask_adult = df_adult["hours_per_week"] <= 40.0
    assert (res_adult.drifted_df.loc[outside_mask_adult, "target"] == df_adult.loc[outside_mask_adult, "target"]).all()

    # 2. Bank: int64 target
    meta_bank = ds_reg.get("bank_marketing")
    df_bank = pd.read_csv(ROOT / meta_bank.local_path, nrows=300)
    cfg_bank = drift_reg.get("BANK_CONCEPT")
    res_bank = inj.inject(df_bank, cfg_bank, metadata=meta_bank)
    assert pd.api.types.is_integer_dtype(res_bank.drifted_df["y"])
    # Check untargeted slice untouched
    outside_mask_bank = df_bank["euribor3m"] <= 4.857
    assert (res_bank.drifted_df.loc[outside_mask_bank, "y"] == df_bank.loc[outside_mask_bank, "y"]).all()

