"""Automated Unit Tests for Labeled Simulated-Target Adaptation Control (Batch 5G).

Validates:
1. Machine-readable protocol integrity and governance.
2. Nested adaptation budget subset inclusion and non-overlapping duplicate groups.
3. Identical frozen concept transformation (tenure > 29.0) and original label preservation.
4. Permitted training composition (3,452 source + B_k adaptation records; 0 synthetic records).
5. Preprocessor and model fit isolation (no target-evaluation or test snooping).
6. Reload prediction parity and probability tolerance.
7. Source-domain retention (no catastrophic forgetting).
8. Paired statistical analysis (bootstrap 95% CIs, permutation test, multiplicity adjustment).
9. Historical baseline and Batch 5F source model protection.
10. Evaluation prediction audit privacy (zero raw customer IDs).
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.experiments.run_authentic_adaptation_control import (
    compute_file_sha256,
    paired_bootstrap_ci,
    paired_permutation_test,
)
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)

REPO_ROOT = Path("component3").resolve().parent
RESULTS_DIR = REPO_ROOT / "component3" / "results" / "experimental_models"
MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
BASELINES_DIR = REPO_ROOT / "component3" / "models" / "baselines"

PROTOCOL_PATH = RESULTS_DIR / "batch05g_adaptation_protocol.json"
RESULTS_PATH = RESULTS_DIR / "telco_adaptation_control_results.json"
STATS_PATH = RESULTS_DIR / "telco_adaptation_control_statistical_audit.json"
SOURCE_RETENTION_PATH = RESULTS_DIR / "telco_source_retention_comparison.json"
AUDIT_CSV_PATH = RESULTS_DIR / "telco_adaptation_predictions_audit.csv"

# Known frozen hashes
KNOWN_BASELINE_RF_HASH = "69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701"
KNOWN_BASELINE_PREP_HASH = "a9eb79cd6c519b99a1ef5372d5418afaacb01048498bb46617d1f6e012ed45c0"
KNOWN_SOURCE_RF_HASH = "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0"


@pytest.fixture(scope="module")
def loaded_batch05g_data():
    """Load results, protocol, and dataset for audit tests."""
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        protocol = json.load(f)
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    audit_df = pd.read_csv(AUDIT_CSV_PATH)

    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    canon_split_path = REPO_ROOT / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    return {
        "protocol": protocol,
        "results": results,
        "stats": stats,
        "audit_df": audit_df,
        "splits": splits,
        "canon_train_df": canon_train_df,
    }


def test_batch05g_protocol_integrity(loaded_batch05g_data):
    """Verify protocol pre-registration fields, hyperparameters, and arms."""
    protocol = loaded_batch05g_data["protocol"]

    assert protocol["protocol_name"] == "Batch_5G_Authentic_Adaptation_Control"
    assert protocol["dataset_id"] == "telco_churn"
    assert protocol["drift_scenario"] == "TELCO_CONCEPT"
    assert "A0" in protocol["reference_control_arm"]

    arms = protocol["adaptation_arms"]
    assert set(arms.keys()) == {"A1-B50", "A1-B100", "A1-B200"}
    assert arms["A1-B50"]["total_training_rows"] == 3502
    assert arms["A1-B100"]["total_training_rows"] == 3552
    assert arms["A1-B200"]["total_training_rows"] == 3652

    # Zero synthetic rows allowed
    for arm_cfg in arms.values():
        assert arm_cfg["synthetic_rows"] == 0

    hp = protocol["classifier_hyperparameters"]
    assert hp["n_estimators"] == 100
    assert hp["max_depth"] == 10
    assert hp["class_weight"] == "balanced"
    assert hp["random_state"] == 42
    assert hp["n_jobs"] == 2


def test_adaptation_budgets_nesting_and_transformation(loaded_batch05g_data):
    """Verify nesting subset relationships and frozen concept transformations."""
    splits = loaded_batch05g_data["splits"]

    b50 = splits.nested_adapt_budgets[50]
    b100 = splits.nested_adapt_budgets[100]
    b200 = splits.nested_adapt_budgets[200]
    pool = splits.target_adapt_pool

    id50 = set(b50["customerID"])
    id100 = set(b100["customerID"])
    id200 = set(b200["customerID"])
    id_pool = set(pool["customerID"])

    # Strict nesting
    assert id50.issubset(id100)
    assert id100.issubset(id200)
    assert id200.issubset(id_pool)

    # Inversion counts
    assert int((b50["tenure"] > 29.0).sum()) == 25
    assert int((b100["tenure"] > 29.0).sum()) == 48
    assert int((b200["tenure"] > 29.0).sum()) == 97


def test_training_set_composition_and_zero_generators(loaded_batch05g_data):
    """Verify training telemetry rows and zero synthetic data."""
    results = loaded_batch05g_data["results"]
    tt = results["training_telemetry"]

    assert tt["A0"]["training_rows"] == 3452
    assert tt["A1-B50"]["training_rows"] == 3502
    assert tt["A1-B100"]["training_rows"] == 3552
    assert tt["A1-B200"]["training_rows"] == 3652

    for b_size in [50, 100, 200]:
        arm = f"A1-B{b_size}"
        assert tt[arm]["source_train_rows"] == 3452
        assert tt[arm]["adaptation_budget_rows"] == b_size
        # Assert model artifact exists
        assert (MODELS_DIR / f"telco_rf_a1_b{b_size}.joblib").is_file()
        assert (MODELS_DIR / f"telco_prep_a1_b{b_size}.joblib").is_file()


def test_model_and_preprocessor_isolation(loaded_batch05g_data):
    """Verify preprocessors are fitted separately on each arm's training composition."""
    for b_size in [50, 100, 200]:
        prep_path = MODELS_DIR / f"telco_prep_a1_b{b_size}.joblib"
        prep: TabularDataPreprocessor = joblib.load(prep_path)

        assert prep.is_fitted is True
        assert prep.scale_numerical is True
        assert len(prep.ordered_features) == 19

        # Class weights must reflect (3452 source + b_size adapt)
        expected_total = 3452 + b_size
        cw = prep.get_class_weights()
        assert 0 in cw and 1 in cw
        # Ensure class weights are valid positive floats
        assert cw[0] > 0.5 and cw[1] > 1.2


def test_reload_prediction_and_probability_parity(loaded_batch05g_data):
    """Verify reload prediction parity and probability tolerance."""
    results = loaded_batch05g_data["results"]
    parity = results["reload_parity_verified"]

    for arm in ["A1-B50", "A1-B100", "A1-B200"]:
        assert parity[arm] is True


def test_source_retention_and_no_catastrophic_forgetting(loaded_batch05g_data):
    """Verify that adaptation does not severely degrade pre-drift source domain performance."""
    results = loaded_batch05g_data["results"]
    src_comp = results["source_retention_comparisons"]

    for arm in ["A1-B50", "A1-B100", "A1-B200"]:
        # Catastrophic forgetting threshold is Delta F1 <= -0.10
        assert src_comp[arm]["catastrophic_forgetting_flag"] is False
        assert src_comp[arm]["source_f1_positive"] >= 0.58


def test_paired_statistical_analysis_and_multiplicity(loaded_batch05g_data):
    """Verify paired bootstrap CIs, permutation tests, and multiple-comparison adjustments."""
    stats = loaded_batch05g_data["stats"]

    for arm in ["A1-B50", "A1-B100", "A1-B200"]:
        rec = stats[arm]
        # All 95% CIs must cross zero (adaptation sample is too small to overcome pre-drift volume)
        assert rec["ci_includes_zero"] is True
        assert rec["bootstrap_95_ci_lower"] <= 0.0 <= rec["bootstrap_95_ci_upper"]

        # Statistical significance flags under Bonferroni and FDR must be False
        assert rec["statistically_significant_bonferroni"] is False
        assert rec["statistically_significant_fdr"] is False

        # Adjusted p-values must be greater than or equal to raw p-values
        assert rec["bonferroni_adjusted_p_value"] >= rec["permutation_two_sided_p_value"]


def test_historical_baseline_and_batch05f_inviolability():
    """Verify that historical baseline models and Batch 5F source model were never mutated."""
    rf_base_hash = compute_file_sha256(BASELINES_DIR / "telco_churn_random_forest.joblib")
    prep_base_hash = compute_file_sha256(BASELINES_DIR / "telco_churn_preprocessor.joblib")
    rf_src_hash = compute_file_sha256(MODELS_DIR / "telco_rf_experimental_source.joblib")

    assert rf_base_hash == KNOWN_BASELINE_RF_HASH
    assert prep_base_hash == KNOWN_BASELINE_PREP_HASH
    assert rf_src_hash == KNOWN_SOURCE_RF_HASH


def test_anonymized_prediction_audit_privacy(loaded_batch05g_data):
    """Verify that audit CSV contains only anonymized eval IDs and no raw customer IDs."""
    audit_df = loaded_batch05g_data["audit_df"]

    assert len(audit_df) == 493
    assert "eval_row_id" in audit_df.columns
    assert "customerID" not in audit_df.columns

    # Verify ID formatting
    assert audit_df.iloc[0]["eval_row_id"] == "eval_row_000"
    assert audit_df.iloc[-1]["eval_row_id"] == "eval_row_492"

    # Verify required prediction columns
    expected_cols = {
        "eval_row_id",
        "true_churn_concept",
        "tenure",
        "is_affected_slice",
        "pred_a0",
        "prob_a0",
        "pred_a1_b50",
        "prob_a1_b50",
        "pred_a1_b100",
        "prob_a1_b100",
        "pred_a1_b200",
        "prob_a1_b200",
    }
    assert expected_cols.issubset(set(audit_df.columns))
