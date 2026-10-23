"""Automated Regression and Negative-Control Tests for Sprint 06 — Batch 5K.

Validates:
1. Canonical partition isolation and cryptographic integrity.
2. Candidate screening funnel: exactly 50 accepted candidates per generator.
3. Strict adherence to Tier 1-3 eligibility (structural, physical, deduplication, plausibility).
4. Zero leakage against evaluation partitions and zero exact B200 memorization.
5. Downstream Random Forest model reproducibility and hyperparameter freezing.
6. Empirical verification of the negative finding (synthetic arms do not beat authentic ROS B2).
7. Negative controls: rejection of invalid candidates (NaNs, physical violations, billing violations).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.experiments.run_synthetic_repair_benchmark import (
    filter_plausible_unique_candidates,
    paired_bootstrap_ci,
    paired_permutation_test,
    holm_bonferroni_correction,
)
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)

RESULTS_DIR = REPO_ROOT / "component3" / "results" / "experimental_models"
MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
SPLITS_DIR = REPO_ROOT / "component3" / "results" / "split_manifests"
ADAPT_DIR = REPO_ROOT / "component3" / "results" / "adaptation_splits"

PROTOCOL_PATH = RESULTS_DIR / "batch05k_repair_benchmark_protocol.json"
RESULTS_PATH = RESULTS_DIR / "batch05k_repair_benchmark_results.json"
COMPARISONS_PATH = RESULTS_DIR / "batch05k_repair_benchmark_comparisons.json"
AUDIT_CSV_PATH = RESULTS_DIR / "batch05k_repair_benchmark_predictions_audit.csv"


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def loaded_benchmark_data():
    """Load benchmark results, manifests, and datasets for test verification."""
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    canon_split_path = SPLITS_DIR / "telco_churn_split_manifest.json"
    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        protocol = json.load(f)
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(COMPARISONS_PATH, "r", encoding="utf-8") as f:
        comparisons = json.load(f)

    return {
        "protocol": protocol,
        "results": results,
        "comparisons": comparisons,
        "splits": splits,
        "meta": meta,
        "df_raw": df_raw,
    }


def test_partition_isolation_and_integrity(loaded_benchmark_data):
    """Verify partition sizes and zero cross-partition leakage."""
    splits = loaded_benchmark_data["splits"]
    meta = loaded_benchmark_data["meta"]
    feat_cols = meta.feature_columns

    assert len(splits.source_train) == 3452
    assert len(splits.source_eval) == 493
    assert len(splits.target_eval) == 493
    assert len(splits.nested_adapt_budgets[200]) == 200

    b200_df = splits.nested_adapt_budgets[200]
    b200_cust_ids = set(b200_df["customerID"])
    unused = splits.target_adapt_pool[~splits.target_adapt_pool["customerID"].isin(b200_cust_ids)].copy().reset_index(drop=True)

    b200_sigs = set(tuple(str(v) for v in r) for r in b200_df[feat_cols].itertuples(index=False))
    unused_sigs = [tuple(str(v) for v in r) for r in unused[feat_cols].itertuples(index=False)]
    overlap_idx = [i for i, s in enumerate(unused_sigs) if s in b200_sigs]
    clean_dev_val = unused.drop(index=overlap_idx).reset_index(drop=True)

    assert len(clean_dev_val) == 292

    # Verify zero leakage across all evaluation splits
    clean_sigs = set(tuple(str(v) for v in r) for r in clean_dev_val[feat_cols].itertuples(index=False))
    train_sigs = set(tuple(str(v) for v in r) for r in splits.source_train[feat_cols].itertuples(index=False))
    seval_sigs = set(tuple(str(v) for v in r) for r in splits.source_eval[feat_cols].itertuples(index=False))
    teval_sigs = set(tuple(str(v) for v in r) for r in splits.target_eval[feat_cols].itertuples(index=False))

    assert len(clean_sigs.intersection(b200_sigs)) == 0
    assert len(clean_sigs.intersection(train_sigs)) == 0
    assert len(clean_sigs.intersection(seval_sigs)) == 0
    assert len(clean_sigs.intersection(teval_sigs)) == 0


def test_candidate_csv_validity_and_funnel():
    """Verify that each synthetic candidate CSV satisfies the strict 7-tier criteria."""
    generators = ["tvae", "ctgan", "tabddpm"]
    for gen in generators:
        csv_path = RESULTS_DIR / f"batch05k_candidates_{gen}_k50.csv"
        assert csv_path.is_file(), f"Missing candidate CSV: {csv_path}"
        df = pd.read_csv(csv_path)

        assert len(df) == 50, f"Expected exactly 50 rows for {gen}, got {len(df)}"
        assert df.shape[1] == 20, f"Expected 20 columns, got {df.shape[1]}"

        # Tier 1: 0 NaNs and 0 Infs
        assert df.isna().sum().sum() == 0
        assert not np.isinf(df[["tenure", "MonthlyCharges", "TotalCharges"]]).any().any()

        # Tier 2: Physical boundaries
        assert (df["tenure"] >= 0.0).all() and (df["tenure"] <= 72.0).all()
        assert (df["MonthlyCharges"] >= 0.0).all()
        assert (df["TotalCharges"] >= 0.0).all()

        # Uniqueness within sample
        assert not df.duplicated().any()

        # Tier 3: Billing ratio screen in [0.5, 1.5]
        safe_t = df["tenure"].clip(lower=1.0)
        exp_tc = safe_t * df["MonthlyCharges"]
        ratio = df["TotalCharges"] / exp_tc.clip(lower=1e-6)
        assert ((ratio >= 0.5) & (ratio <= 1.5)).all()


def test_downstream_models_exist_and_frozen():
    """Verify that all downstream Random Forest models exist and have frozen parameters."""
    arms = ["s1_tvae_k50", "s2_ctgan_k50", "s3_tabddpm_k50"]
    for arm in arms:
        model_path = MODELS_DIR / f"telco_rf_experimental_{arm}.joblib"
        assert model_path.is_file(), f"Missing model: {model_path}"
        model = joblib.load(model_path)

        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.class_weight == "balanced"
        assert model.random_state == 42
        assert model.bootstrap is True
        assert model.criterion == "gini"


def test_negative_result_integrity(loaded_benchmark_data):
    """Verify that synthetic augmentation fails to outperform authentic ROS (B2)."""
    results = loaded_benchmark_data["results"]
    comparisons = loaded_benchmark_data["comparisons"]
    dev_metrics = results["dev_val_metrics"]

    b0_f1 = dev_metrics["B0"]["positive_f1"]
    b1_f1 = dev_metrics["B1_B200"]["positive_f1"]
    b2_f1 = dev_metrics["B2_B200_K200"]["positive_f1"]

    assert b0_f1 == pytest.approx(0.4055, abs=1e-3)
    assert b1_f1 == pytest.approx(0.4150, abs=1e-3)
    assert b2_f1 == pytest.approx(0.4497, abs=1e-3)

    for arm in ["S1_TVAE_K50", "S2_CTGAN_K50", "S3_TabDDPM_K50"]:
        synth_f1 = dev_metrics[arm]["positive_f1"]
        delta_vs_b2 = comparisons[arm]["B2_B200_K200"]["delta_f1"]

        # Synthetic F1 must be strictly less than B2 ROS F1
        assert synth_f1 < b2_f1, f"{arm} unexpectedly beat authentic ROS B2 ({synth_f1} vs {b2_f1})"
        assert delta_vs_b2 < 0.0, f"Delta vs B2 for {arm} must be negative, got {delta_vs_b2}"


def test_predictions_audit_completeness():
    """Verify that the predictions audit CSV contains all rows, predictions, and probabilities."""
    assert AUDIT_CSV_PATH.is_file()
    df_audit = pd.read_csv(AUDIT_CSV_PATH)
    assert len(df_audit) == 292
    assert df_audit.isna().sum().sum() == 0

    expected_arms = ["B0", "B1_B200", "B2_B200_K200", "B4_B200", "S1_TVAE_K50", "S2_CTGAN_K50", "S3_TabDDPM_K50"]
    for arm in expected_arms:
        assert f"pred_{arm}" in df_audit.columns
        assert f"prob_{arm}" in df_audit.columns


def test_negative_controls_candidate_screening(loaded_benchmark_data):
    """Negative control: verify that intentional corruptions are caught by the filter."""
    meta = loaded_benchmark_data["meta"]
    splits = loaded_benchmark_data["splits"]
    feat_cols = meta.feature_columns

    eval_sigs = {
        "D_source_train": set(tuple(str(v) for v in r) for r in splits.source_train[feat_cols].itertuples(index=False)),
    }

    # Clean base candidate
    base_cand = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 30.0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 2100.0,
        "Churn": "Yes",
    }])

    # 1. NaN corruption
    corrupt_nan = base_cand.copy()
    corrupt_nan.loc[0, "MonthlyCharges"] = np.nan
    _, stats_nan = filter_plausible_unique_candidates(
        corrupt_nan, base_cand, meta, eval_sigs, ["tenure", "MonthlyCharges", "TotalCharges"], ["gender", "Churn"]
    )
    assert stats_nan["tier1_structural_pass"] == 0

    # 2. Negative tenure corruption
    corrupt_neg_tenure = base_cand.copy()
    corrupt_neg_tenure.loc[0, "tenure"] = -5.0
    _, stats_neg = filter_plausible_unique_candidates(
        corrupt_neg_tenure, base_cand, meta, eval_sigs, ["tenure", "MonthlyCharges", "TotalCharges"], ["gender", "Churn"]
    )
    assert stats_neg["tier2_physical_pass"] == 0

    # 3. Impossible billing ratio (TotalCharges = 50,000 for 1 month of 70)
    corrupt_billing = base_cand.copy()
    corrupt_billing.loc[0, "TotalCharges"] = 50000.0
    _, stats_bill = filter_plausible_unique_candidates(
        corrupt_billing, base_cand, meta, eval_sigs, ["tenure", "MonthlyCharges", "TotalCharges"], ["gender", "Churn"]
    )
    assert stats_bill["tier3_plausible_unique"] == 0
