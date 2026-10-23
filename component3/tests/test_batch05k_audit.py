"""Automated Audit and Regression Tests for Sprint 06 — Batch 5K-Audit.

Validates:
1. Metric and prediction parity between saved CSV audit and JSON results.
2. Arm-to-model mapping and provenance verification.
3. Augmentation-budget accounting: documenting K=50 vs K=200 mismatch and matched K=50 control.
4. Locked-evaluation isolation: verifying candidate filtering without D_target_eval access.
5. Statistical test correctness: finite-sample correction, two-sided permutation, and Holm adjustment.
6. Inviolable preservation of historical baseline, split manifests, and source models.
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
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

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
from component3.src.splitting.experimental_splitter import ExperimentalPartitionBuilder

RESULTS_DIR = REPO_ROOT / "component3" / "results" / "experimental_models"
MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
SPLITS_DIR = REPO_ROOT / "component3" / "results" / "split_manifests"
ADAPT_DIR = REPO_ROOT / "component3" / "results" / "adaptation_splits"
BASELINES_DIR = REPO_ROOT / "component3" / "models" / "baselines"

AUDIT_CSV_PATH = RESULTS_DIR / "batch05k_repair_benchmark_predictions_audit.csv"
RESULTS_JSON_PATH = RESULTS_DIR / "batch05k_repair_benchmark_results.json"
COMPARISONS_JSON_PATH = RESULTS_DIR / "batch05k_repair_benchmark_comparisons.json"
HISTORICAL_5I_JSON_PATH = RESULTS_DIR / "telco_authentic_controls_dev_metrics_corrected.json"


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_metric_and_prediction_parity():
    """Verify that recomputing metrics from batch05k_repair_benchmark_predictions_audit.csv exactly matches results.json."""
    assert AUDIT_CSV_PATH.is_file()
    assert RESULTS_JSON_PATH.is_file()

    df_audit = pd.read_csv(AUDIT_CSV_PATH)
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    dev_metrics = results["dev_val_metrics"]
    y_true = df_audit["y_true"].values

    all_arms = ["B0", "B1_B200", "B2_B200_K200", "B4_B200", "S1_TVAE_K50", "S2_CTGAN_K50", "S3_TabDDPM_K50"]

    for arm in all_arms:
        pred = df_audit[f"pred_{arm}"].values
        pos_f1 = round(float(f1_score(y_true, pred, pos_label=1, zero_division=0)), 4)
        macro_f1 = round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4)
        bal_acc = round(float(balanced_accuracy_score(y_true, pred)), 4)
        acc = round(float(accuracy_score(y_true, pred)), 4)

        expected = dev_metrics[arm]
        assert pos_f1 == expected["positive_f1"], f"F1 mismatch for {arm}: {pos_f1} vs {expected['positive_f1']}"
        assert macro_f1 == expected["macro_f1"], f"Macro F1 mismatch for {arm}: {macro_f1} vs {expected['macro_f1']}"
        assert bal_acc == expected["balanced_accuracy"], f"Bal Acc mismatch for {arm}: {bal_acc} vs {expected['balanced_accuracy']}"
        assert acc == expected["accuracy"], f"Accuracy mismatch for {arm}: {acc} vs {expected['accuracy']}"


def test_matched_k50_ros_control_provenance_and_comparison():
    """Verify that matched B2_B200_K50 control exists, was trained in Batch 5I, and exhibits F1=0.4028."""
    model_k50_path = MODELS_DIR / "telco_rf_b2_b200_k50.joblib"
    assert model_k50_path.is_file(), f"Expected saved control {model_k50_path} to exist!"

    model_k50 = joblib.load(model_k50_path)
    assert model_k50.n_estimators == 100
    assert model_k50.max_depth == 10
    assert model_k50.class_weight == "balanced"
    assert model_k50.random_state == 42

    with open(HISTORICAL_5I_JSON_PATH, "r", encoding="utf-8") as f:
        b5i = json.load(f)

    assert "B2_B200_K50" in b5i
    k50_f1 = b5i["B2_B200_K50"]["overall"]["positive_f1"]
    assert k50_f1 == 0.4028

    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    dev_metrics = results["dev_val_metrics"]

    # When matched on budget K=50, TVAE, CTGAN, and TabDDPM all match or exceed B2_B200_K50
    assert dev_metrics["S1_TVAE_K50"]["positive_f1"] > k50_f1
    assert dev_metrics["S2_CTGAN_K50"]["positive_f1"] > k50_f1
    assert dev_metrics["S3_TabDDPM_K50"]["positive_f1"] > k50_f1


def test_candidate_screening_isolation_without_target_eval():
    """Verify that candidate screening operates identically without accessing D_target_eval."""
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    with open(SPLITS_DIR / "telco_churn_split_manifest.json") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(canon_train_df, meta, adaptation_budgets=(50, 100, 200), seed=42)

    b200_df = splits.nested_adapt_budgets[200]
    b200_flips = b200_df["tenure"] > 29.0
    b200_post = b200_df.copy()
    b200_post.loc[b200_flips, "Churn"] = b200_post.loc[b200_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    source_prep = joblib.load(MODELS_DIR / "telco_preprocessor_experimental_source.joblib")
    bundle_b200 = source_prep.transform(b200_post)
    gen_input = bundle_b200.generator_df.copy()
    gen_input["Churn"] = b200_post["Churn"].values
    discrete_cols = bundle_b200.generator_discrete_columns + ["Churn"]
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Screening signatures using ONLY training-side data (D_source_train and B200)
    training_only_eval_sigs = {
        "D_source_train": set(tuple(str(v) for v in r) for r in splits.source_train[meta.feature_columns].itertuples(index=False)),
    }

    tvae_raw = pd.read_csv(REPO_ROOT / "component3" / "results" / "generators" / "batch05j_synthetic_candidates_tvae.csv")
    plausible_clean, stats_clean = filter_plausible_unique_candidates(
        df_cand=tvae_raw,
        df_train_b200=gen_input,
        meta=meta,
        eval_signatures=training_only_eval_sigs,
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
    )

    # Exactly 65 plausible candidates survive, identical to when D_target_eval was included
    assert len(plausible_clean) == 65
    assert stats_clean["tier3_plausible_unique"] == 65


def test_statistical_test_corrections():
    """Verify two-sided permutation test, finite-sample p-value correction, and Holm adjustment."""
    # 1. Finite sample p-value formula: (extreme_count + 1) / (n_permutations + 1)
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 10)
    cand_preds = y_true.copy()
    ref_preds = y_true.copy()

    perm_ident = paired_permutation_test(y_true, cand_preds, ref_preds, n_permutations=999, seed=42)
    assert perm_ident["observed_diff"] == 0.0
    assert perm_ident["p_value"] == 1.0  # All permutations >= 0, (1000) / (1000) = 1.0

    # 2. Step-down Holm-Bonferroni correction
    raw_p = [0.01, 0.04, 0.03, 0.20]
    # m = 4. Sorted: 0.01 (x4 -> 0.04), 0.03 (x3 -> 0.09), 0.04 (x2 -> 0.08, capped at 0.09), 0.20 (x1 -> 0.20)
    adj_p = holm_bonferroni_correction(raw_p)
    assert adj_p == [0.04, 0.09, 0.09, 0.20]


def test_historical_artifact_bitwise_preservation():
    """Verify that all historical baseline and control models are bitwise identical."""
    expected_hashes = {
        "historical_baseline_rf": (
            BASELINES_DIR / "telco_churn_random_forest.joblib",
            "69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701",
        ),
        "source_rf": (
            MODELS_DIR / "telco_rf_experimental_source.joblib",
            "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0",
        ),
        "source_prep": (
            MODELS_DIR / "telco_preprocessor_experimental_source.joblib",
            "a188f0285e6cf07741bbb69be8b3df44eece492e69166c78d92353f1f4d41224",
        ),
        "canonical_split": (
            SPLITS_DIR / "telco_churn_split_manifest.json",
            "e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68",
        ),
        "adaptation_split": (
            ADAPT_DIR / "telco_churn_adaptation_split_manifest.json",
            "34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59",
        ),
    }

    for name, (path, expected_h) in expected_hashes.items():
        actual_h = compute_file_sha256(path)
        assert actual_h == expected_h, f"Mutation detected in {name}: expected {expected_h}, got {actual_h}"
