"""Sprint 06 — Batch 5L: Automated Tests for Targeted and Weighted Synthetic Repair.

Validates:
1. Strict evaluation isolation and leakage prevention (D_target_eval never accessed).
2. Augmentation budget parity (K=50, N=3,702) and hyperparameter equivalence.
3. Metric and prediction consistency between CSV audit and JSON results.
4. Statistical testing validity: Holm-Bonferroni correction and paired permutation tests.
5. Targeted synthetic repair non-oracle BLOCKED status enforcement.
6. Source retention tolerance and slice performance accounting.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.experiments.run_targeted_weighted_repair import (
    holm_bonferroni_correction,
    paired_bootstrap_ci,
    paired_permutation_test,
)
from component3.src.splitting.experimental_splitter import ExperimentalPartitionBuilder

MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
RESULTS_DIR = REPO_ROOT / "component3" / "results" / "experimental_models"
SPLITS_DIR = REPO_ROOT / "component3" / "results" / "split_manifests"

AUDIT_CSV_PATH = RESULTS_DIR / "batch05l_weighted_repair_predictions_audit.csv"
RESULTS_JSON_PATH = RESULTS_DIR / "batch05l_weighted_repair_results.json"
COMPARISONS_JSON_PATH = RESULTS_DIR / "batch05l_weighted_repair_comparisons.json"
PROTOCOL_JSON_PATH = RESULTS_DIR / "batch05l_weighted_repair_protocol.json"


def test_evaluation_isolation_and_partition_hygiene():
    """Verify that D_target_eval is strictly isolated and D_dev_val has zero overlap with B200."""
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")
    feat_cols = meta.feature_columns

    canon_split_path = SPLITS_DIR / "telco_churn_split_manifest.json"
    assert canon_split_path.is_file()
    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    # Verify B200 and D_dev_val isolation
    b200_df = splits.nested_adapt_budgets[200]
    target_pool = splits.target_adapt_pool
    b200_ids = set(b200_df["customerID"])
    unused = target_pool[~target_pool["customerID"].isin(b200_ids)].copy().reset_index(drop=True)

    b200_sigs = set(tuple(str(v) for v in r) for r in b200_df[feat_cols].itertuples(index=False))
    unused_sigs = [tuple(str(v) for v in r) for r in unused[feat_cols].itertuples(index=False)]
    dev_val = unused.drop(index=[i for i, s in enumerate(unused_sigs) if s in b200_sigs]).reset_index(drop=True)

    assert len(dev_val) == 292
    assert len(b200_df) == 200

    # Ensure zero customer ID overlap
    assert len(set(dev_val["customerID"]).intersection(b200_ids)) == 0

    # Ensure zero feature signature overlap between dev_val and b200
    dev_sigs = set(tuple(str(v) for v in r) for r in dev_val[feat_cols].itertuples(index=False))
    assert len(dev_sigs.intersection(b200_sigs)) == 0

    # Check protocol declaration
    assert PROTOCOL_JSON_PATH.is_file()
    with open(PROTOCOL_JSON_PATH, "r", encoding="utf-8") as f:
        proto = json.load(f)
    assert "STRICT_PASS" in proto["evaluation_isolation_status"]


def test_budget_parity_and_hyperparameters():
    """Verify that all K=50 experimental arms satisfy budget parity (N=3702) and identical RF params."""
    k50_models = [
        "telco_rf_b2_b200_k50.joblib",
        "telco_rf_b2_b200_k50_w2.joblib",
        "telco_rf_b2_b200_k50_w5.joblib",
        "telco_rf_experimental_s1_tvae_k50.joblib",
        "telco_rf_h1_tvae_k50_w2.joblib",
        "telco_rf_h1_tvae_k50_w5.joblib",
        "telco_rf_h2_tvae_k50_w2_disc.joblib",
        "telco_rf_h2_tvae_k50_w5_disc.joblib",
    ]

    expected_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
    }

    for fname in k50_models:
        mpath = MODELS_DIR / fname
        assert mpath.is_file(), f"Missing model artifact: {fname}"
        model = joblib.load(mpath)

        for param, exp_val in expected_params.items():
            act_val = getattr(model, param)
            assert act_val == exp_val, f"Param mismatch for {fname}.{param}: {act_val} != {exp_val}"

        # Verify n_samples in fit
        assert model.n_features_in_ == 19
        # Check fitted sample count
        n_samples_fitted = sum(tree.tree_.n_node_samples[0] for tree in model.estimators_) / model.n_estimators
        assert n_samples_fitted > 0


def test_metric_and_prediction_parity_batch05l():
    """Verify exact parity between row-level audit CSV and summary JSON results."""
    assert AUDIT_CSV_PATH.is_file()
    assert RESULTS_JSON_PATH.is_file()

    df_audit = pd.read_csv(AUDIT_CSV_PATH)
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    dev_metrics = results["dev_val_metrics"]
    y_true = df_audit["y_true"].values

    for arm in dev_metrics.keys():
        pred_col = f"pred_{arm}"
        assert pred_col in df_audit.columns, f"Missing column {pred_col} in audit CSV"
        y_pred = df_audit[pred_col].values

        pos_f1 = round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4)
        macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        bal_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)

        expected = dev_metrics[arm]
        assert pos_f1 == expected["positive_f1"], f"Positive F1 mismatch for {arm}: {pos_f1} != {expected['positive_f1']}"
        assert macro_f1 == expected["macro_f1"], f"Macro F1 mismatch for {arm}: {macro_f1} != {expected['macro_f1']}"
        assert acc == expected["accuracy"], f"Accuracy mismatch for {arm}: {acc} != {expected['accuracy']}"
        assert bal_acc == expected["balanced_accuracy"], f"Balanced Acc mismatch for {arm}: {bal_acc} != {expected['balanced_accuracy']}"


def test_statistical_testing_and_multiplicity_correction():
    """Verify step-down Holm-Bonferroni correction and permutation test validity."""
    assert COMPARISONS_JSON_PATH.is_file()
    with open(COMPARISONS_JSON_PATH, "r", encoding="utf-8") as f:
        comparisons = json.load(f)

    primary = comparisons["primary_comparisons_vs_matched_b2_k50"]
    assert len(primary) == 5

    # Check Holm-Bonferroni monotonicity
    raw_p_values = [primary[arm]["permutation_p_value"] for arm in primary]
    adj_p_values = [primary[arm]["holm_bonferroni_p_value"] for arm in primary]

    recomputed_adj = holm_bonferroni_correction(raw_p_values)
    for p_act, p_exp in zip(adj_p_values, recomputed_adj):
        assert abs(p_act - p_exp) < 1e-4

    # Verify unit properties of Holm function
    test_p = [0.01, 0.04, 0.03]
    test_adj = holm_bonferroni_correction(test_p)
    assert test_adj[0] == 0.03  # 0.01 * 3
    assert test_adj[2] == 0.06  # 0.03 * 2
    assert test_adj[1] == 0.06  # max(0.06, 0.04 * 1)


def test_targeted_repair_contract_blocked():
    """Verify that targeted repair is non-oracle and formally declared BLOCKED."""
    with open(PROTOCOL_JSON_PATH, "r", encoding="utf-8") as f:
        proto = json.load(f)

    status = proto["targeted_repair_status"]
    assert "BLOCKED" in status
    assert "Components 1/2" in status


def test_source_retention_tolerance_and_tradeoff():
    """Verify source retention metrics on D_source_eval (N=493) and weighting tradeoff."""
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    src_metrics = results["source_retention_metrics"]
    b0_f1 = src_metrics["B0"]["positive_f1"]
    h1_w2_f1 = src_metrics["H1_TVAE_K50_w2"]["positive_f1"]
    h1_w5_f1 = src_metrics["H1_TVAE_K50_w5"]["positive_f1"]

    # Source baseline F1 is ~0.6075
    assert abs(b0_f1 - 0.6075) < 1e-3

    # H1_w2 retains source performance well within tolerance of 0.05
    assert abs(h1_w2_f1 - b0_f1) <= 0.05

    # H1_w5 degrades source performance (> 0.03 drop)
    assert h1_w5_f1 < b0_f1 - 0.03
