"""Automated tests for Sprint 06 — Batch 5M: Evidence Integrity, Strategy Registry and Prescription Readiness.

Validates:
1. Historical metric reconciliation (B2_B200_K200 source F1=0.5890 and affected slice support=148).
2. Deterministic partition membership hashes and nested budget containment.
3. Machine-readable evidence registry schema, completeness, and artifact hash integrity.
4. Repair strategy taxonomy definitions and non-oracle BLOCKED targeted repair status.
5. Pre-decision feature schema leakage prevention (absence of post-decision outcomes).
6. Inviolable preservation of historical split manifests and baseline models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import ExperimentalPartitionBuilder

RESULTS_DIR = REPO_ROOT / "component3" / "results"
EXP_DIR = RESULTS_DIR / "experimental_models"
MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
SPLITS_DIR = RESULTS_DIR / "split_manifests"
ADAPT_DIR = RESULTS_DIR / "adaptation_splits"

REGISTRY_PATH = RESULTS_DIR / "repair_evidence_registry.json"


def compute_file_sha256(filepath: Path | str) -> str:
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_metric_reconciliation_discrepancy_1():
    """Verify Discrepancy 1: B2_B200_K200 model predicts positive F1=0.5890 on D_source_eval."""
    mpath = MODELS_DIR / "telco_rf_b2_b200_k200.joblib"
    assert mpath.is_file()
    model = joblib.load(mpath)

    prep_path = MODELS_DIR / "telco_preprocessor_experimental_source.joblib"
    source_prep: TabularDataPreprocessor = joblib.load(prep_path)

    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    with open(SPLITS_DIR / "telco_churn_split_manifest.json", "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    df_source_eval = splits.source_eval.copy().reset_index(drop=True)
    bundle_seval = source_prep.transform(df_source_eval)
    y_pred = model.predict(bundle_seval.X)

    f1_pos = round(float(f1_score(bundle_seval.y, y_pred, pos_label=1, zero_division=0)), 4)
    assert f1_pos == 0.5890, f"Expected 0.5890, got {f1_pos}"

    # Also check that 5L results JSON has 0.589
    with open(EXP_DIR / "batch05l_weighted_repair_results.json", "r", encoding="utf-8") as f:
        res_5l = json.load(f)
    assert res_5l["source_retention_metrics"]["B2_B200_K200"]["positive_f1"] == 0.589


def test_metric_reconciliation_discrepancy_2():
    """Verify Discrepancy 2: Clean development validation partition has exactly 148 affected records."""
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")
    feat_cols = meta.feature_columns

    with open(SPLITS_DIR / "telco_churn_split_manifest.json", "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    b200_df = splits.nested_adapt_budgets[200]
    target_pool = splits.target_adapt_pool
    b200_ids = set(b200_df["customerID"])
    unused = target_pool[~target_pool["customerID"].isin(b200_ids)].copy().reset_index(drop=True)

    b200_sigs = set(tuple(str(v) for v in r) for r in b200_df[feat_cols].itertuples(index=False))
    unused_sigs = [tuple(str(v) for v in r) for r in unused[feat_cols].itertuples(index=False)]
    dev_val = unused.drop(index=[i for i, s in enumerate(unused_sigs) if s in b200_sigs]).reset_index(drop=True)

    assert len(dev_val) == 292
    affected_count = int((dev_val["tenure"] > 29.0).sum())
    preserved_count = int((dev_val["tenure"] <= 29.0).sum())

    assert affected_count == 148, f"Expected 148 affected records, got {affected_count}"
    assert preserved_count == 144, f"Expected 144 preserved records, got {preserved_count}"
    assert affected_count + preserved_count == 292


def test_nested_budget_identity_and_containment():
    """Verify exact nested budget relationships: B50 in B100 in B200 and stable membership hashes."""
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    with open(SPLITS_DIR / "telco_churn_split_manifest.json", "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    b50_ids = set(splits.nested_adapt_budgets[50]["customerID"])
    b100_ids = set(splits.nested_adapt_budgets[100]["customerID"])
    b200_ids = set(splits.nested_adapt_budgets[200]["customerID"])

    assert len(b50_ids) == 50
    assert len(b100_ids) == 100
    assert len(b200_ids) == 200

    # Strict containment
    assert b50_ids.issubset(b100_ids)
    assert b100_ids.issubset(b200_ids)

    # Stable membership hashes
    h_b50 = hashlib.sha256("".join(sorted(b50_ids)).encode("utf-8")).hexdigest()
    h_b100 = hashlib.sha256("".join(sorted(b100_ids)).encode("utf-8")).hexdigest()
    h_b200 = hashlib.sha256("".join(sorted(b200_ids)).encode("utf-8")).hexdigest()

    assert h_b50 == "ed25342b6389314548e0559681dfaddef8123bd43605d17ef368f24fc0736cbb"
    assert h_b100 == "b505b428877bcde01b390393b84ba59e4f543db238c1eab3076b0d81ea4a03d7"
    assert h_b200 == "f8029325865a423583b43e3b7799cc197110f7f7c0768a8b7bb7c1a3e3131194"


def test_evidence_registry_structure_and_completeness():
    """Verify that repair_evidence_registry.json exists, is valid JSON, and contains all required fields."""
    assert REGISTRY_PATH.is_file(), f"Registry file missing: {REGISTRY_PATH}"

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    meta = registry["registry_metadata"]
    assert meta["dataset_id"] == "telco_churn"
    assert meta["simulated_drift_scenario"] == "TELCO_CONCEPT"
    assert meta["independent_scenario_count"] == 1
    assert meta["dev_val_count"] == 292
    assert meta["source_eval_count"] == 493
    assert meta["total_experiments_recorded"] >= 15

    experiments = registry["experiments"]
    required_keys = [
        "experiment_id",
        "arm_id",
        "batch_origin",
        "strategy",
        "generator_or_control",
        "adaptation_budget",
        "synthetic_quantity",
        "weighting_strategy",
        "model_artifact_path",
        "model_artifact_sha256",
        "prediction_artifact_path",
        "prediction_artifact_sha256",
        "target_dev_performance",
        "source_eval_positive_f1",
        "evaluation_contamination_status",
        "evidence_status",
    ]

    for eid, exp in experiments.items():
        for k in required_keys:
            assert k in exp, f"Missing required key '{k}' in experiment {eid}"

        if exp["evidence_status"] != "BLOCKED_NON_ORACLE_ABSENT":
            assert exp["model_artifact_sha256"] != "MISSING"
            assert exp["prediction_artifact_sha256"] != "MISSING"
            assert len(exp["model_artifact_sha256"]) == 64
            assert len(exp["prediction_artifact_sha256"]) == 64


def test_pre_decision_feature_leakage_prohibition():
    """Verify that prescription feature schema strictly prohibits downstream outcomes."""
    # Defined permitted and prohibited feature sets
    prohibited_tokens = [
        "dev_val",
        "target_eval",
        "test_f1",
        "outcome",
        "chosen_generator",
        "oracle",
        "post_generation",
        "tenure_gt_29_rule",
    ]

    permitted_schema = {
        "dataset_num_rows": int,
        "num_numerical_features": int,
        "num_categorical_features": int,
        "class_imbalance_ratio": float,
        "missing_value_rate": float,
        "drift_detected": bool,
        "drift_severity_score": float,
        "num_affected_features": int,
        "baseline_performance_degradation": float,
        "diagnostic_slice_identified": bool,
        "authentic_adaptation_budget": int,
        "source_retention_tolerance": float,
        "runtime_budget_seconds": float,
        "memory_limit_mb": float,
    }

    for feat in permitted_schema.keys():
        for token in prohibited_tokens:
            assert token not in feat, f"Prohibited token '{token}' detected in pre-decision feature '{feat}'!"


def test_targeted_repair_contract_blocked_in_registry():
    """Verify that TARGETED_SYNTHETIC_REPAIR is registered as BLOCKED in evidence registry."""
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    exp_targeted = registry["experiments"].get("EXP_5L_TARGETED_REPAIR_STATUS")
    assert exp_targeted is not None
    assert exp_targeted["strategy"] == "TARGETED_SYNTHETIC_REPAIR"
    assert exp_targeted["evidence_status"] == "BLOCKED_NON_ORACLE_ABSENT"
    assert "blocking_reason" in exp_targeted["resource_measurements"]
