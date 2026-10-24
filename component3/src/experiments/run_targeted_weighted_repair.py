"""Sprint 06 — Batch 5L: Targeted and Domain-Weighted Synthetic Repair Pilot.

Evaluates domain-weighted hybrid repair (H1, H2) using TVAE synthetic data (K=50)
under joint and discounted weighting policies against matched authentic controls:
- B0: Frozen unadapted source baseline (N=3,452)
- B1_B200: Naive authentic pooling (N=3,652, w=1)
- B2_B200_K50: Primary matched-budget authentic ROS control (N=3,702, w=1)
- B2_B200_K200: Extended authentic ROS reference (N=3,852, w=1)
- B5_B200_w2 / w5: Authentic weighted controls (N=3,652, w_target in {2, 5})
- B2_B200_K50_w2 / w5: Matched authentic weighted ROS controls (N=3,702, w_target=w_dup in {2, 5})
- S1_TVAE_K50: Unweighted synthetic TVAE arm (N=3,702, w=1)
- H1_TVAE_K50_w2 / w5: Joint hybrid weighting (w_target = w_synth in {2, 5})
- H2_TVAE_K50_w2_disc / w5_disc: Discounted synthetic weighting

Targeted repair is formally declared BLOCKED due to absence of an upstream C1/C2 diagnostic payload.
Evaluation isolation strictly enforced: D_target_eval is never loaded into memory.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def paired_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_candidate: np.ndarray,
    y_pred_reference: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Calculate paired bootstrap confidence intervals for positive-class F1 difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        f1_cand = f1_score(y_true[idx], y_pred_candidate[idx], pos_label=1, zero_division=0)
        f1_ref = f1_score(y_true[idx], y_pred_reference[idx], pos_label=1, zero_division=0)
        diffs[b] = f1_cand - f1_ref

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    return {
        "lower_ci": round(lower, 4),
        "upper_ci": round(upper, 4),
        "mean_diff": round(mean_diff, 4),
    }


def paired_permutation_test(
    y_true: np.ndarray,
    y_pred_candidate: np.ndarray,
    y_pred_reference: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Calculate paired two-sided permutation test p-value for positive-class F1 difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_diff = float(
        f1_score(y_true, y_pred_candidate, pos_label=1, zero_division=0)
        - f1_score(y_true, y_pred_reference, pos_label=1, zero_division=0)
    )

    perm_diffs = np.empty(n_permutations, dtype=np.float64)
    for r in range(n_permutations):
        swap = rng.random(n) < 0.5
        perm_cand = np.where(swap, y_pred_reference, y_pred_candidate)
        perm_ref = np.where(swap, y_pred_candidate, y_pred_reference)
        f1_c = f1_score(y_true, perm_cand, pos_label=1, zero_division=0)
        f1_r = f1_score(y_true, perm_ref, pos_label=1, zero_division=0)
        perm_diffs[r] = f1_c - f1_r

    extreme_count = int(np.sum(np.abs(perm_diffs) >= np.abs(obs_diff) - 1e-9))
    p_val = float((extreme_count + 1) / (n_permutations + 1))
    return {
        "observed_diff": round(obs_diff, 4),
        "p_value": round(p_val, 4),
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    adjusted_p = np.zeros(m, dtype=np.float64)
    running_max = 0.0

    for rank, idx in enumerate(sorted_indices):
        multiplier = m - rank
        raw_p = p_values[idx]
        adj = min(1.0, raw_p * multiplier)
        running_max = max(running_max, adj)
        adjusted_p[idx] = round(running_max, 4)

    return adjusted_p.tolist()


def compute_metrics_bundle(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Compute standard classification metrics bundle."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    f1_pos = float(f1_score(y, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y, y_pred, pos_label=1, zero_division=0))

    roc_auc = float(roc_auc_score(y, y_prob)) if y_prob is not None and len(np.unique(y)) > 1 else None
    pr_auc = float(average_precision_score(y, y_prob)) if y_prob is not None and len(np.unique(y)) > 1 else None
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        "positive_f1": round(f1_pos, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm,
        "positive_support": int(np.sum(y == 1)),
        "negative_support": int(np.sum(y == 0)),
        "total_support": len(y),
    }


def execute_targeted_weighted_repair(
    seed: int = 42,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Batch 5L targeted and weighted synthetic repair benchmark."""
    t_start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)

    root = Path(base_dir) if base_dir else Path("component3").resolve().parent
    models_dir = root / "component3" / "models" / "experimental"
    results_dir = root / "component3" / "results" / "experimental_models"
    figures_dir = root / "component3" / "results" / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Input Verification and Cryptographic Integrity
    # -------------------------------------------------------------------------
    canon_split_path = root / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    adapt_split_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
    source_model_path = root / "component3" / "models" / "experimental" / "telco_rf_experimental_source.joblib"
    source_prep_path = root / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib"
    tvae_cand_path = root / "component3" / "results" / "experimental_models" / "batch05k_candidates_tvae_k50.csv"

    expected_hashes = {
        "canonical_split": "e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68",
        "adaptation_split": "34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59",
        "source_model": "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0",
        "source_prep": "a188f0285e6cf07741bbb69be8b3df44eece492e69166c78d92353f1f4d41224",
    }

    verified_hashes = {}
    for name, expected_h in expected_hashes.items():
        p = {
            "canonical_split": canon_split_path,
            "adaptation_split": adapt_split_path,
            "source_model": source_model_path,
            "source_prep": source_prep_path,
        }[name]
        actual_h = compute_file_sha256(p)
        if actual_h != expected_h:
            raise ValueError(f"Hash mismatch for {name}: expected {expected_h}, got {actual_h}")
        verified_hashes[name] = actual_h

    # Load canonical dataset and rebuild splits
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")
    feat_cols = meta.feature_columns

    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=seed,
    )

    df_source_train = splits.source_train.copy().reset_index(drop=True)
    df_source_eval = splits.source_eval.copy().reset_index(drop=True)
    df_target_adapt_pool = splits.target_adapt_pool.copy().reset_index(drop=True)
    # STRICT EVALUATION ISOLATION: df_target_eval is NEVER loaded or accessed!

    # Isolate B200 and clean D_dev_val (N=292)
    b200_df = splits.nested_adapt_budgets[200].copy().reset_index(drop=True)
    b200_cust_ids = set(b200_df["customerID"])
    raw_unused = df_target_adapt_pool[~df_target_adapt_pool["customerID"].isin(b200_cust_ids)].copy().reset_index(drop=True)

    b200_sigs = set(tuple(str(v) for v in r) for r in b200_df[feat_cols].itertuples(index=False))
    unused_sigs = [tuple(str(v) for v in r) for r in raw_unused[feat_cols].itertuples(index=False)]
    overlap_idx = [i for i, s in enumerate(unused_sigs) if s in b200_sigs]
    clean_dev_val = raw_unused.drop(index=overlap_idx).copy().reset_index(drop=True)

    if len(clean_dev_val) != 292:
        raise ValueError(f"Expected 292 rows in clean dev val, got {len(clean_dev_val)}")

    # Apply exact TELCO_CONCEPT ground truth to B200 and D_dev_val
    b200_flips = b200_df["tenure"] > 29.0
    b200_post = b200_df.copy()
    b200_post.loc[b200_flips, "Churn"] = b200_post.loc[b200_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    dev_flips = clean_dev_val["tenure"] > 29.0
    clean_dev_val_post = clean_dev_val.copy()
    clean_dev_val_post.loc[dev_flips, "Churn"] = clean_dev_val_post.loc[dev_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    # Transform partitions using source preprocessor
    source_prep: TabularDataPreprocessor = joblib.load(source_prep_path)

    bundle_src = source_prep.transform(df_source_train)
    X_src, y_src = bundle_src.X, bundle_src.y

    bundle_b200 = source_prep.transform(b200_post)
    X_b200, y_b200 = bundle_b200.X, bundle_b200.y

    bundle_dev = source_prep.transform(clean_dev_val_post)
    X_dev, y_dev = bundle_dev.X, bundle_dev.y

    bundle_seval = source_prep.transform(df_source_eval)
    X_seval, y_seval = bundle_seval.X, bundle_seval.y

    # Sliced masks on D_dev_val
    mask_affected = dev_flips.values
    mask_preserved = (~dev_flips).values

    # -------------------------------------------------------------------------
    # STEP 2: Candidate Loading & Matched ROS Duplicates
    # -------------------------------------------------------------------------
    if not tvae_cand_path.is_file():
        raise FileNotFoundError(f"Missing TVAE candidate artifact: {tvae_cand_path}")

    tvae_cand_df = pd.read_csv(tvae_cand_path)
    if len(tvae_cand_df) != 50:
        raise ValueError(f"Expected 50 TVAE candidates, got {len(tvae_cand_df)}")

    bundle_tvae = source_prep.transform(tvae_cand_df)
    X_tvae, y_tvae = bundle_tvae.X, bundle_tvae.y

    # Deterministic authentic ROS duplicates (K=50) for matched weighted controls
    rng = np.random.default_rng(seed)
    ros_indices_50 = rng.choice(len(b200_post), size=50, replace=True)
    b200_ros50 = b200_post.iloc[ros_indices_50].copy().reset_index(drop=True)
    bundle_ros50 = source_prep.transform(b200_ros50)
    X_ros50, y_ros50 = bundle_ros50.X, bundle_ros50.y

    # Combined matrices
    X_hybrid = np.vstack([X_src, X_b200, X_tvae])
    y_hybrid = np.concatenate([y_src, y_b200, y_tvae])

    X_ros_matched = np.vstack([X_src, X_b200, X_ros50])
    y_ros_matched = np.concatenate([y_src, y_b200, y_ros50])

    n_src = len(X_src)
    n_b200 = len(X_b200)
    n_synth = len(X_tvae)
    n_ros50 = len(X_ros50)

    # -------------------------------------------------------------------------
    # STEP 3: Load Existing Control Models and Train Pilot Arms
    # -------------------------------------------------------------------------
    all_models: dict[str, Any] = {}
    training_times: dict[str, float] = {}

    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 2,
        "bootstrap": True,
        "criterion": "gini",
    }

    # Historical reload arms
    all_models["B0"] = joblib.load(source_model_path)
    all_models["B1_B200"] = joblib.load(models_dir / "telco_rf_a1_b200.joblib")
    all_models["B2_B200_K50"] = joblib.load(models_dir / "telco_rf_b2_b200_k50.joblib")
    all_models["B2_B200_K200"] = joblib.load(models_dir / "telco_rf_b2_b200_k200.joblib")
    all_models["B5_B200_w2"] = joblib.load(models_dir / "telco_rf_b5_b200_w2.joblib")
    all_models["B5_B200_w5"] = joblib.load(models_dir / "telco_rf_b5_b200_w5.joblib")
    all_models["S1_TVAE_K50"] = joblib.load(models_dir / "telco_rf_experimental_s1_tvae_k50.joblib")

    # Newly trained arms for Batch 5L
    new_train_specs = {
        # Matched authentic weighted ROS controls
        "B2_B200_K50_w2": (
            X_ros_matched,
            y_ros_matched,
            np.concatenate([np.full(n_src, 1.0), np.full(n_b200, 2.0), np.full(n_ros50, 2.0)]),
        ),
        "B2_B200_K50_w5": (
            X_ros_matched,
            y_ros_matched,
            np.concatenate([np.full(n_src, 1.0), np.full(n_b200, 5.0), np.full(n_ros50, 5.0)]),
        ),
        # Weighted hybrid synthetic arms
        "H1_TVAE_K50_w2": (
            X_hybrid,
            y_hybrid,
            np.concatenate([np.full(n_src, 1.0), np.full(n_b200, 2.0), np.full(n_synth, 2.0)]),
        ),
        "H1_TVAE_K50_w5": (
            X_hybrid,
            y_hybrid,
            np.concatenate([np.full(n_src, 1.0), np.full(n_b200, 5.0), np.full(n_synth, 5.0)]),
        ),
        "H2_TVAE_K50_w2_disc": (
            X_hybrid,
            y_hybrid,
            np.concatenate([np.full(n_src, 1.0), np.full(n_b200, 2.0), np.full(n_synth, 1.0)]),
        ),
        "H2_TVAE_K50_w5_disc": (
            X_hybrid,
            y_hybrid,
            np.concatenate([np.full(n_src, 1.0), np.full(n_b200, 5.0), np.full(n_synth, 2.5)]),
        ),
    }

    for arm_id, (X_tr, y_tr, w_tr) in new_train_specs.items():
        t0 = time.perf_counter()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr, y_tr, sample_weight=w_tr)
        dur = round(time.perf_counter() - t0, 3)

        # Save model
        model_filename = f"telco_rf_{arm_id.lower()}.joblib"
        joblib.dump(rf, models_dir / model_filename)

        all_models[arm_id] = rf
        training_times[arm_id] = dur

    # -------------------------------------------------------------------------
    # STEP 4: Evaluation on D_dev_val and D_source_eval
    # -------------------------------------------------------------------------
    dev_metrics = {}
    source_metrics = {}
    affected_slice_metrics = {}
    preserved_slice_metrics = {}
    predictions_map = {}
    probabilities_map = {}

    for arm_id, model in all_models.items():
        # Overall dev metrics
        dev_metrics[arm_id] = compute_metrics_bundle(model, X_dev, y_dev)
        # Sliced dev metrics
        affected_slice_metrics[arm_id] = compute_metrics_bundle(model, X_dev[mask_affected], y_dev[mask_affected])
        preserved_slice_metrics[arm_id] = compute_metrics_bundle(model, X_dev[mask_preserved], y_dev[mask_preserved])
        # Source retention metrics
        source_metrics[arm_id] = compute_metrics_bundle(model, X_seval, y_seval)

        predictions_map[arm_id] = model.predict(X_dev)
        probabilities_map[arm_id] = model.predict_proba(X_dev)[:, 1]

    # -------------------------------------------------------------------------
    # STEP 5: Statistical Testing across Pre-Registered Hypothesis Family
    # -------------------------------------------------------------------------
    # Primary comparison: vs Matched B2_B200_K50
    primary_cand_arms = ["H1_TVAE_K50_w2", "H1_TVAE_K50_w5", "H2_TVAE_K50_w2_disc", "H2_TVAE_K50_w5_disc", "S1_TVAE_K50"]
    primary_comparisons = {}
    raw_p_values = []

    ref_preds_k50 = predictions_map["B2_B200_K50"]

    for arm_id in primary_cand_arms:
        cand_preds = predictions_map[arm_id]
        delta_f1 = round(dev_metrics[arm_id]["positive_f1"] - dev_metrics["B2_B200_K50"]["positive_f1"], 4)
        boot_ci = paired_bootstrap_ci(y_dev, cand_preds, ref_preds_k50, n_bootstraps=2000, seed=seed)
        perm_res = paired_permutation_test(y_dev, cand_preds, ref_preds_k50, n_permutations=2000, seed=seed)

        raw_p_values.append(perm_res["p_value"])
        primary_comparisons[arm_id] = {
            "reference_arm": "B2_B200_K50",
            "delta_f1": delta_f1,
            "bootstrap_ci_95": [boot_ci["lower_ci"], boot_ci["upper_ci"]],
            "bootstrap_mean_diff": boot_ci["mean_diff"],
            "permutation_p_value": perm_res["p_value"],
        }

    # Step-down Holm-Bonferroni correction across the 5 primary hypotheses
    adj_p_values = holm_bonferroni_correction(raw_p_values)
    for arm_id, adj_p in zip(primary_cand_arms, adj_p_values):
        primary_comparisons[arm_id]["holm_bonferroni_p_value"] = adj_p
        primary_comparisons[arm_id]["statistically_significant_adjusted"] = bool(adj_p < 0.05)

    # Secondary comparisons vs. Authentic Weighted Counterparts
    secondary_comparisons = {
        "H1_vs_B2_w2": {
            "candidate": "H1_TVAE_K50_w2",
            "reference": "B2_B200_K50_w2",
            "delta_f1": round(dev_metrics["H1_TVAE_K50_w2"]["positive_f1"] - dev_metrics["B2_B200_K50_w2"]["positive_f1"], 4),
            "bootstrap_ci_95": paired_bootstrap_ci(y_dev, predictions_map["H1_TVAE_K50_w2"], predictions_map["B2_B200_K50_w2"], n_bootstraps=2000, seed=seed),
            "permutation_test": paired_permutation_test(y_dev, predictions_map["H1_TVAE_K50_w2"], predictions_map["B2_B200_K50_w2"], n_permutations=2000, seed=seed),
        },
        "H1_vs_B5_w2": {
            "candidate": "H1_TVAE_K50_w2",
            "reference": "B5_B200_w2",
            "delta_f1": round(dev_metrics["H1_TVAE_K50_w2"]["positive_f1"] - dev_metrics["B5_B200_w2"]["positive_f1"], 4),
            "bootstrap_ci_95": paired_bootstrap_ci(y_dev, predictions_map["H1_TVAE_K50_w2"], predictions_map["B5_B200_w2"], n_bootstraps=2000, seed=seed),
            "permutation_test": paired_permutation_test(y_dev, predictions_map["H1_TVAE_K50_w2"], predictions_map["B5_B200_w2"], n_permutations=2000, seed=seed),
        },
        "H1_vs_B2_w5": {
            "candidate": "H1_TVAE_K50_w5",
            "reference": "B2_B200_K50_w5",
            "delta_f1": round(dev_metrics["H1_TVAE_K50_w5"]["positive_f1"] - dev_metrics["B2_B200_K50_w5"]["positive_f1"], 4),
            "bootstrap_ci_95": paired_bootstrap_ci(y_dev, predictions_map["H1_TVAE_K50_w5"], predictions_map["B2_B200_K50_w5"], n_bootstraps=2000, seed=seed),
            "permutation_test": paired_permutation_test(y_dev, predictions_map["H1_TVAE_K50_w5"], predictions_map["B2_B200_K50_w5"], n_permutations=2000, seed=seed),
        },
    }

    # -------------------------------------------------------------------------
    # STEP 6: Predictions Audit CSV
    # -------------------------------------------------------------------------
    audit_rows = []
    for i in range(len(clean_dev_val_post)):
        tc_raw = clean_dev_val_post.iloc[i]["TotalCharges"]
        try:
            tc_val = float(str(tc_raw).strip())
        except (ValueError, TypeError):
            tc_val = 0.0

        row_dict = {
            "eval_index": i,
            "tenure": float(clean_dev_val_post.iloc[i]["tenure"]),
            "MonthlyCharges": float(clean_dev_val_post.iloc[i]["MonthlyCharges"]),
            "TotalCharges": tc_val,
            "is_affected_slice": bool(dev_flips.iloc[i]),
            "y_true": int(y_dev[i]),
        }
        for arm_id in all_models.keys():
            row_dict[f"pred_{arm_id}"] = int(predictions_map[arm_id][i])
            row_dict[f"prob_{arm_id}"] = round(float(probabilities_map[arm_id][i]), 4)
        audit_rows.append(row_dict)

    df_audit = pd.DataFrame(audit_rows)
    audit_csv_path = results_dir / "batch05l_weighted_repair_predictions_audit.csv"
    df_audit.to_csv(audit_csv_path, index=False)

    # -------------------------------------------------------------------------
    # STEP 7: Serialization of Results and Protocols
    # -------------------------------------------------------------------------
    total_elapsed = round(time.perf_counter() - t_start, 2)
    process_rss = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

    protocol_manifest = {
        "protocol_name": "Batch_5L_Targeted_and_Weighted_Synthetic_Repair",
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "targeted_repair_status": "BLOCKED (No legitimate non-oracle diagnostic slice from Components 1/2)",
        "evaluation_isolation_status": "STRICT_PASS (D_target_eval never loaded or accessed)",
        "execution_telemetry": {
            "total_elapsed_seconds": total_elapsed,
            "peak_rss_mb": process_rss,
            "os": sys.platform,
            "python_version": sys.version.split()[0],
        },
        "verified_hashes": verified_hashes,
        "evaluated_arms": list(all_models.keys()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    results_manifest = {
        "dev_val_metrics": dev_metrics,
        "affected_slice_metrics": affected_slice_metrics,
        "preserved_slice_metrics": preserved_slice_metrics,
        "source_retention_metrics": source_metrics,
        "training_times_seconds": training_times,
    }

    comparisons_manifest = {
        "primary_comparisons_vs_matched_b2_k50": primary_comparisons,
        "secondary_comparisons_vs_authentic_weighted": secondary_comparisons,
    }

    with open(results_dir / "batch05l_weighted_repair_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol_manifest, f, indent=2)

    with open(results_dir / "batch05l_weighted_repair_results.json", "w", encoding="utf-8") as f:
        json.dump(results_manifest, f, indent=2)

    with open(results_dir / "batch05l_weighted_repair_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons_manifest, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 8: Visualization Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    plot_arms = [
        "B0", "B1_B200", "B2_B200_K50", "B2_B200_K50_w2", "B5_B200_w2",
        "S1_TVAE_K50", "H1_TVAE_K50_w2", "H1_TVAE_K50_w5", "H2_TVAE_K50_w2_disc"
    ]
    colors = [
        "#4a5568", "#2b6cb0", "#3182ce", "#2b6cb0", "#805ad5",
        "#38a169", "#2f855a", "#276749", "#dd6b20"
    ]

    dev_f1_vals = [dev_metrics[a]["positive_f1"] for a in plot_arms]
    src_f1_vals = [source_metrics[a]["positive_f1"] for a in plot_arms]

    # Panel A: Adaptation Utility
    bars0 = axes[0].bar(plot_arms, dev_f1_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Post-Drift Adaptation Utility (D_dev_val, N=292)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Positive Churn F1", fontsize=10)
    axes[0].set_ylim(0.35, 0.48)
    axes[0].axhline(y=dev_metrics["B2_B200_K50"]["positive_f1"], color="#e53e3e", linestyle="--", label=f"Matched ROS K50 ({dev_metrics['B2_B200_K50']['positive_f1']})")
    axes[0].axhline(y=dev_metrics["B0"]["positive_f1"], color="#718096", linestyle=":", label=f"B0 Source Baseline ({dev_metrics['B0']['positive_f1']})")
    axes[0].set_xticks(range(len(plot_arms)))
    axes[0].set_xticklabels(plot_arms, rotation=45, ha="right", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=8)

    for bar in bars0:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, h + 0.003, f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    # Panel B: Source Retention
    bars1 = axes[1].bar(plot_arms, src_f1_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("Source Retention (D_source_eval, N=493)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Source Positive F1", fontsize=10)
    axes[1].set_ylim(0.50, 0.65)
    axes[1].axhline(y=source_metrics["B0"]["positive_f1"], color="#718096", linestyle=":", label=f"B0 Source Baseline ({source_metrics['B0']['positive_f1']})")
    axes[1].set_xticks(range(len(plot_arms)))
    axes[1].set_xticklabels(plot_arms, rotation=45, ha="right", fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left", fontsize=8)

    for bar in bars1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, h + 0.003, f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig_path = figures_dir / "batch05l_weighted_repair_metrics.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return {
        "status": "SUCCESS",
        "total_elapsed_seconds": total_elapsed,
        "dev_metrics": dev_metrics,
        "source_metrics": source_metrics,
        "primary_comparisons": primary_comparisons,
        "secondary_comparisons": secondary_comparisons,
        "affected_slice_metrics": affected_slice_metrics,
        "preserved_slice_metrics": preserved_slice_metrics,
    }


if __name__ == "__main__":
    print("=" * 75)
    print("SPRINT 06 — BATCH 5L: TARGETED AND WEIGHTED SYNTHETIC REPAIR PILOT")
    print("=" * 75)
    res = execute_targeted_weighted_repair()
    print("\nBenchmark Execution Complete in", res["total_elapsed_seconds"], "s.")
    print("\nPrimary Development Validation Metrics (D_dev_val, N=292):")
    for arm, m in res["dev_metrics"].items():
        print(f"  {arm:24s} | Pos F1: {m['positive_f1']:.4f} | Macro F1: {m['macro_f1']:.4f} | Bal Acc: {m['balanced_accuracy']:.4f}")
    print("\nPrimary Statistical Comparisons vs Matched B2_B200_K50:")
    for arm, c in res["primary_comparisons"].items():
        print(f"  {arm:24s} | Delta F1: {c['delta_f1']:+.4f} | 95% CI: {c['bootstrap_ci_95']} | Holm p: {c['holm_bonferroni_p_value']:.4f}")
