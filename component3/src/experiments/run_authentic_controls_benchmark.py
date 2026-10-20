"""Execution Engine for Sprint 06 — Batch 5I: Strong Authentic Adaptation Controls.

Implements and evaluates the six non-synthetic adaptation controls (B0-B5)
under the frozen TELCO_CONCEPT distribution shift:
- B0: Frozen unadapted source model (Reload Batch 5F A0)
- B1: Naive authentic pooling (Reload Batch 5G A1 models)
- B2: Authentic Random OverSampling (ROS, K=50, K=200)
- B3: Balanced domain weighting (w_target = 3,452 / B_k)
- B4: Target-only model (trained strictly on B_k)
- B5: Fixed target-weight grid (w_target in {2.0, 5.0, 10.0, 20.0})

Evaluated exclusively on clean, group-isolated development validation data (N=292).
Source-domain retention evaluated diagnostically on D_source_eval (N=493).
Zero access to locked D_target_eval, canonical validation, or canonical test.
"""

from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
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
) -> tuple[float, float, float]:
    """Calculate paired bootstrap confidence intervals for positive-class F1 difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        f1_cand = f1_score(y_true[idx], y_pred_candidate[idx], zero_division=0)
        f1_ref = f1_score(y_true[idx], y_pred_reference[idx], zero_division=0)
        diffs[b] = f1_cand - f1_ref

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    return round(lower, 4), round(upper, 4), round(mean_diff, 4)


def paired_permutation_test(
    y_true: np.ndarray,
    y_pred_candidate: np.ndarray,
    y_pred_reference: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate paired two-sided permutation test p-value for positive-class F1 difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_diff = float(
        f1_score(y_true, y_pred_candidate, zero_division=0)
        - f1_score(y_true, y_pred_reference, zero_division=0)
    )

    perm_diffs = np.empty(n_permutations, dtype=np.float64)
    for r in range(n_permutations):
        swap = rng.random(n) < 0.5
        perm_cand = np.where(swap, y_pred_reference, y_pred_candidate)
        perm_ref = np.where(swap, y_pred_candidate, y_pred_reference)
        f1_c = f1_score(y_true, perm_cand, zero_division=0)
        f1_r = f1_score(y_true, perm_ref, zero_division=0)
        perm_diffs[r] = f1_c - f1_r

    # Finite-sample p-value correction: (extreme_count + 1) / (n_permutations + 1)
    extreme_count = int(np.sum(np.abs(perm_diffs) >= np.abs(obs_diff) - 1e-9))
    p_val = float((extreme_count + 1) / (n_permutations + 1))
    return round(obs_diff, 4), round(p_val, 4)


def compute_metrics_bundle(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    slice_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics bundle, optionally masked to a slice."""
    if slice_mask is not None:
        X_sub = X[slice_mask]
        y_sub = y[slice_mask]
    else:
        X_sub = X
        y_sub = y

    y_pred = model.predict(X_sub)
    y_prob = model.predict_proba(X_sub)[:, 1] if hasattr(model, "predict_proba") else None

    f1_pos = float(f1_score(y_sub, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_sub, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_sub, y_pred))
    bal_acc = float(balanced_accuracy_score(y_sub, y_pred))
    prec = float(precision_score(y_sub, y_pred, zero_division=0))
    rec = float(recall_score(y_sub, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_sub, y_prob)) if y_prob is not None and len(np.unique(y_sub)) > 1 else None
    pr_auc = float(average_precision_score(y_sub, y_prob)) if y_prob is not None and len(np.unique(y_sub)) > 1 else None
    cm = confusion_matrix(y_sub, y_pred).tolist()

    pos_support = int(np.sum(y_sub == 1))
    neg_support = int(np.sum(y_sub == 0))

    return {
        "positive_f1": round(f1_pos, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "average_precision": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm,
        "positive_support": pos_support,
        "negative_support": neg_support,
        "total_support": len(y_sub),
    }


def execute_authentic_controls_benchmark(
    seed: int = 42,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Batch 5I benchmark across B0-B5 control arms."""
    root = Path(base_dir) if base_dir else Path("component3").resolve().parent
    results_dir = root / "component3" / "results" / "experimental_models"
    figures_dir = root / "component3" / "results" / "figures"
    models_dir = root / "component3" / "models" / "experimental"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # STEP 1: Input Verification and Cryptographic Freezing
    # -------------------------------------------------------------------------
    canon_split_path = root / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    adapt_split_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
    leakage_audit_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_group_overlap_audit.json"
    source_model_path = root / "component3" / "models" / "experimental" / "telco_rf_experimental_source.joblib"
    source_prep_path = root / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib"
    baseline_rf_path = root / "component3" / "models" / "baselines" / "telco_churn_random_forest.joblib"

    expected_hashes = {
        "canonical_split": "e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68",
        "adaptation_split": "34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59",
        "leakage_audit": "884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518",
        "historical_baseline_rf": "69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701",
        "source_rf": "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0",
    }

    for name, expected_hash in expected_hashes.items():
        actual_path = {
            "canonical_split": canon_split_path,
            "adaptation_split": adapt_split_path,
            "leakage_audit": leakage_audit_path,
            "historical_baseline_rf": baseline_rf_path,
            "source_rf": source_model_path,
        }[name]
        actual_hash = compute_file_sha256(actual_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Cryptographic hash mismatch for {name}: expected {expected_hash}, got {actual_hash}")

    with open(adapt_split_path, "r", encoding="utf-8") as f:
        adapt_manifest = json.load(f)
    with open(leakage_audit_path, "r", encoding="utf-8") as f:
        leakage_audit = json.load(f)

    if leakage_audit.get("leakage_audit_status") != "PASS":
        raise ValueError("Audit status is not PASS!")
    if leakage_audit.get("total_cross_partition_index_overlaps", -1) != 0:
        raise ValueError("Nonzero cross-partition index overlap detected!")
    if leakage_audit.get("total_cross_partition_group_overlaps", -1) != 0:
        raise ValueError("Nonzero cross-partition group overlap detected!")

    rc = adapt_manifest["realized_counts"]
    if rc != {"source_train": 3452, "source_eval": 493, "target_adapt_pool": 493, "target_eval": 493}:
        raise ValueError(f"Partition count mismatch: {rc}")

    # Load canonical data and split containers
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    train_indices = canon_manifest["train_indices"]
    canon_train_df = df_raw.iloc[train_indices].copy().reset_index(drop=True)
    current_fp = compute_dataset_fingerprint(canon_train_df)
    if current_fp != adapt_manifest["source_dataset_fingerprint"]:
        raise ValueError(f"Fingerprint mismatch: expected {adapt_manifest['source_dataset_fingerprint']}, got {current_fp}")

    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=seed,
    )

    df_source_train = splits.source_train.copy().reset_index(drop=True)
    df_source_eval = splits.source_eval.copy().reset_index(drop=True)
    df_target_adapt_pool = splits.target_adapt_pool.copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # STEP 2: Dynamic Development Validation Isolation
    # -------------------------------------------------------------------------
    # Dynamically identify B200 members from nested adaptation budget container
    b200_df = splits.nested_adapt_budgets[200].copy().reset_index(drop=True)
    feat_cols = meta.feature_columns

    # Dynamic membership identification without hardcoding customer IDs
    b200_cust_ids = set(b200_df["customerID"])
    is_in_b200 = df_target_adapt_pool["customerID"].isin(b200_cust_ids)
    raw_unused_pool = df_target_adapt_pool[~is_in_b200].copy().reset_index(drop=True)

    # Compute duplicate-feature signatures using frozen canonical feature schema
    b200_signatures = set(tuple(row) for row in b200_df[feat_cols].itertuples(index=False))
    unused_signatures = [tuple(row) for row in raw_unused_pool[feat_cols].itertuples(index=False)]

    # Exclude any development record whose feature signature overlaps ANY B200 record
    overlapping_indices = [idx for idx, sig in enumerate(unused_signatures) if sig in b200_signatures]
    clean_dev_val = raw_unused_pool.drop(index=overlapping_indices).copy().reset_index(drop=True)

    realized_clean_dev_val_count = len(clean_dev_val)
    if realized_clean_dev_val_count != 292:
        raise ValueError(f"Expected clean dev_val size 292, got {realized_clean_dev_val_count}!")

    # Verify no overlap with any other partition
    clean_dev_signatures = set(tuple(row) for row in clean_dev_val[feat_cols].itertuples(index=False))
    train_signatures = set(tuple(row) for row in df_source_train[feat_cols].itertuples(index=False))
    source_eval_signatures = set(tuple(row) for row in df_source_eval[feat_cols].itertuples(index=False))
    target_eval_signatures = set(tuple(row) for row in splits.target_eval[feat_cols].itertuples(index=False))

    overlap_b200 = len(clean_dev_signatures.intersection(b200_signatures))
    overlap_train = len(clean_dev_signatures.intersection(train_signatures))
    overlap_seval = len(clean_dev_signatures.intersection(source_eval_signatures))
    overlap_teval = len(clean_dev_signatures.intersection(target_eval_signatures))

    if overlap_b200 != 0 or overlap_train != 0 or overlap_seval != 0 or overlap_teval != 0:
        raise ValueError(
            f"Cross-partition signature leakage detected! B200:{overlap_b200}, "
            f"Train:{overlap_train}, SourceEval:{overlap_seval}, TargetEval:{overlap_teval}"
        )

    # Apply exact frozen TELCO_CONCEPT transformation to dev validation labels
    clean_dev_val["_original_label"] = clean_dev_val["Churn"].copy()
    flips_dev = clean_dev_val["tenure"] > 29.0
    clean_dev_val.loc[flips_dev, "Churn"] = clean_dev_val.loc[flips_dev, "Churn"].map({"No": "Yes", "Yes": "No"})
    clean_dev_val["_concept_provenance"] = "SYNTHETIC_CONCEPT_TRUTH"

    # Save anonymized development split audit and manifest
    dev_split_audit = {
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "target_adapt_pool_total": len(df_target_adapt_pool),
        "b200_total": len(b200_df),
        "raw_unused_adaptation_pool_total": len(raw_unused_pool),
        "duplicate_signature_exclusions": len(overlapping_indices),
        "clean_dev_val_realized_count": realized_clean_dev_val_count,
        "original_class_distribution": clean_dev_val["_original_label"].value_counts().to_dict(),
        "transformed_class_distribution": clean_dev_val["Churn"].value_counts().to_dict(),
        "affected_slice_tenure_gt_29_count": int(flips_dev.sum()),
        "preserved_slice_tenure_le_29_count": int((~flips_dev).sum()),
        "cross_partition_signature_overlaps": {
            "with_b200": overlap_b200,
            "with_source_train": overlap_train,
            "with_source_eval": overlap_seval,
            "with_locked_target_eval": overlap_teval,
        },
        "leakage_verification_status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir / "telco_dev_val_split_audit.json", "w", encoding="utf-8") as f:
        json.dump(dev_split_audit, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 3: Pre-Register Machine-Readable Protocol
    # -------------------------------------------------------------------------
    protocol = {
        "protocol_name": "Batch_5I_Strong_Authentic_Adaptation_Controls",
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "control_arms": {
            "B0": "Frozen Source Model (Batch 5F A0, 3,452 source records, 0 adapt)",
            "B1": "Naive Authentic Pooling (3,452 source + B_k target records, uniform weights)",
            "B2": "Authentic Random OverSampling (3,452 source + B_k target + K duplicated target records)",
            "B3": "Balanced Domain Weighting (w_source=1.0, w_target = 3,452 / B_k)",
            "B4": "Target-Only Model (B_k target records only, 0 source records)",
            "B5": "Fixed Target-Weight Grid (w_source=1.0, w_target in {2.0, 5.0, 10.0, 20.0})",
        },
        "adaptation_budgets": [50, 100, 200],
        "oversampling_quantities_K": [50, 200],
        "prespecified_weights": [2.0, 5.0, 10.0, 20.0],
        "classifier_hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 2,
            "bootstrap": True,
            "criterion": "gini",
        },
        "preprocessing_policy": (
            "Source-Fitted Preprocessor (telco_preprocessor_experimental_source.joblib). "
            "Applied consistently across all arms to maintain identical feature spaces and one-hot encodings."
        ),
        "evaluation_partition": "D_adapt,dev_val (N=292 clean, group-isolated records)",
        "source_retention_partition": "D_source_eval (N=493 clean pre-drift records)",
        "locked_target_evaluation_status": "STRICTLY UNTOUCHED",
        "statistical_comparison_plan": {
            "paired_bootstrap_iterations": 2000,
            "paired_permutation_iterations": 2000,
            "multiplicity_adjustments": ["bonferroni", "benjamini_hochberg_fdr"],
            "significance_alpha": 0.05,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir / "batch05i_authentic_controls_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 4: Preprocessing and Fit Isolation
    # -------------------------------------------------------------------------
    source_prep: TabularDataPreprocessor = joblib.load(source_prep_path)

    # Preprocess dev_val, source_train, and source_eval
    bundle_val = source_prep.transform(clean_dev_val)
    X_val = bundle_val.X
    y_val = bundle_val.y
    slice_tenure_gt_29 = flips_dev.to_numpy()
    slice_tenure_le_29 = (~flips_dev).to_numpy()

    bundle_src_train = source_prep.transform(df_source_train)
    X_src_train = bundle_src_train.X
    y_src_train = bundle_src_train.y

    bundle_src_eval = source_prep.transform(df_source_eval)
    X_src_eval = bundle_src_eval.X
    y_src_eval = bundle_src_eval.y

    # Prepare transformed adaptation budgets B50, B100, B200
    budget_bundles: dict[int, dict[str, Any]] = {}
    for b_size in [50, 100, 200]:
        b_df = splits.nested_adapt_budgets[b_size].copy().reset_index(drop=True)
        b_flips = b_df["tenure"] > 29.0
        b_df.loc[b_flips, "Churn"] = b_df.loc[b_flips, "Churn"].map({"No": "Yes", "Yes": "No"})
        b_bundle = source_prep.transform(b_df)
        budget_bundles[b_size] = {
            "df": b_df,
            "X": b_bundle.X,
            "y": b_bundle.y,
            "n_flips": int(b_flips.sum()),
        }

    # -------------------------------------------------------------------------
    # STEP 5: Train & Evaluate Control Arms
    # -------------------------------------------------------------------------
    models: dict[str, Any] = {}
    telemetry: dict[str, dict[str, Any]] = {}

    # B0 — Frozen source model (Batch 5F A0)
    rf_b0: RandomForestClassifier = joblib.load(source_model_path)
    models["B0"] = rf_b0
    telemetry["B0"] = {"fit_time_seconds": 0.0, "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2)}

    # B1 — Naive authentic pooling (Reused from Batch 5G A1 models)
    for b_size in [50, 100, 200]:
        arm_id = f"B1_B{b_size}"
        b1_model_path = models_dir / f"telco_rf_a1_b{b_size}.joblib"
        if not b1_model_path.is_file():
            # If not on disk, fit fresh
            X_b = budget_bundles[b_size]["X"]
            y_b = budget_bundles[b_size]["y"]
            X_comb = np.vstack([X_src_train, X_b])
            y_comb = np.concatenate([y_src_train, y_b])
            rf_b1 = RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                random_state=seed, n_jobs=2, bootstrap=True, criterion="gini"
            )
            rf_b1.fit(X_comb, y_comb)
            models[arm_id] = rf_b1
        else:
            models[arm_id] = joblib.load(b1_model_path)
        telemetry[arm_id] = {"fit_time_seconds": 0.0, "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2)}

    # B2 — Authentic Random OverSampling (ROS, K=50, K=200)
    for b_size in [50, 100, 200]:
        X_b = budget_bundles[b_size]["X"]
        y_b = budget_bundles[b_size]["y"]
        X_comb_base = np.vstack([X_src_train, X_b])
        y_comb_base = np.concatenate([y_src_train, y_b])

        for k in [50, 200]:
            arm_id = f"B2_B{b_size}_K{k}"
            b2_model_path = models_dir / f"telco_rf_{arm_id.lower()}.joblib"
            if b2_model_path.is_file():
                models[arm_id] = joblib.load(b2_model_path)
                telemetry[arm_id] = {
                    "fit_time_seconds": 0.0,
                    "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                    "training_rows": len(y_comb_base) + k,
                    "unique_target_rows": len(y_b),
                    "duplicated_target_rows": k,
                }
            else:
                t0 = time.perf_counter()
                rng_ros = np.random.default_rng(seed + k)
                ros_indices = rng_ros.choice(len(y_b), size=k, replace=True)
                X_ros = np.vstack([X_comb_base, X_b[ros_indices]])
                y_ros = np.concatenate([y_comb_base, y_b[ros_indices]])

                rf_b2 = RandomForestClassifier(
                    n_estimators=100, max_depth=10, class_weight="balanced",
                    random_state=seed, n_jobs=2, bootstrap=True, criterion="gini"
                )
                rf_b2.fit(X_ros, y_ros)
                dur = time.perf_counter() - t0
                models[arm_id] = rf_b2
                telemetry[arm_id] = {
                    "fit_time_seconds": round(dur, 4),
                    "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                    "training_rows": len(y_ros),
                    "unique_target_rows": len(y_b),
                    "duplicated_target_rows": k,
                }
                joblib.dump(rf_b2, b2_model_path)

    # B3 — Balanced Domain Weighting (w_target = 3,452 / B_k)
    for b_size in [50, 100, 200]:
        arm_id = f"B3_B{b_size}"
        b3_model_path = models_dir / f"telco_rf_{arm_id.lower()}.joblib"
        X_b = budget_bundles[b_size]["X"]
        y_b = budget_bundles[b_size]["y"]
        if b3_model_path.is_file():
            models[arm_id] = joblib.load(b3_model_path)
            telemetry[arm_id] = {
                "fit_time_seconds": 0.0,
                "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "training_rows": len(y_src_train) + len(y_b),
                "source_sample_weight": 1.0,
                "target_sample_weight": round(len(y_src_train) / len(y_b), 4),
                "total_target_effective_weight": round(float(len(y_src_train)), 1),
            }
        else:
            t0 = time.perf_counter()
            X_comb = np.vstack([X_src_train, X_b])
            y_comb = np.concatenate([y_src_train, y_b])
            w_target = len(y_src_train) / len(y_b)
            sample_weights = np.concatenate([np.ones(len(y_src_train), dtype=np.float64), np.full(len(y_b), w_target, dtype=np.float64)])

            rf_b3 = RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                random_state=seed, n_jobs=2, bootstrap=True, criterion="gini"
            )
            rf_b3.fit(X_comb, y_comb, sample_weight=sample_weights)
            dur = time.perf_counter() - t0
            models[arm_id] = rf_b3
            telemetry[arm_id] = {
                "fit_time_seconds": round(dur, 4),
                "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "training_rows": len(y_comb),
                "source_sample_weight": 1.0,
                "target_sample_weight": round(w_target, 4),
                "total_target_effective_weight": round(float(w_target * len(y_b)), 1),
            }
            joblib.dump(rf_b3, b3_model_path)

    # B4 — Target-Only Model (B_k only)
    for b_size in [50, 100, 200]:
        arm_id = f"B4_B{b_size}"
        b4_model_path = models_dir / f"telco_rf_{arm_id.lower()}.joblib"
        X_b = budget_bundles[b_size]["X"]
        y_b = budget_bundles[b_size]["y"]
        if b4_model_path.is_file():
            models[arm_id] = joblib.load(b4_model_path)
            telemetry[arm_id] = {
                "fit_time_seconds": 0.0,
                "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "training_rows": len(y_b),
                "source_rows": 0,
            }
        else:
            t0 = time.perf_counter()
            rf_b4 = RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                random_state=seed, n_jobs=2, bootstrap=True, criterion="gini"
            )
            rf_b4.fit(X_b, y_b)
            dur = time.perf_counter() - t0
            models[arm_id] = rf_b4
            telemetry[arm_id] = {
                "fit_time_seconds": round(dur, 4),
                "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "training_rows": len(y_b),
                "source_rows": 0,
            }
            joblib.dump(rf_b4, b4_model_path)

    # B5 — Fixed Target-Weight Grid (w in {2, 5, 10, 20})
    for b_size in [50, 100, 200]:
        X_b = budget_bundles[b_size]["X"]
        y_b = budget_bundles[b_size]["y"]
        X_comb = np.vstack([X_src_train, X_b])
        y_comb = np.concatenate([y_src_train, y_b])

        for w in [2.0, 5.0, 10.0, 20.0]:
            w_str = str(int(w))
            arm_id = f"B5_B{b_size}_w{w_str}"
            b5_model_path = models_dir / f"telco_rf_{arm_id.lower()}.joblib"
            if b5_model_path.is_file():
                models[arm_id] = joblib.load(b5_model_path)
                telemetry[arm_id] = {
                    "fit_time_seconds": 0.0,
                    "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                    "training_rows": len(y_comb),
                    "target_weight": w,
                }
            else:
                t0 = time.perf_counter()
                sample_weights = np.concatenate([np.ones(len(y_src_train), dtype=np.float64), np.full(len(y_b), w, dtype=np.float64)])

                rf_b5 = RandomForestClassifier(
                    n_estimators=100, max_depth=10, class_weight="balanced",
                    random_state=seed, n_jobs=2, bootstrap=True, criterion="gini"
                )
                rf_b5.fit(X_comb, y_comb, sample_weight=sample_weights)
                dur = time.perf_counter() - t0
                models[arm_id] = rf_b5
                telemetry[arm_id] = {
                    "fit_time_seconds": round(dur, 4),
                    "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                    "training_rows": len(y_comb),
                    "target_weight": w,
                }
                joblib.dump(rf_b5, b5_model_path)

    # -------------------------------------------------------------------------
    # STEP 6: Compute Development Validation Metrics
    # -------------------------------------------------------------------------
    dev_metrics: dict[str, dict[str, Any]] = {}
    predictions_map: dict[str, np.ndarray] = {}
    probabilities_map: dict[str, np.ndarray] = {}

    for arm_id, model in models.items():
        # Overall dev metrics
        overall = compute_metrics_bundle(model, X_val, y_val)
        # Sliced metrics
        slice_gt_29 = compute_metrics_bundle(model, X_val, y_val, slice_mask=slice_tenure_gt_29)
        slice_le_29 = compute_metrics_bundle(model, X_val, y_val, slice_mask=slice_tenure_le_29)

        dev_metrics[arm_id] = {
            "overall": overall,
            "slice_tenure_gt_29": slice_gt_29,
            "slice_tenure_le_29": slice_le_29,
            "telemetry": telemetry.get(arm_id, {}),
        }

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))
        predictions_map[arm_id] = preds
        probabilities_map[arm_id] = probs

    with open(results_dir / "telco_authentic_controls_dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 7: Scientific Comparisons (Paired Bootstrap & Permutation)
    # -------------------------------------------------------------------------
    # Prespecified paired comparisons:
    # 1. B1 vs B0 (naive pooling benefit)
    # 2. B2-K200 vs B1 (authentic duplication benefit)
    # 3. B3 vs B1 (balanced domain weighting benefit)
    # 4. B4 vs B1 (target-only learning benefit)
    # 5. B5-w2 vs B1 (target-weight selection benefit)
    # 6. B3 vs B5-w2 (fixed domain balance vs heuristic weight)
    comparisons: dict[str, dict[str, Any]] = {}

    for b_size in [50, 100, 200]:
        b1_id = f"B1_B{b_size}"
        b2_id = f"B2_B{b_size}_K200"
        b3_id = f"B3_B{b_size}"
        b4_id = f"B4_B{b_size}"
        b5_best_id = f"B5_B{b_size}_w2"

        comparison_pairs = [
            (f"{b1_id}_vs_B0", b1_id, "B0"),
            (f"{b2_id}_vs_{b1_id}", b2_id, b1_id),
            (f"{b3_id}_vs_{b1_id}", b3_id, b1_id),
            (f"{b4_id}_vs_{b1_id}", b4_id, b1_id),
            (f"{b5_best_id}_vs_{b1_id}", b5_best_id, b1_id),
            (f"{b3_id}_vs_{b5_best_id}", b3_id, b5_best_id),
        ]

        raw_p_values = []
        comp_records = []
        for pair_name, cand_id, ref_id in comparison_pairs:
            f1_cand = dev_metrics[cand_id]["overall"]["positive_f1"]
            f1_ref = dev_metrics[ref_id]["overall"]["positive_f1"]
            delta_f1 = round(f1_cand - f1_ref, 4)

            low_ci, up_ci, mean_diff = paired_bootstrap_ci(
                y_val, predictions_map[cand_id], predictions_map[ref_id],
                n_bootstraps=2000, seed=seed,
            )
            _, p_perm = paired_permutation_test(
                y_val, predictions_map[cand_id], predictions_map[ref_id],
                n_permutations=2000, seed=seed,
            )
            raw_p_values.append(p_perm)
            comp_records.append({
                "comparison_id": pair_name,
                "candidate_arm": cand_id,
                "reference_arm": ref_id,
                "candidate_f1": f1_cand,
                "reference_f1": f1_ref,
                "delta_f1": delta_f1,
                "bootstrap_95_ci_lower": low_ci,
                "bootstrap_95_ci_upper": up_ci,
                "ci_includes_zero": bool(low_ci <= 0.0 <= up_ci),
                "permutation_raw_p": p_perm,
            })

        # Multiplicity adjustments (Bonferroni and Benjamini-Hochberg FDR)
        m = len(raw_p_values)
        sorted_indices = np.argsort(raw_p_values)
        bonf_p = [min(round(p * m, 4), 1.0) for p in raw_p_values]

        sorted_p = [raw_p_values[idx] for idx in sorted_indices]
        q_vals = [sorted_p[rank] * m / (rank + 1) for rank in range(m)]
        for r in range(m - 2, -1, -1):
            q_vals[r] = min(q_vals[r], q_vals[r + 1])
        bh_p = [0.0] * m
        for rank, idx in enumerate(sorted_indices):
            bh_p[idx] = min(round(float(q_vals[rank]), 4), 1.0)

        for i, rec in enumerate(comp_records):
            rec["bonferroni_p"] = bonf_p[i]
            rec["significant_bonferroni"] = bool(bonf_p[i] < 0.05 and not rec["ci_includes_zero"])
            rec["benjamini_hochberg_fdr_p"] = bh_p[i]
            rec["significant_fdr"] = bool(bh_p[i] < 0.05 and not rec["ci_includes_zero"])
            comparisons[rec["comparison_id"]] = rec

    with open(results_dir / "telco_authentic_controls_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 8: Source-Domain Retention Diagnostic (D_source_eval, N=493)
    # -------------------------------------------------------------------------
    source_retention: dict[str, dict[str, Any]] = {}
    f1_a0_src = float(f1_score(y_src_eval, rf_b0.predict(X_src_eval), zero_division=0))

    for arm_id, model in models.items():
        y_pred_src = model.predict(X_src_eval)
        f1_src = float(f1_score(y_src_eval, y_pred_src, zero_division=0))
        macro_src = float(f1_score(y_src_eval, y_pred_src, average="macro", zero_division=0))
        bal_src = float(balanced_accuracy_score(y_src_eval, y_pred_src))
        delta_vs_a0 = round(f1_src - f1_a0_src, 4)

        source_retention[arm_id] = {
            "source_f1_positive": round(f1_src, 4),
            "source_macro_f1": round(macro_src, 4),
            "source_balanced_accuracy": round(bal_src, 4),
            "delta_f1_vs_a0": delta_vs_a0,
            "catastrophic_forgetting_flag": bool(delta_vs_a0 < -0.05),
        }

    with open(results_dir / "telco_authentic_controls_source_retention.json", "w", encoding="utf-8") as f:
        json.dump(source_retention, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 9: Anonymized Development Prediction Audit CSV
    # -------------------------------------------------------------------------
    audit_rows = []
    for row_idx in range(len(clean_dev_val)):
        row_dict = {
            "dev_row_id": f"dev_val_row_{row_idx:03d}",
            "original_label": clean_dev_val["_original_label"].iloc[row_idx],
            "ground_truth_target_label": clean_dev_val["Churn"].iloc[row_idx],
            "ground_truth_binary": int(y_val[row_idx]),
            "tenure_value": float(clean_dev_val["tenure"].iloc[row_idx]),
            "affected_slice_tenure_gt_29": bool(slice_tenure_gt_29[row_idx]),
            "b0_pred": int(predictions_map["B0"][row_idx]),
            "b0_prob": round(float(probabilities_map["B0"][row_idx]), 4),
            "b1_b200_pred": int(predictions_map["B1_B200"][row_idx]),
            "b1_b200_prob": round(float(probabilities_map["B1_B200"][row_idx]), 4),
            "b2_b200_k200_pred": int(predictions_map["B2_B200_K200"][row_idx]),
            "b2_b200_k200_prob": round(float(probabilities_map["B2_B200_K200"][row_idx]), 4),
            "b3_b200_pred": int(predictions_map["B3_B200"][row_idx]),
            "b3_b200_prob": round(float(probabilities_map["B3_B200"][row_idx]), 4),
            "b4_b200_pred": int(predictions_map["B4_B200"][row_idx]),
            "b4_b200_prob": round(float(probabilities_map["B4_B200"][row_idx]), 4),
            "b5_b200_w2_pred": int(predictions_map["B5_B200_w2"][row_idx]),
            "b5_b200_w2_prob": round(float(probabilities_map["B5_B200_w2"][row_idx]), 4),
        }
        audit_rows.append(row_dict)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(results_dir / "telco_authentic_controls_predictions_audit.csv", index=False)

    # -------------------------------------------------------------------------
    # STEP 10: Comparative Metrics Visualization
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Development Validation F1 across Control Arms
    budgets = [50, 100, 200]
    arms_to_plot = ["B1", "B2_K50", "B2_K200", "B3", "B4", "B5_w2"]
    arm_labels = {
        "B1": "B1 (Naive Pooling)",
        "B2_K50": "B2 (ROS K=50)",
        "B2_K200": "B2 (ROS K=200)",
        "B3": "B3 (Domain Weighting)",
        "B4": "B4 (Target-Only)",
        "B5_w2": "B5 (Weight w=2)",
    }

    x = np.arange(len(budgets))
    width = 0.13

    for idx, arm_prefix in enumerate(arms_to_plot):
        f1_vals = []
        for b in budgets:
            if arm_prefix.startswith("B2"):
                k_val = arm_prefix.split("_")[1]
                key = f"B2_B{b}_{k_val}"
            elif arm_prefix.startswith("B5"):
                w_val = arm_prefix.split("_")[1]
                key = f"B5_B{b}_{w_val}"
            else:
                key = f"{arm_prefix}_B{b}"
            f1_vals.append(dev_metrics[key]["overall"]["positive_f1"])

        axes[0].bar(x + idx * width - 0.35, f1_vals, width, label=arm_labels[arm_prefix])

    axes[0].axhline(dev_metrics["B0"]["overall"]["positive_f1"], color="red", linestyle="--", label=f"B0 Frozen Source ({dev_metrics['B0']['overall']['positive_f1']:.4f})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"B{b}" for b in budgets])
    axes[0].set_ylabel("Positive-Class F1 (Dev Validation)")
    axes[0].set_title("A. Authentic Adaptation Controls on Dev Validation")
    axes[0].set_ylim(0.25, 0.85)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Trade-Off: Target Dev F1 vs Source Retention F1 (Budget B200)
    target_f1s = []
    source_f1s = []
    point_labels = []

    selected_arms_b200 = ["B0", "B1_B200", "B2_B200_K200", "B3_B200", "B4_B200", "B5_B200_w2", "B5_B200_w20"]
    short_names = ["B0", "B1", "B2 (ROS 200)", "B3 (Dom Wgt)", "B4 (Tgt Only)", "B5 (w=2)", "B5 (w=20)"]

    for arm_id, sname in zip(selected_arms_b200, short_names):
        target_f1s.append(dev_metrics[arm_id]["overall"]["positive_f1"])
        source_f1s.append(source_retention[arm_id]["source_f1_positive"])
        point_labels.append(sname)

    scatter = axes[1].scatter(target_f1s, source_f1s, c=np.arange(len(point_labels)), cmap="viridis", s=120, edgecolors="black", zorder=3)
    for i, label in enumerate(point_labels):
        axes[1].annotate(
            label, (target_f1s[i], source_f1s[i]),
            textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight="bold"
        )

    axes[1].axhline(0.6075 - 0.05, color="red", linestyle=":", label="Forgetting Threshold (Delta F1 < -0.05)")
    axes[1].set_xlabel("Target Dev Validation F1")
    axes[1].set_ylabel("Source Domain Retention F1")
    axes[1].set_title("B. Target Recovery vs. Source Retention (Budget B200)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    plt.savefig(figures_dir / "batch05i_authentic_controls_metrics.png", dpi=300)
    plt.close()

    total_duration = round(time.perf_counter() - start_time, 2)
    gc.collect()

    return {
        "status": "SUCCESS",
        "clean_dev_val_rows": realized_clean_dev_val_count,
        "total_duration_seconds": total_duration,
        "dev_metrics": dev_metrics,
        "comparisons": comparisons,
        "source_retention": source_retention,
    }


if __name__ == "__main__":
    out = execute_authentic_controls_benchmark()
    print("Execution complete! Clean dev_val rows:", out["clean_dev_val_rows"])
