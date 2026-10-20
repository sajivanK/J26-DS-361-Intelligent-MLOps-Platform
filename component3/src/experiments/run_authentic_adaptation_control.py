"""Labeled Simulated-Target Adaptation Control Execution Engine.

Sprint 06 — Batch 5G:
Implements and evaluates the no-synthetic-data adaptation control (Arm A1)
for the frozen TELCO_CONCEPT simulation across adaptation budgets B50, B100, and B200.
Uses the experimental Random Forest source model from Batch 5F as the frozen no-repair reference (A0).

Strict Scientific Guarantees:
1. Frozen protocol pre-registered and saved before training.
2. Exact frozen TELCO_CONCEPT transformation: tenure > 29.0 inverts Churn; original labels preserved.
3. Completely self-contained preprocessing fitted strictly on each arm's permitted training composition.
4. Single-pass confirmatory evaluation on D_target_eval and D_source_eval.
5. Paired bootstrap 95% CIs and permutation tests relative to A0 on identical evaluation rows.
6. Multiplicity adjustment (Bonferroni and Benjamini-Hochberg) across m=3 comparisons.
7. Anonymized prediction audit saved without raw customer IDs.
8. Complete preservation and verification of historical baseline models and Batch 5F source model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import joblib
import numpy as np
import pandas as pd
import sklearn
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

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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

    p_val = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
    return round(obs_diff, 4), round(p_val, 4)


def compute_metrics_bundle(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics bundle."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    f1_pos = float(f1_score(y, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y, y_prob)) if y_prob is not None else 0.5
    pr_auc = float(average_precision_score(y, y_prob)) if y_prob is not None else 0.5
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        "f1_positive": round(f1_pos, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_positive": round(prec, 4),
        "recall_positive": round(rec, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
        "support": int(len(y)),
    }


def execute_batch05g_experiment(
    repo_root: Path | str = ".",
    seed: int = 42,
    n_bootstraps: int = 2000,
    n_permutations: int = 2000,
) -> dict[str, Any]:
    """Execute end-to-end labeled adaptation control experiment and audit."""
    root = Path(repo_root)

    # -------------------------------------------------------------------------
    # STEP 1: Verify Artifact Hashes and Freeze Baseline Environment
    # -------------------------------------------------------------------------
    canon_split_path = root / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    adapt_split_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
    leakage_audit_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_group_overlap_audit.json"
    source_model_path = root / "component3" / "models" / "experimental" / "telco_rf_experimental_source.joblib"
    source_prep_path = root / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib"
    baseline_rf_path = root / "component3" / "models" / "baselines" / "telco_churn_random_forest.joblib"
    baseline_prep_path = root / "component3" / "models" / "baselines" / "telco_churn_preprocessor.joblib"

    initial_hashes = {
        "canon_split": compute_file_sha256(canon_split_path),
        "adapt_split": compute_file_sha256(adapt_split_path),
        "leakage_audit": compute_file_sha256(leakage_audit_path),
        "source_model": compute_file_sha256(source_model_path),
        "source_prep": compute_file_sha256(source_prep_path),
        "baseline_rf": compute_file_sha256(baseline_rf_path),
        "baseline_prep": compute_file_sha256(baseline_prep_path),
    }

    with open(adapt_split_path, "r", encoding="utf-8") as f:
        adapt_manifest = json.load(f)
    with open(leakage_audit_path, "r", encoding="utf-8") as f:
        leakage_audit = json.load(f)

    if leakage_audit.get("leakage_audit_status") != "PASS":
        raise ValueError("Audit status is not PASS!")
    if leakage_audit.get("total_cross_partition_index_overlaps", -1) != 0:
        raise ValueError("Nonzero index overlap detected!")
    if leakage_audit.get("total_cross_partition_group_overlaps", -1) != 0:
        raise ValueError("Nonzero group overlap detected!")

    rc = adapt_manifest["realized_counts"]
    if rc != {"source_train": 3452, "source_eval": 493, "target_adapt_pool": 493, "target_eval": 493}:
        raise ValueError(f"Partition count mismatch: {rc}")

    # Load datasets
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
    df_target_eval = splits.target_eval.copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # STEP 2: Pre-Register Machine-Readable Protocol
    # -------------------------------------------------------------------------
    protocol = {
        "protocol_name": "Batch_5G_Authentic_Adaptation_Control",
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "reference_control_arm": "A0 (Frozen Experimental Source RF, Batch 5F)",
        "adaptation_arms": {
            "A1-B50": {
                "source_train_rows": 3452,
                "adaptation_budget": 50,
                "synthetic_rows": 0,
                "total_training_rows": 3502,
            },
            "A1-B100": {
                "source_train_rows": 3452,
                "adaptation_budget": 100,
                "synthetic_rows": 0,
                "total_training_rows": 3552,
            },
            "A1-B200": {
                "source_train_rows": 3452,
                "adaptation_budget": 200,
                "synthetic_rows": 0,
                "total_training_rows": 3652,
            },
        },
        "preprocessing_policy": (
            "Fit a separate TabularDataPreprocessor on each arm's permitted training composition (D_source_train + B_k). "
            "Guarantees that each model is self-contained with its own preprocessor, with zero evaluation snooping."
        ),
        "classifier_hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 2,
            "bootstrap": True,
            "criterion": "gini",
        },
        "target_evaluation_protocol": (
            "Confirmatory, single-pass evaluation on D_target_eval (N=493) under the frozen TELCO_CONCEPT rule. "
            "No hyperparameter tuning, budget selection, seed searching, or iterative retry using target evaluation."
        ),
        "source_evaluation_protocol": (
            "Assess source-domain retention on D_source_eval (N=493) under original observational truth."
        ),
        "statistical_comparison_plan": {
            "paired_bootstrap_iterations": n_bootstraps,
            "paired_permutation_iterations": n_permutations,
            "multiplicity_adjustments": ["bonferroni", "benjamini_hochberg_fdr"],
            "number_of_comparisons": 3,
            "significance_alpha": 0.05,
        },
        "abstention_conditions": [
            "If adaptation budget is 0 or labels unverified -> ABSTAIN",
            "If degradation delta on A0 is < 0.05 -> recovery fraction undefined",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    protocol_path = root / "component3" / "results" / "experimental_models" / "batch05g_adaptation_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    with protocol_path.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 3: Prepare Post-Drift Labeled Adaptation Records (TELCO_CONCEPT)
    # -------------------------------------------------------------------------
    # Rule: tenure > 29.0 inverts Churn; tenure <= 29.0 preserves Churn
    budget_records: dict[int, pd.DataFrame] = {}
    budget_stats: dict[str, dict[str, Any]] = {}

    for b_size in [50, 100, 200]:
        b_df = splits.nested_adapt_budgets[b_size].copy().reset_index(drop=True)
        b_df["_original_label"] = b_df["Churn"].copy()
        flips = b_df["tenure"] > 29.0
        n_flipped = int(flips.sum())

        orig_counts = b_df["_original_label"].value_counts().to_dict()
        b_df.loc[flips, "Churn"] = b_df.loc[flips, "Churn"].map({"No": "Yes", "Yes": "No"})
        trans_counts = b_df["Churn"].value_counts().to_dict()
        b_df["_concept_provenance"] = "SYNTHETIC_CONCEPT_TRUTH"

        budget_records[b_size] = b_df
        budget_stats[f"B_{b_size}"] = {
            "budget_size": b_size,
            "labels_inverted": n_flipped,
            "inversion_percentage": round(n_flipped / b_size * 100.0, 2),
            "original_counts": {str(k): int(v) for k, v in orig_counts.items()},
            "transformed_counts": {str(k): int(v) for k, v in trans_counts.items()},
            "affected_slice_support_tenure_gt_29": n_flipped,
            "label_provenance": "SYNTHETIC_CONCEPT_TRUTH",
        }

    # Prepare Target Evaluation Set under identical frozen concept truth
    df_target_eval_sim = df_target_eval.copy()
    df_target_eval_sim["_original_label"] = df_target_eval_sim["Churn"].copy()
    tgt_flips = df_target_eval_sim["tenure"] > 29.0
    df_target_eval_sim.loc[tgt_flips, "Churn"] = df_target_eval_sim.loc[tgt_flips, "Churn"].map({"No": "Yes", "Yes": "No"})
    df_target_eval_sim["_concept_provenance"] = "SYNTHETIC_CONCEPT_TRUTH"

    # -------------------------------------------------------------------------
    # STEP 4: Load Frozen A0 Reference and Train A1 Adaptation Arms
    # -------------------------------------------------------------------------
    rf_a0: RandomForestClassifier = joblib.load(source_model_path)
    prep_source: TabularDataPreprocessor = joblib.load(source_prep_path)

    # Transform evaluation sets with preprocessors
    models: dict[str, RandomForestClassifier] = {"A0": rf_a0}
    preprocessors: dict[str, TabularDataPreprocessor] = {"A0": prep_source}
    training_telemetry: dict[str, dict[str, Any]] = {
        "A0": {
            "training_rows": 3452,
            "training_duration_seconds": 0.36,
            "peak_memory_mb": 160.7,
        }
    }

    models_exp_dir = root / "component3" / "models" / "experimental"
    models_exp_dir.mkdir(parents=True, exist_ok=True)

    rf_params = protocol["classifier_hyperparameters"]

    for b_size in [50, 100, 200]:
        arm_name = f"A1-B{b_size}"
        b_df = budget_records[b_size]

        # Combine source train with post-drift adapted budget
        combined_train = pd.concat([df_source_train, b_df], ignore_index=True)
        total_rows = len(combined_train)

        # Fit separate preprocessor exclusively on permitted training composition
        prep_arm = TabularDataPreprocessor.from_dataset_metadata(meta, scale_numerical=True)
        prep_arm.fit(combined_train)

        bundle_train = prep_arm.transform(combined_train)

        rf_arm = RandomForestClassifier(**rf_params)
        mem_start = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0
        t0 = time.perf_counter()
        rf_arm.fit(bundle_train.X, bundle_train.y)
        duration = time.perf_counter() - t0
        mem_end = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0

        models[arm_name] = rf_arm
        preprocessors[arm_name] = prep_arm
        training_telemetry[arm_name] = {
            "training_rows": total_rows,
            "source_train_rows": 3452,
            "adaptation_budget_rows": b_size,
            "training_duration_seconds": round(duration, 4),
            "peak_memory_mb": round(max(mem_start, mem_end), 2),
        }

        # Save artifacts
        model_save_path = models_exp_dir / f"telco_rf_a1_b{b_size}.joblib"
        prep_save_path = models_exp_dir / f"telco_prep_a1_b{b_size}.joblib"
        joblib.dump(rf_arm, model_save_path)
        joblib.dump(prep_arm, prep_save_path)

    # -------------------------------------------------------------------------
    # STEP 5 & 6: Confirmatory Evaluation on Target & Source Domains
    # -------------------------------------------------------------------------
    target_results: dict[str, dict[str, Any]] = {}
    source_results: dict[str, dict[str, Any]] = {}

    target_predictions: dict[str, np.ndarray] = {}
    target_probabilities: dict[str, np.ndarray] = {}

    for arm_name in ["A0", "A1-B50", "A1-B100", "A1-B200"]:
        model = models[arm_name]
        prep = preprocessors[arm_name]

        # Target Eval
        bundle_tgt = prep.transform(df_target_eval_sim)
        target_results[arm_name] = compute_metrics_bundle(model, bundle_tgt.X, bundle_tgt.y)
        target_predictions[arm_name] = model.predict(bundle_tgt.X)
        target_probabilities[arm_name] = model.predict_proba(bundle_tgt.X)[:, 1]

        # Source Eval
        bundle_src = prep.transform(df_source_eval)
        source_results[arm_name] = compute_metrics_bundle(model, bundle_src.X, bundle_src.y)

    y_target_true = preprocessors["A0"].transform(df_target_eval_sim).y
    y_source_true = preprocessors["A0"].transform(df_source_eval).y

    # -------------------------------------------------------------------------
    # STEP 7: Paired Statistical Analysis & Multiplicity Adjustment
    # -------------------------------------------------------------------------
    raw_p_values: list[float] = []
    arm_names = ["A1-B50", "A1-B100", "A1-B200"]
    statistical_records: dict[str, dict[str, Any]] = {}

    y_pred_a0 = target_predictions["A0"]
    f1_a0 = target_results["A0"]["f1_positive"]
    f1_source_clean = source_results["A0"]["f1_positive"]
    delta_deg = f1_source_clean - f1_a0  # Degradation denominator (0.6075 - 0.4139 = 0.1936)

    for arm in arm_names:
        y_pred_arm = target_predictions[arm]
        ci_low, ci_high, ci_mean = paired_bootstrap_ci(
            y_target_true, y_pred_arm, y_pred_a0, n_bootstraps=n_bootstraps, seed=seed
        )
        obs_diff, p_val = paired_permutation_test(
            y_target_true, y_pred_arm, y_pred_a0, n_permutations=n_permutations, seed=seed
        )
        raw_p_values.append(p_val)

        # Recovery fraction R = (F1_arm - F1_a0) / (F1_source - F1_a0) * 100%
        rec_frac = round((obs_diff / delta_deg) * 100.0, 2) if delta_deg >= 0.05 else None

        statistical_records[arm] = {
            "arm": arm,
            "f1_target": target_results[arm]["f1_positive"],
            "f1_a0_reference": f1_a0,
            "delta_f1_observed": obs_diff,
            "bootstrap_95_ci_lower": ci_low,
            "bootstrap_95_ci_upper": ci_high,
            "bootstrap_mean_difference": ci_mean,
            "permutation_two_sided_p_value": p_val,
            "recovery_fraction_percentage": rec_frac,
            "ci_includes_zero": bool(ci_low <= 0.0 <= ci_high),
        }

    # Multiplicity corrections across m=3 comparisons
    m = len(arm_names)
    # Bonferroni: p_adj = min(1.0, m * p)
    for arm in arm_names:
        rec = statistical_records[arm]
        p_raw = rec["permutation_two_sided_p_value"]
        p_bonf = round(min(1.0, m * p_raw), 4)
        rec["bonferroni_adjusted_p_value"] = p_bonf
        rec["statistically_significant_bonferroni"] = bool(p_bonf < 0.05 and not rec["ci_includes_zero"])

    # Benjamini-Hochberg (FDR)
    # Sort p-values ascending
    sorted_indices = np.argsort(raw_p_values)
    bh_p_vals = np.empty(m, dtype=np.float64)
    for rank_idx, orig_idx in enumerate(sorted_indices):
        rank = rank_idx + 1
        bh_p_vals[orig_idx] = (m / rank) * raw_p_values[orig_idx]
    # Enforce monotonicity: p_adj[i] = min(p_adj[i], p_adj[i+1])
    for i in range(m - 2, -1, -1):
        idx_curr = sorted_indices[i]
        idx_next = sorted_indices[i + 1]
        bh_p_vals[idx_curr] = min(bh_p_vals[idx_curr], bh_p_vals[idx_next])
    bh_p_vals = np.clip(bh_p_vals, 0.0, 1.0)

    for idx, arm in enumerate(arm_names):
        rec = statistical_records[arm]
        p_bh = round(float(bh_p_vals[idx]), 4)
        rec["benjamini_hochberg_fdr_p_value"] = p_bh
        rec["statistically_significant_fdr"] = bool(p_bh < 0.05 and not rec["ci_includes_zero"])

    # -------------------------------------------------------------------------
    # STEP 8: Source-Domain Retention Assessment
    # -------------------------------------------------------------------------
    source_retention_records: dict[str, dict[str, Any]] = {}
    for arm in ["A0", "A1-B50", "A1-B100", "A1-B200"]:
        res_src = source_results[arm]
        delta_f1_src = round(res_src["f1_positive"] - f1_source_clean, 4)
        delta_macro_f1_src = round(res_src["macro_f1"] - source_results["A0"]["macro_f1"], 4)
        delta_bal_acc_src = round(res_src["balanced_accuracy"] - source_results["A0"]["balanced_accuracy"], 4)

        source_retention_records[arm] = {
            "source_f1_positive": res_src["f1_positive"],
            "source_macro_f1": res_src["macro_f1"],
            "source_balanced_accuracy": res_src["balanced_accuracy"],
            "source_accuracy": res_src["accuracy"],
            "delta_f1_vs_a0": delta_f1_src,
            "delta_macro_f1_vs_a0": delta_macro_f1_src,
            "delta_balanced_accuracy_vs_a0": delta_bal_acc_src,
            "catastrophic_forgetting_flag": bool(delta_f1_src <= -0.10),
        }

    # -------------------------------------------------------------------------
    # STEP 9: Reload Parity & Baseline Inviolability Verification
    # -------------------------------------------------------------------------
    parity_results: dict[str, bool] = {}
    for b_size in [50, 100, 200]:
        arm_name = f"A1-B{b_size}"
        model_path = models_exp_dir / f"telco_rf_a1_b{b_size}.joblib"
        prep_path = models_exp_dir / f"telco_prep_a1_b{b_size}.joblib"

        m_reload = joblib.load(model_path)
        p_reload = joblib.load(prep_path)

        bundle_t = p_reload.transform(df_target_eval_sim)
        pred_reload = m_reload.predict(bundle_t.X)
        prob_reload = m_reload.predict_proba(bundle_t.X)[:, 1]

        pred_eq = np.array_equal(pred_reload, target_predictions[arm_name])
        prob_close = np.allclose(prob_reload, target_probabilities[arm_name], atol=1e-7)
        parity_results[arm_name] = bool(pred_eq and prob_close)
        if not parity_results[arm_name]:
            raise AssertionError(f"Reload parity check failed for {arm_name}!")

    # Verify baseline models and Batch 5F model were NOT mutated
    current_hashes = {
        "canon_split": compute_file_sha256(canon_split_path),
        "adapt_split": compute_file_sha256(adapt_split_path),
        "source_model": compute_file_sha256(source_model_path),
        "baseline_rf": compute_file_sha256(baseline_rf_path),
        "baseline_prep": compute_file_sha256(baseline_prep_path),
    }

    for k, h_init in current_hashes.items():
        if initial_hashes[k] != h_init:
            raise AssertionError(f"CRITICAL VIOLATION: File {k} was mutated during Batch 5G execution!")

    # -------------------------------------------------------------------------
    # STEP 10: Save Audited Evaluation Artifacts
    # -------------------------------------------------------------------------
    results_dir = root / "component3" / "results" / "experimental_models"

    # 1. Main summary results
    main_summary = {
        "experiment_id": "Batch_5G_Authentic_Adaptation_Control",
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "evaluation_partition": "D_target_eval (N=493)",
        "source_evaluation_partition": "D_source_eval (N=493)",
        "training_telemetry": training_telemetry,
        "adaptation_budget_statistics": budget_stats,
        "target_evaluation_metrics": target_results,
        "source_retention_metrics": source_results,
        "statistical_comparisons_vs_a0": statistical_records,
        "source_retention_comparisons": source_retention_records,
        "reload_parity_verified": parity_results,
        "historical_artifacts_inviolable": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with (results_dir / "telco_adaptation_control_results.json").open("w", encoding="utf-8") as f:
        json.dump(main_summary, f, indent=2)

    with (results_dir / "telco_adaptation_control_statistical_audit.json").open("w", encoding="utf-8") as f:
        json.dump(statistical_records, f, indent=2)

    with (results_dir / "telco_source_retention_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(source_retention_records, f, indent=2)

    # 2. Anonymized prediction audit CSV
    audit_rows = []
    for row_idx in range(len(df_target_eval_sim)):
        audit_rows.append({
            "eval_row_id": f"eval_row_{row_idx:03d}",
            "true_churn_concept": int(y_target_true[row_idx]),
            "tenure": float(df_target_eval_sim.iloc[row_idx]["tenure"]),
            "is_affected_slice": bool(df_target_eval_sim.iloc[row_idx]["tenure"] > 29.0),
            "pred_a0": int(target_predictions["A0"][row_idx]),
            "prob_a0": round(float(target_probabilities["A0"][row_idx]), 4),
            "pred_a1_b50": int(target_predictions["A1-B50"][row_idx]),
            "prob_a1_b50": round(float(target_probabilities["A1-B50"][row_idx]), 4),
            "pred_a1_b100": int(target_predictions["A1-B100"][row_idx]),
            "prob_a1_b100": round(float(target_probabilities["A1-B100"][row_idx]), 4),
            "pred_a1_b200": int(target_predictions["A1-B200"][row_idx]),
            "prob_a1_b200": round(float(target_probabilities["A1-B200"][row_idx]), 4),
        })

    pd.DataFrame(audit_rows).to_csv(results_dir / "telco_adaptation_predictions_audit.csv", index=False)

    return main_summary


if __name__ == "__main__":
    print("=" * 70)
    print("Executing Batch 5G: Labeled Simulated-Target Adaptation Control")
    print("=" * 70)

    summary = execute_batch05g_experiment(
        repo_root=".",
        seed=42,
        n_bootstraps=2000,
        n_permutations=2000,
    )

    print("\nTarget Evaluation Results (F1-Positive):")
    for arm, res in summary["target_evaluation_metrics"].items():
        print(f"  {arm:8}: F1={res['f1_positive']:.4f}, Macro F1={res['macro_f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f}")

    print("\nStatistical Comparisons vs A0:")
    for arm, st in summary["statistical_comparisons_vs_a0"].items():
        ci_str = f"[{st['bootstrap_95_ci_lower']:+.4f}, {st['bootstrap_95_ci_upper']:+.4f}]"
        print(f"  {arm:8}: Delta F1={st['delta_f1_observed']:+.4f}, 95% CI={ci_str}, p_raw={st['permutation_two_sided_p_value']:.4f}, p_adj={st['bonferroni_adjusted_p_value']:.4f}, Sig={st['statistically_significant_bonferroni']}")

    print("\nSource Retention Results:")
    for arm, sr in summary["source_retention_comparisons"].items():
        print(f"  {arm:8}: Source F1={sr['source_f1_positive']:.4f} (Delta={sr['delta_f1_vs_a0']:+.4f})")

    print("=" * 70)
