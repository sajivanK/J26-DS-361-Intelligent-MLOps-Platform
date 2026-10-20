"""Sprint 06 — Batch 5K: Controlled Synthetic Repair Utility Evaluation.

Evaluates downstream Random Forest models retrained with synthetic post-drift data
from TVAE, CTGAN, and TabDDPM (Arms S1-S3, K=50) against authoritative authentic controls:
- B0: Frozen unadapted source model (D_source_train, N=3,452)
- B1: Naive authentic adaptation pooling (D_source_train + B200, N=3,652)
- B2: Authentic Random OverSampling (D_source_train + B200 + B200_dup, N=3,852)
- B4: Target-only adaptation (B200, N=200)

Primary evaluation: clean, group-isolated D_dev_val (N=292).
Secondary evaluation: D_source_eval (N=493) for source retention diagnosis.
Locked D_target_eval (N=493) is strictly untouched and forbidden.

Extended K=200 arms (S4, S5, S6) are pre-registered as INFEASIBLE under the
verified multi-tier eligibility policy due to resource budgets and billing plausibility.
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
from scipy.spatial.distance import cdist
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
import torch

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.tvae_generator import TVAEGenerator
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


def filter_plausible_unique_candidates(
    df_cand: pd.DataFrame,
    df_train_b200: pd.DataFrame,
    meta: Any,
    eval_signatures: dict[str, set[tuple]],
    continuous_cols: list[str],
    discrete_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute multi-tier eligibility screening on raw synthetic candidates."""
    n_requested = len(df_cand)
    feat_cols = meta.feature_columns

    # Tier 1: Structural
    nan_inf_mask = df_cand.isna().any(axis=1) | np.isinf(df_cand[continuous_cols]).any(axis=1)
    valid_domains = {col: set(df_train_b200[col].astype(str).unique()) for col in discrete_cols}
    invalid_cat_mask = pd.Series(False, index=df_cand.index)
    for col in discrete_cols:
        if col in df_cand.columns:
            invalid_cat_mask = invalid_cat_mask | (~df_cand[col].astype(str).isin(valid_domains[col]))

    tier1_pass_mask = (~nan_inf_mask) & (~invalid_cat_mask)
    t1_count = int(tier1_pass_mask.sum())

    # Tier 2: Hard physical constraints
    neg_tenure = df_cand["tenure"] < 0.0
    gt72_tenure = df_cand["tenure"] > 72.0
    neg_monthly = df_cand["MonthlyCharges"] < 0.0
    neg_total = df_cand["TotalCharges"] < 0.0
    phys_violation = neg_tenure | gt72_tenure | neg_monthly | neg_total
    tier2_phys_mask = tier1_pass_mask & (~phys_violation)
    t2_count = int(tier2_phys_mask.sum())

    # Deduplication & Memorization checks on Tier 2 survivors
    stage2_df = df_cand[tier2_phys_mask].copy()
    stage3_dedup_mask = ~stage2_df.duplicated()

    b200_all_tuples = set(tuple(str(v) for v in r) for r in df_train_b200.itertuples(index=False))
    stage3_b200_nomatch_mask = pd.Series(
        [tuple(str(v) for v in r) not in b200_all_tuples for r in stage2_df.itertuples(index=False)],
        index=stage2_df.index,
    )

    stage3_leak_mask = pd.Series(True, index=stage2_df.index)
    for part_name, sig_set in eval_signatures.items():
        part_match = pd.Series(
            [tuple(str(v) for v in r) not in sig_set for r in stage2_df[feat_cols].itertuples(index=False)],
            index=stage2_df.index,
        )
        stage3_leak_mask = stage3_leak_mask & part_match

    stage3_df = stage2_df[stage3_dedup_mask & stage3_b200_nomatch_mask & stage3_leak_mask]
    t3_unique_count = len(stage3_df)

    # Tier 3: Plausibility / Billing ratio
    safe_t = stage3_df["tenure"].clip(lower=1.0)
    exp_tc = safe_t * stage3_df["MonthlyCharges"]
    billing_ratio = stage3_df["TotalCharges"] / (exp_tc.clip(lower=1e-6))
    plausible_df = stage3_df[(billing_ratio >= 0.5) & (billing_ratio <= 1.5)].copy().reset_index(drop=True)
    t4_plausible_count = len(plausible_df)

    stats = {
        "n_requested": n_requested,
        "tier1_structural_pass": t1_count,
        "tier2_physical_pass": t2_count,
        "tier2_unique_hard_valid": t3_unique_count,
        "tier3_plausible_unique": t4_plausible_count,
        "yield_pct": round(t4_plausible_count / max(1, n_requested) * 100, 2),
    }

    return plausible_df, stats


def compute_privacy_metrics(
    df_synth: pd.DataFrame,
    df_b200: pd.DataFrame,
    continuous_cols: list[str],
) -> dict[str, float]:
    """Compute nearest-neighbor DCR privacy metrics on min-max normalized continuous features."""
    b200_num = df_b200[continuous_cols].values
    num_mins = b200_num.min(axis=0)
    num_maxs = b200_num.max(axis=0)
    num_ranges = np.where((num_maxs - num_mins) == 0, 1.0, num_maxs - num_mins)

    b200_norm = (b200_num - num_mins) / num_ranges
    synth_norm = (df_synth[continuous_cols].values - num_mins) / num_ranges

    dists = cdist(synth_norm, b200_norm, metric="euclidean")
    min_dists = dists.min(axis=1)

    return {
        "dcr_p05": round(float(np.percentile(min_dists, 5)), 4),
        "dcr_median": round(float(np.median(min_dists)), 4),
        "near_duplicate_count": int((min_dists < 0.01).sum()),
    }


def execute_synthetic_repair_benchmark(
    seed: int = 42,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Batch 5K controlled synthetic repair benchmark."""
    t_start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = Path(base_dir) if base_dir else Path("component3").resolve().parent
    models_dir = root / "component3" / "models" / "experimental"
    results_dir = root / "component3" / "results" / "experimental_models"
    figures_dir = root / "component3" / "results" / "figures"
    gen_results_dir = root / "component3" / "results" / "generators"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Input Verification and Cryptographic Integrity
    # -------------------------------------------------------------------------
    canon_split_path = root / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    adapt_split_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
    leakage_audit_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_group_overlap_audit.json"
    source_model_path = root / "component3" / "models" / "experimental" / "telco_rf_experimental_source.joblib"
    source_prep_path = root / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib"

    expected_hashes = {
        "canonical_split": "e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68",
        "adaptation_split": "34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59",
        "leakage_audit": "884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518",
        "source_model": "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0",
        "source_prep": "a188f0285e6cf07741bbb69be8b3df44eece492e69166c78d92353f1f4d41224",
    }

    verified_hashes = {}
    for name, expected_h in expected_hashes.items():
        p = {
            "canonical_split": canon_split_path,
            "adaptation_split": adapt_split_path,
            "leakage_audit": leakage_audit_path,
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
    df_target_eval = splits.target_eval.copy().reset_index(drop=True)

    # Confirm partition counts
    if len(df_source_train) != 3452:
        raise ValueError(f"D_source_train must have 3452 rows, got {len(df_source_train)}")
    if len(df_source_eval) != 493:
        raise ValueError(f"D_source_eval must have 493 rows, got {len(df_source_eval)}")
    if len(df_target_eval) != 493:
        raise ValueError(f"D_target_eval must have 493 rows, got {len(df_target_eval)}")

    # Isolate B200 and clean D_dev_val (N=292)
    b200_df = splits.nested_adapt_budgets[200].copy().reset_index(drop=True)
    if len(b200_df) != 200:
        raise ValueError(f"B200 must have 200 rows, got {len(b200_df)}")

    b200_cust_ids = set(b200_df["customerID"])
    raw_unused = df_target_adapt_pool[~df_target_adapt_pool["customerID"].isin(b200_cust_ids)].copy().reset_index(drop=True)

    b200_sigs = set(tuple(str(v) for v in r) for r in b200_df[feat_cols].itertuples(index=False))
    unused_sigs = [tuple(str(v) for v in r) for r in raw_unused[feat_cols].itertuples(index=False)]
    overlap_idx = [i for i, s in enumerate(unused_sigs) if s in b200_sigs]
    clean_dev_val = raw_unused.drop(index=overlap_idx).copy().reset_index(drop=True)

    if len(clean_dev_val) != 292:
        raise ValueError(f"Clean dev_val must have exactly 292 records, got {len(clean_dev_val)}")

    # Apply exact TELCO_CONCEPT ground truth to B200 and D_dev_val
    b200_flips = b200_df["tenure"] > 29.0
    b200_post = b200_df.copy()
    b200_post.loc[b200_flips, "Churn"] = b200_post.loc[b200_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    dev_flips = clean_dev_val["tenure"] > 29.0
    clean_dev_val_post = clean_dev_val.copy()
    clean_dev_val_post.loc[dev_flips, "Churn"] = clean_dev_val_post.loc[dev_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    # Evaluation signatures for leakage audit
    eval_sigs = {
        "D_source_train": set(tuple(str(v) for v in r) for r in df_source_train[feat_cols].itertuples(index=False)),
        "D_source_eval": set(tuple(str(v) for v in r) for r in df_source_eval[feat_cols].itertuples(index=False)),
        "D_dev_val": set(tuple(str(v) for v in r) for r in clean_dev_val_post[feat_cols].itertuples(index=False)),
        "D_target_eval": set(tuple(str(v) for v in r) for r in df_target_eval[feat_cols].itertuples(index=False)),
    }

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

    gen_input = bundle_b200.generator_df.copy()
    gen_input["Churn"] = b200_post["Churn"].values
    discrete_cols = bundle_b200.generator_discrete_columns + ["Churn"]
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # -------------------------------------------------------------------------
    # STEP 2: Candidate Sampling and Multi-Tier Quality Screening
    # -------------------------------------------------------------------------
    candidate_pools: dict[str, pd.DataFrame] = {}
    candidate_stats: dict[str, Any] = {}
    candidate_hashes: dict[str, str] = {}
    candidate_privacy: dict[str, Any] = {}
    generator_runtimes: dict[str, float] = {}

    # TVAE: Extract first 50 valid unique plausible candidates from Batch 5J candidate pool
    t_gen_start = time.perf_counter()
    tvae_csv_path = gen_results_dir / "batch05j_synthetic_candidates_tvae.csv"
    if not tvae_csv_path.is_file():
        raise FileNotFoundError(f"Missing TVAE candidate pool: {tvae_csv_path}")
    raw_tvae = pd.read_csv(tvae_csv_path)
    plausible_tvae, tvae_stats = filter_plausible_unique_candidates(
        df_cand=raw_tvae,
        df_train_b200=gen_input,
        meta=meta,
        eval_signatures=eval_sigs,
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
    )
    if len(plausible_tvae) < 50:
        raise ValueError(f"TVAE candidate pool insufficient: expected >= 50, got {len(plausible_tvae)}")
    tvae_50 = plausible_tvae.iloc[:50].copy().reset_index(drop=True)
    tvae_sample_time = round(time.perf_counter() - t_gen_start, 2)
    generator_runtimes["TVAE"] = tvae_sample_time
    candidate_pools["TVAE"] = tvae_50
    candidate_stats["TVAE"] = tvae_stats
    candidate_privacy["TVAE"] = compute_privacy_metrics(tvae_50, gen_input, continuous_cols)

    # CTGAN: Sample N=500 from fitted generator with seed 42 to resolve candidate shortage
    t_gen_start = time.perf_counter()
    ctgan = CTGANGenerator(epochs=150, discrete_columns=discrete_cols)
    ctgan.fit(gen_input)
    raw_ctgan = ctgan.sample(500)
    plausible_ctgan, ctgan_stats = filter_plausible_unique_candidates(
        df_cand=raw_ctgan,
        df_train_b200=gen_input,
        meta=meta,
        eval_signatures=eval_sigs,
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
    )
    if len(plausible_ctgan) < 50:
        raise ValueError(f"CTGAN candidate pool insufficient: expected >= 50, got {len(plausible_ctgan)}")
    ctgan_50 = plausible_ctgan.iloc[:50].copy().reset_index(drop=True)
    ctgan_sample_time = round(time.perf_counter() - t_gen_start, 2)
    generator_runtimes["CTGAN"] = ctgan_sample_time
    candidate_pools["CTGAN"] = ctgan_50
    candidate_stats["CTGAN"] = ctgan_stats
    candidate_privacy["CTGAN"] = compute_privacy_metrics(ctgan_50, gen_input, continuous_cols)

    # TabDDPM: Sample N=500 from fitted generator with seed 42 to resolve candidate shortage
    t_gen_start = time.perf_counter()
    tabddpm = TabDDPMGenerator(
        epochs=20,
        batch_size=64,
        num_timesteps=50,
        learning_rate=0.001,
        discrete_columns=discrete_cols,
    )
    tabddpm.fit(gen_input)
    raw_tabddpm = tabddpm.sample(500)
    plausible_tabddpm, tabddpm_stats = filter_plausible_unique_candidates(
        df_cand=raw_tabddpm,
        df_train_b200=gen_input,
        meta=meta,
        eval_signatures=eval_sigs,
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
    )
    if len(plausible_tabddpm) < 50:
        raise ValueError(f"TabDDPM candidate pool insufficient: expected >= 50, got {len(plausible_tabddpm)}")
    tabddpm_50 = plausible_tabddpm.iloc[:50].copy().reset_index(drop=True)
    tabddpm_sample_time = round(time.perf_counter() - t_gen_start, 2)
    generator_runtimes["TabDDPM"] = tabddpm_sample_time
    candidate_pools["TabDDPM"] = tabddpm_50
    candidate_stats["TabDDPM"] = tabddpm_stats
    candidate_privacy["TabDDPM"] = compute_privacy_metrics(tabddpm_50, gen_input, continuous_cols)

    # Save frozen candidate CSV artifacts
    for g_name, df_cand in candidate_pools.items():
        csv_filename = f"batch05k_candidates_{g_name.lower()}_k50.csv"
        csv_path = results_dir / csv_filename
        df_cand.to_csv(csv_path, index=False)
        candidate_hashes[g_name] = compute_file_sha256(csv_path)

    # -------------------------------------------------------------------------
    # STEP 3: Load Authentic Control Models (B0, B1, B2, B4)
    # -------------------------------------------------------------------------
    control_models = {
        "B0": joblib.load(source_model_path),
        "B1_B200": joblib.load(models_dir / "telco_rf_a1_b200.joblib"),
        "B2_B200_K200": joblib.load(models_dir / "telco_rf_b2_b200_k200.joblib"),
        "B4_B200": joblib.load(models_dir / "telco_rf_b4_b200.joblib"),
    }

    # -------------------------------------------------------------------------
    # STEP 4: Train Downstream Random Forest Models for S1, S2, S3
    # -------------------------------------------------------------------------
    synthetic_models = {}
    synthetic_training_times = {}

    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 2,
        "bootstrap": True,
        "criterion": "gini",
    }

    synthetic_arms = {
        "S1_TVAE_K50": ("TVAE", tvae_50),
        "S2_CTGAN_K50": ("CTGAN", ctgan_50),
        "S3_TabDDPM_K50": ("TabDDPM", tabddpm_50),
    }

    for arm_id, (gen_name, df_cand) in synthetic_arms.items():
        t_train_start = time.perf_counter()
        bundle_synth = source_prep.transform(df_cand)
        X_train_arm = np.vstack([X_src, X_b200, bundle_synth.X])
        y_train_arm = np.concatenate([y_src, y_b200, bundle_synth.y])

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train_arm, y_train_arm)
        train_time = round(time.perf_counter() - t_train_start, 3)

        # Save model and preprocessor copy
        model_filename = f"telco_rf_experimental_{arm_id.lower()}.joblib"
        prep_filename = f"telco_prep_experimental_{arm_id.lower()}.joblib"
        joblib.dump(rf, models_dir / model_filename)
        joblib.dump(source_prep, models_dir / prep_filename)

        synthetic_models[arm_id] = rf
        synthetic_training_times[arm_id] = train_time

    # Combine all evaluated models
    all_models = {**control_models, **synthetic_models}

    # -------------------------------------------------------------------------
    # STEP 5: Evaluate Primary Development Utility and Source Retention
    # -------------------------------------------------------------------------
    dev_metrics = {}
    source_metrics = {}
    predictions_dev = {}
    probabilities_dev = {}

    for arm_id, model in all_models.items():
        dev_metrics[arm_id] = compute_metrics_bundle(model, X_dev, y_dev)
        source_metrics[arm_id] = compute_metrics_bundle(model, X_seval, y_seval)
        predictions_dev[arm_id] = model.predict(X_dev)
        probabilities_dev[arm_id] = model.predict_proba(X_dev)[:, 1]

    # -------------------------------------------------------------------------
    # STEP 6: Statistical Uncertainty and Pairwise Comparisons
    # -------------------------------------------------------------------------
    comparisons = {}
    raw_p_values = []
    comparison_keys = []

    reference_arms = ["B0", "B1_B200", "B2_B200_K200"]

    for arm_id in synthetic_arms.keys():
        arm_comp = {}
        for ref_id in reference_arms:
            cand_preds = predictions_dev[arm_id]
            ref_preds = predictions_dev[ref_id]

            delta_f1 = round(dev_metrics[arm_id]["positive_f1"] - dev_metrics[ref_id]["positive_f1"], 4)
            boot_ci = paired_bootstrap_ci(y_dev, cand_preds, ref_preds, n_bootstraps=2000, seed=seed)
            perm_res = paired_permutation_test(y_dev, cand_preds, ref_preds, n_permutations=2000, seed=seed)

            raw_p_values.append(perm_res["p_value"])
            comp_key = (arm_id, ref_id)
            comparison_keys.append(comp_key)

            arm_comp[ref_id] = {
                "delta_f1": delta_f1,
                "bootstrap_ci_95": [boot_ci["lower_ci"], boot_ci["upper_ci"]],
                "bootstrap_mean_diff": boot_ci["mean_diff"],
                "permutation_p_value": perm_res["p_value"],
                "statistically_significant_nominal": bool(perm_res["p_value"] < 0.05),
            }
        comparisons[arm_id] = arm_comp

    # Apply step-down Holm-Bonferroni correction
    adjusted_p_values = holm_bonferroni_correction(raw_p_values)
    for (arm_id, ref_id), adj_p in zip(comparison_keys, adjusted_p_values):
        comparisons[arm_id][ref_id]["holm_bonferroni_p_value"] = adj_p
        comparisons[arm_id][ref_id]["statistically_significant_adjusted"] = bool(adj_p < 0.05)

    # -------------------------------------------------------------------------
    # STEP 7: Generate Anonymous Prediction-Level Audit CSV
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
            "y_true": int(y_dev[i]),
        }
        for arm_id in all_models.keys():
            row_dict[f"pred_{arm_id}"] = int(predictions_dev[arm_id][i])
            row_dict[f"prob_{arm_id}"] = round(float(probabilities_dev[arm_id][i]), 4)
        audit_rows.append(row_dict)

    df_audit = pd.DataFrame(audit_rows)
    audit_csv_path = results_dir / "batch05k_repair_benchmark_predictions_audit.csv"
    df_audit.to_csv(audit_csv_path, index=False)

    # -------------------------------------------------------------------------
    # STEP 8: Protocol, Results, and Comparison JSON Serialization
    # -------------------------------------------------------------------------
    total_elapsed = round(time.perf_counter() - t_start, 2)
    process_mem = psutil.Process().memory_info().rss / (1024 * 1024)

    protocol_summary = {
        "protocol_name": "Batch_5K_Controlled_Synthetic_Repair_Utility_Evaluation",
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "execution_environment": {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
            "total_elapsed_seconds": total_elapsed,
            "peak_process_rss_mb": round(process_mem, 2),
        },
        "verified_hashes": verified_hashes,
        "candidate_file_hashes": candidate_hashes,
        "partition_sizes": {
            "D_source_train": len(df_source_train),
            "B200": len(b200_df),
            "D_dev_val": len(clean_dev_val_post),
            "D_source_eval": len(df_source_eval),
            "D_target_eval": len(df_target_eval),
        },
        "comparison_arms": {
            "B0": "Frozen source model (N=3,452)",
            "B1_B200": "Naive authentic pooling (N=3,652)",
            "B2_B200_K200": "Authentic Random OverSampling (N=3,852)",
            "B4_B200": "Target-only B200 model (N=200)",
            "S1_TVAE_K50": "Source + B200 + 50 TVAE records (N=3,702)",
            "S2_CTGAN_K50": "Source + B200 + 50 CTGAN records (N=3,702)",
            "S3_TabDDPM_K50": "Source + B200 + 50 TabDDPM records (N=3,702)",
        },
        "infeasible_arms": {
            "S4_TVAE_K200": "Pre-registered INFEASIBLE: sampling budget limit and continuous distortion",
            "S5_CTGAN_K200": "Pre-registered INFEASIBLE: sampling budget limit and continuous distortion",
            "S6_TabDDPM_K200": "Pre-registered INFEASIBLE: sampling budget limit and continuous distortion",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    results_summary = {
        "dev_val_metrics": dev_metrics,
        "source_retention_metrics": source_metrics,
        "candidate_screening_funnel": candidate_stats,
        "candidate_privacy_metrics": candidate_privacy,
        "generator_runtimes_seconds": generator_runtimes,
        "classifier_training_times_seconds": synthetic_training_times,
    }

    with open(results_dir / "batch05k_repair_benchmark_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol_summary, f, indent=2)

    with open(results_dir / "batch05k_repair_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    with open(results_dir / "batch05k_repair_benchmark_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 9: Visualization Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    arm_order = ["B0", "B1_B200", "B2_B200_K200", "B4_B200", "S1_TVAE_K50", "S2_CTGAN_K50", "S3_TabDDPM_K50"]
    colors = ["#4a5568", "#2b6cb0", "#2c5282", "#9f7aea", "#319795", "#d69e2e", "#dd6b20"]

    dev_f1_vals = [dev_metrics[a]["positive_f1"] for a in arm_order]
    src_f1_vals = [source_metrics[a]["positive_f1"] for a in arm_order]

    # Panel A: Adaptation Utility on D_dev_val
    bars_dev = axes[0].bar(arm_order, dev_f1_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Post-Drift Adaptation Utility (D_dev_val, N=292)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Positive Churn F1 Score", fontsize=11)
    axes[0].set_ylim(0, 0.9)
    axes[0].axhline(y=dev_metrics["B2_B200_K200"]["positive_f1"], color="#e53e3e", linestyle="--", label=f"B2 Authentic ROS ({dev_metrics['B2_B200_K200']['positive_f1']})")
    axes[0].axhline(y=dev_metrics["B0"]["positive_f1"], color="#718096", linestyle=":", label=f"B0 Frozen Source ({dev_metrics['B0']['positive_f1']})")
    axes[0].set_xticklabels(arm_order, rotation=35, ha="right", fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=9)

    for bar in bars_dev:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, height + 0.015, f"{height:.4f}", ha="center", va="bottom", fontsize=8)

    # Panel B: Source Retention on D_source_eval
    bars_src = axes[1].bar(arm_order, src_f1_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("Source Concept Retention (D_source_eval, N=493)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Source Positive F1 Score", fontsize=11)
    axes[1].set_ylim(0, 0.75)
    axes[1].axhline(y=source_metrics["B0"]["positive_f1"], color="#718096", linestyle=":", label=f"B0 Source Model ({source_metrics['B0']['positive_f1']})")
    axes[1].set_xticklabels(arm_order, rotation=35, ha="right", fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=9)

    for bar in bars_src:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height + 0.015, f"{height:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot_path = figures_dir / "batch05k_repair_benchmark_metrics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return {
        "status": "SUCCESS",
        "total_elapsed_seconds": total_elapsed,
        "dev_metrics": dev_metrics,
        "source_metrics": source_metrics,
        "comparisons": comparisons,
        "candidate_stats": candidate_stats,
        "candidate_privacy": candidate_privacy,
        "candidate_hashes": candidate_hashes,
    }


if __name__ == "__main__":
    print("=" * 75)
    print("SPRINT 06 — BATCH 5K: CONTROLLED SYNTHETIC REPAIR UTILITY EVALUATION")
    print("=" * 75)
    res = execute_synthetic_repair_benchmark()
    print("\nBenchmark Execution Complete.")
    print("\nDevelopment Validation Metrics (D_dev_val, N=292):")
    for arm, m in res["dev_metrics"].items():
        print(f"  {arm:16s} | Pos F1: {m['positive_f1']:.4f} | Macro F1: {m['macro_f1']:.4f} | Bal Acc: {m['balanced_accuracy']:.4f}")
    print("\nSource Retention Metrics (D_source_eval, N=493):")
    for arm, m in res["source_metrics"].items():
        print(f"  {arm:16s} | Source Pos F1: {m['positive_f1']:.4f} | Macro F1: {m['macro_f1']:.4f}")
    print("\nPairwise Comparisons vs Authentic Controls (Delta F1 on D_dev_val):")
    for arm, comp in res["comparisons"].items():
        print(f"  {arm}:")
        for ref, c in comp.items():
            print(f"    vs {ref:14s} | Delta F1: {c['delta_f1']:+.4f} | 95% CI: {c['bootstrap_ci_95']} | p_adj: {c['holm_bonferroni_p_value']:.4f}")
