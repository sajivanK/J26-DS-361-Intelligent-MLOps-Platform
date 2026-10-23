"""Sprint 06 — Batch 5J / Batch 5J-Audit: Post-Drift Synthetic Generator Feasibility & Candidate Audit.

Evaluates CTGAN, TVAE, and TabDDPM on CPU using strictly the 200 labeled
post-drift adaptation records (B200) under the frozen TELCO_CONCEPT distribution shift.

Audit Corrections Applied:
1. Reconciled TELCO_CONCEPT label flips: all 97 rows with tenure > 29.0 are inverted
   (84 No->Yes and 13 Yes->No), producing exactly 124 Yes and 76 No.
2. Removed 'pure deterministic rule agreement' as a candidate quality metric. Replaced with
   purely descriptive conditional feature-label distributions against B200 and D_dev_val.
3. Separated independent criteria pass counts from sequential filter funnel counts.
4. Corrected frozen benchmark references:
   - B0: f1_dev = 0.4055 (clean D_dev_val, N=292), f1_source = 0.6075 (D_source_eval, N=493)
   - B2_B200_K200: f1_dev = 0.4497, f1_source = 0.5890
   - B4_B200: f1_dev = 0.7660, f1_source = 0.3492 (reconciled from 0.3089 transcription error)
5. Zero retraining in audit mode: evaluates the frozen candidate CSV artifacts.
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

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def compute_tvd(p_counts: dict[Any, int], q_counts: dict[Any, int], total_p: int, total_q: int) -> float:
    """Compute Total Variation Distance between two categorical frequency distributions."""
    all_keys = set(str(k) for k in p_counts.keys()).union(set(str(k) for k in q_counts.keys()))
    p_norm = {str(k): v / max(1, total_p) for k, v in p_counts.items()}
    q_norm = {str(k): v / max(1, total_q) for k, v in q_counts.items()}
    tvd = 0.5 * sum(abs(p_norm.get(k, 0.0) - q_norm.get(k, 0.0)) for k in all_keys)
    return float(tvd)


def evaluate_candidate_tiers(
    df_synth: pd.DataFrame,
    df_train_b200: pd.DataFrame,
    meta: Any,
    eval_signatures: dict[str, set[tuple]],
    continuous_cols: list[str],
    discrete_cols: list[str],
    clean_dev_val: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Execute rigorous three-tier validation, sequential funnel accounting, and descriptive checks."""
    n_requested = len(df_synth)
    n_returned = len(df_synth)
    feat_cols = meta.feature_columns

    # -------------------------------------------------------------------------
    # TIER 1: Structural Correctness
    # -------------------------------------------------------------------------
    expected_cols = list(df_train_b200.columns)
    col_order_match = list(df_synth.columns) == expected_cols
    missing_cols = [c for c in expected_cols if c not in df_synth.columns]
    unexpected_cols = [c for c in df_synth.columns if c not in expected_cols]

    nan_counts = df_synth.isna().sum().to_dict()
    total_nans = int(sum(nan_counts.values()))

    # Check infinite values in continuous columns
    inf_counts = {c: int(np.isinf(df_synth[c]).sum()) for c in continuous_cols if c in df_synth.columns}
    total_infs = int(sum(inf_counts.values()))

    # Check categorical domains with robust string representation
    invalid_category_counts: dict[str, int] = {}
    valid_domains = {col: set(df_train_b200[col].astype(str).unique()) for col in discrete_cols}
    for col in discrete_cols:
        if col in df_synth.columns:
            invalid_mask = ~df_synth[col].astype(str).isin(valid_domains[col])
            invalid_category_counts[col] = int(invalid_mask.sum())
    total_invalid_categories = int(sum(invalid_category_counts.values()))

    # Target representation
    target_valid = set(df_synth["Churn"].astype(str).unique()).issubset({"No", "Yes"})

    # Row-level Tier 1 mask
    row_nan_inf = df_synth.isna().any(axis=1) | np.isinf(df_synth[continuous_cols]).any(axis=1)
    row_invalid_cat = pd.Series(False, index=df_synth.index)
    for col in discrete_cols:
        row_invalid_cat = row_invalid_cat | (~df_synth[col].astype(str).isin(valid_domains[col]))

    tier1_pass_mask = (~row_nan_inf) & (~row_invalid_cat)
    tier1_structural_valid_count = int(tier1_pass_mask.sum())

    # -------------------------------------------------------------------------
    # TIER 2: Hard Physical Constraints
    # -------------------------------------------------------------------------
    neg_tenure_mask = df_synth["tenure"] < 0.0
    gt72_tenure_mask = df_synth["tenure"] > 72.0
    neg_monthly_mask = df_synth["MonthlyCharges"] < 0.0
    neg_total_mask = df_synth["TotalCharges"] < 0.0

    physical_violation_mask = neg_tenure_mask | gt72_tenure_mask | neg_monthly_mask | neg_total_mask
    tier2_physical_independent_mask = ~physical_violation_mask
    tier2_physical_valid_count = int(tier2_physical_independent_mask.sum())

    # Uniqueness within sample
    is_duplicate_within_sample = df_synth.duplicated()
    within_sample_duplicate_count = int(is_duplicate_within_sample.sum())

    # Exact training match check against B200 (all 20 columns, string normalized)
    b200_all_tuples = set(tuple(str(r_val) for r_val in r) for r in df_train_b200.itertuples(index=False))
    synth_all_tuples = [tuple(str(r_val) for r_val in r) for r in df_synth.itertuples(index=False)]
    exact_b200_match_mask = pd.Series([t in b200_all_tuples for t in synth_all_tuples], index=df_synth.index)
    exact_b200_match_count = int(exact_b200_match_mask.sum())

    # Forbidden feature signature match check (19 feature columns, string normalized)
    synth_feat_tuples = [tuple(str(r_val) for r_val in r) for r in df_synth[feat_cols].itertuples(index=False)]
    leakage_match_counts: dict[str, int] = {}
    row_leakage_mask = pd.Series(False, index=df_synth.index)
    for part_name, sig_set in eval_signatures.items():
        part_match_mask = pd.Series([t in sig_set for t in synth_feat_tuples], index=df_synth.index)
        leakage_match_counts[part_name] = int(part_match_mask.sum())
        row_leakage_mask = row_leakage_mask | part_match_mask

    # -------------------------------------------------------------------------
    # TIER 3: Empirical Plausibility
    # -------------------------------------------------------------------------
    # Billing ratio screen: TotalCharges / (max(1, tenure) * MonthlyCharges) in [0.5, 1.5]
    # Intended treatment for tenure=0: safe_tenure is clamped to 1.0, expecting TotalCharges ~ MonthlyCharges.
    safe_tenure = df_synth["tenure"].clip(lower=1.0)
    expected_tc = safe_tenure * df_synth["MonthlyCharges"]
    billing_ratio = df_synth["TotalCharges"] / (expected_tc.clip(lower=1e-6))
    billing_ratio_valid_mask = (billing_ratio >= 0.5) & (billing_ratio <= 1.5)
    billing_ratio_independent_count = int(billing_ratio_valid_mask.sum())

    # -------------------------------------------------------------------------
    # SEQUENTIAL FILTER FUNNEL ACCOUNTING
    # -------------------------------------------------------------------------
    # Stage 0: Total requested/returned
    seq_stage0_count = n_requested

    # Stage 1: Tier 1 structural survivors
    seq_stage1_mask = tier1_pass_mask
    seq_stage1_count = int(seq_stage1_mask.sum())

    # Stage 2: Tier 2 hard physical survivors (among Stage 1)
    seq_stage2_mask = seq_stage1_mask & tier2_physical_independent_mask
    seq_stage2_count = int(seq_stage2_mask.sum())

    # Stage 3: Tier 2 unique hard-valid survivors (deduplication applied to Stage 2 survivors)
    stage2_survivors = df_synth[seq_stage2_mask].copy()
    stage3_dedup_mask = ~stage2_survivors.duplicated()
    stage3_b200_match_mask = pd.Series(
        [tuple(str(v) for v in r) not in b200_all_tuples for r in stage2_survivors.itertuples(index=False)],
        index=stage2_survivors.index,
    )
    stage3_leakage_mask = pd.Series(True, index=stage2_survivors.index)
    for part_name, sig_set in eval_signatures.items():
        part_match = pd.Series(
            [tuple(str(v) for v in r) not in sig_set for r in stage2_survivors[feat_cols].itertuples(index=False)],
            index=stage2_survivors.index,
        )
        stage3_leakage_mask = stage3_leakage_mask & part_match

    stage3_survivors = stage2_survivors[stage3_dedup_mask & stage3_b200_match_mask & stage3_leakage_mask]
    seq_stage3_count = len(stage3_survivors)

    # Stage 4: Tier 3 plausible unique survivors (billing ratio applied strictly to Stage 3 survivors)
    safe_t_s3 = stage3_survivors["tenure"].clip(lower=1.0)
    exp_tc_s3 = safe_t_s3 * stage3_survivors["MonthlyCharges"]
    ratio_s3 = stage3_survivors["TotalCharges"] / (exp_tc_s3.clip(lower=1e-6))
    stage4_survivors = stage3_survivors[(ratio_s3 >= 0.5) & (ratio_s3 <= 1.5)]
    seq_stage4_count = len(stage4_survivors)

    # Statistical Similarity Metrics
    ks_results: dict[str, dict[str, float]] = {}
    for c in continuous_cols:
        ks_stat, p_val = ks_2samp(df_synth[c], df_train_b200[c])
        ks_results[c] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
        }
    mean_ks_stat = float(np.mean([v["ks_statistic"] for v in ks_results.values()]))

    tvd_results: dict[str, float] = {}
    for c in discrete_cols:
        p_c = df_synth[c].value_counts().to_dict()
        q_c = df_train_b200[c].value_counts().to_dict()
        tvd_val = compute_tvd(p_c, q_c, len(df_synth), len(df_train_b200))
        tvd_results[c] = round(tvd_val, 4)
    mean_tvd = float(np.mean(list(tvd_results.values())))

    # Distance to Closest Record (DCR) on min-max normalized continuous features
    b200_num = df_train_b200[continuous_cols].values
    num_mins = b200_num.min(axis=0)
    num_maxs = b200_num.max(axis=0)
    num_ranges = np.where((num_maxs - num_mins) == 0, 1.0, num_maxs - num_mins)

    b200_norm = (b200_num - num_mins) / num_ranges
    synth_norm = (df_synth[continuous_cols].values - num_mins) / num_ranges

    dists = cdist(synth_norm, b200_norm, metric="euclidean")
    min_dists = dists.min(axis=1)
    dcr_p05 = float(np.percentile(min_dists, 5))
    dcr_median = float(np.median(min_dists))
    near_duplicate_count = int((min_dists < 0.01).sum())

    # -------------------------------------------------------------------------
    # DESCRIPTIVE FEATURE-LABEL RELATIONSHIPS (Audited: No Deterministic Gating)
    # -------------------------------------------------------------------------
    gt29_mask = df_synth["tenure"] > 29.0
    le29_mask = ~gt29_mask

    slice_gt29_count = int(gt29_mask.sum())
    slice_le29_count = int(le29_mask.sum())

    slice_gt29_churn_yes = int((df_synth.loc[gt29_mask, "Churn"] == "Yes").sum())
    slice_gt29_churn_no = int((df_synth.loc[gt29_mask, "Churn"] == "No").sum())

    slice_le29_churn_yes = int((df_synth.loc[le29_mask, "Churn"] == "Yes").sum())
    slice_le29_churn_no = int((df_synth.loc[le29_mask, "Churn"] == "No").sum())

    churn_rate_gt29 = round(slice_gt29_churn_yes / max(1, slice_gt29_count) * 100, 2)
    churn_rate_le29 = round(slice_le29_churn_yes / max(1, slice_le29_count) * 100, 2)
    conditional_diff_delta = round(churn_rate_gt29 - churn_rate_le29, 2)

    class_counts = df_synth["Churn"].value_counts().to_dict()
    churn_yes_count = int(class_counts.get("Yes", 0))
    churn_no_count = int(class_counts.get("No", 0))
    minority_ratio = float(min(churn_yes_count, churn_no_count) / max(1, max(churn_yes_count, churn_no_count)))

    # Rejection Accounting (Purely technical, physical, uniqueness, and billing ratio reasons)
    rejection_reasons = {
        "tier1_nan_inf_count": int(row_nan_inf.sum()),
        "tier1_invalid_categories_count": int(row_invalid_cat.sum()),
        "tier2_negative_tenure_count": int(neg_tenure_mask.sum()),
        "tier2_excessive_tenure_gt72_count": int(gt72_tenure_mask.sum()),
        "tier2_negative_monthly_charges_count": int(neg_monthly_mask.sum()),
        "tier2_negative_total_charges_count": int(neg_total_mask.sum()),
        "tier2_within_sample_duplicates_count": within_sample_duplicate_count,
        "tier2_exact_training_b200_matches": exact_b200_match_count,
        "tier2_forbidden_signature_matches": leakage_match_counts,
        "tier3_billing_ratio_violations_count": int((~billing_ratio_valid_mask).sum()),
    }

    return {
        "n_requested": n_requested,
        "n_returned": n_returned,
        "independent_pass_counts": {
            "tier1_structural_valid": tier1_structural_valid_count,
            "tier1_structural_valid_pct": round(tier1_structural_valid_count / n_requested * 100, 2),
            "tier2_physical_valid": tier2_physical_valid_count,
            "tier2_physical_valid_pct": round(tier2_physical_valid_count / n_requested * 100, 2),
            "tier2_within_sample_unique": int((~is_duplicate_within_sample).sum()),
            "tier2_within_sample_unique_pct": round(int((~is_duplicate_within_sample).sum()) / n_requested * 100, 2),
            "tier2_training_non_match": int((~exact_b200_match_mask).sum()),
            "tier2_training_non_match_pct": round(int((~exact_b200_match_mask).sum()) / n_requested * 100, 2),
            "tier2_leakage_free": int((~row_leakage_mask).sum()),
            "tier2_leakage_free_pct": round(int((~row_leakage_mask).sum()) / n_requested * 100, 2),
            "tier3_billing_ratio_valid": billing_ratio_independent_count,
            "tier3_billing_ratio_valid_pct": round(billing_ratio_independent_count / n_requested * 100, 2),
        },
        "sequential_funnel_counts": {
            "stage0_requested": seq_stage0_count,
            "stage1_tier1_structural_pass": seq_stage1_count,
            "stage2_tier2_hard_physical_pass": seq_stage2_count,
            "stage3_tier2_unique_hard_valid": seq_stage3_count,
            "stage4_tier3_plausible_unique": seq_stage4_count,
            "final_plausible_unique_yield_pct": round(seq_stage4_count / n_requested * 100, 2),
        },
        "tier1_structural": {
            "col_order_match": col_order_match,
            "missing_cols": missing_cols,
            "unexpected_cols": unexpected_cols,
            "total_nans": total_nans,
            "total_infs": total_infs,
            "total_invalid_categories": total_invalid_categories,
            "target_valid": target_valid,
            "structural_valid_count": tier1_structural_valid_count,
            "structural_valid_pct": round(tier1_structural_valid_count / n_requested * 100, 2),
        },
        "tier2_hard_constraints": {
            "negative_tenure": int(neg_tenure_mask.sum()),
            "tenure_gt_72": int(gt72_tenure_mask.sum()),
            "negative_monthly_charges": int(neg_monthly_mask.sum()),
            "negative_total_charges": int(neg_total_mask.sum()),
            "hard_valid_count": seq_stage2_count,
            "hard_valid_pct": round(seq_stage2_count / n_requested * 100, 2),
            "within_sample_duplicates": within_sample_duplicate_count,
            "exact_b200_matches": exact_b200_match_count,
            "forbidden_signature_matches": leakage_match_counts,
            "unique_hard_valid_count": seq_stage3_count,
            "unique_hard_valid_pct": round(seq_stage3_count / n_requested * 100, 2),
        },
        "tier3_empirical_plausibility": {
            "billing_ratio_independent_valid_count": billing_ratio_independent_count,
            "billing_ratio_independent_valid_pct": round(billing_ratio_independent_count / n_requested * 100, 2),
            "plausible_unique_count": seq_stage4_count,
            "plausible_unique_pct": round(seq_stage4_count / n_requested * 100, 2),
            "ks_continuous": ks_results,
            "mean_ks_statistic": round(mean_ks_stat, 4),
            "tvd_discrete": tvd_results,
            "mean_tvd": round(mean_tvd, 4),
            "dcr_p05": round(dcr_p05, 4),
            "dcr_median": round(dcr_median, 4),
            "near_duplicates_under_0_01": near_duplicate_count,
        },
        "descriptive_feature_label_relationships": {
            "class_distribution": {"Yes": churn_yes_count, "No": churn_no_count},
            "minority_class_pct": round(min(churn_yes_count, churn_no_count) / n_requested * 100, 2),
            "class_balance_ratio": round(minority_ratio, 4),
            "affected_slice_tenure_gt_29": {
                "count": slice_gt29_count,
                "churn_yes": slice_gt29_churn_yes,
                "churn_no": slice_gt29_churn_no,
                "churn_rate_pct": churn_rate_gt29,
            },
            "preserved_slice_tenure_le_29": {
                "count": slice_le29_count,
                "churn_yes": slice_le29_churn_yes,
                "churn_no": slice_le29_churn_no,
                "churn_rate_pct": churn_rate_le29,
            },
            "conditional_difference_delta_pct": conditional_diff_delta,
            "training_reference_b200": {
                "overall_churn_rate_pct": 62.00,
                "churn_rate_gt_29_pct": 86.60,
                "churn_rate_le_29_pct": 38.83,
                "conditional_difference_delta_pct": 47.76,
            },
            "independent_dev_reference_dev_val": {
                "overall_churn_rate_pct": 66.44,
                "churn_rate_gt_29_pct": 89.19,
                "churn_rate_le_29_pct": 43.06,
                "conditional_difference_delta_pct": 46.13,
            },
            "scientific_interpretation": (
                "Descriptive conditional rates measure macroscopic feature-label alignment. "
                "Matching these rates does not prove individual synthetic labels are counterfactually valid."
            ),
        },
        "rejection_reasons": rejection_reasons,
    }


def audit_frozen_candidates(
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Audit existing frozen candidate CSV artifacts without retraining generators."""
    root = Path(base_dir) if base_dir else Path("component3").resolve().parent
    results_dir = root / "component3" / "results" / "generators"

    # -------------------------------------------------------------------------
    # STEP 1: Verify and Freeze Inputs
    # -------------------------------------------------------------------------
    canon_split_path = root / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    adapt_split_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
    leakage_audit_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_group_overlap_audit.json"
    source_prep_path = root / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib"

    expected_hashes = {
        "canonical_split": "e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68",
        "adaptation_split": "34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59",
        "leakage_audit": "884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518",
    }

    for name, expected_hash in expected_hashes.items():
        actual_path = {
            "canonical_split": canon_split_path,
            "adaptation_split": adapt_split_path,
            "leakage_audit": leakage_audit_path,
        }[name]
        actual_hash = compute_file_sha256(actual_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {name}: expected {expected_hash}, got {actual_hash}")

    # Load canonical dataset
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    train_indices = canon_manifest["train_indices"]
    canon_train_df = df_raw.iloc[train_indices].copy().reset_index(drop=True)

    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    df_source_train = splits.source_train.copy().reset_index(drop=True)
    df_source_eval = splits.source_eval.copy().reset_index(drop=True)
    df_target_adapt_pool = splits.target_adapt_pool.copy().reset_index(drop=True)
    df_target_eval = splits.target_eval.copy().reset_index(drop=True)

    # Carve B200 and isolate clean development validation
    b200_df = splits.nested_adapt_budgets[200].copy().reset_index(drop=True)
    if len(b200_df) != 200:
        raise ValueError(f"B200 size must be exactly 200, got {len(b200_df)}")

    feat_cols = meta.feature_columns
    b200_cust_ids = set(b200_df["customerID"])
    raw_unused = df_target_adapt_pool[~df_target_adapt_pool["customerID"].isin(b200_cust_ids)].copy().reset_index(drop=True)

    b200_signatures = set(tuple(str(v) for v in r) for r in b200_df[feat_cols].itertuples(index=False))
    unused_signatures = [tuple(str(v) for v in r) for r in raw_unused[feat_cols].itertuples(index=False)]
    overlapping_idx = [i for i, s in enumerate(unused_signatures) if s in b200_signatures]
    clean_dev_val = raw_unused.drop(index=overlapping_idx).copy().reset_index(drop=True)

    if len(clean_dev_val) != 292:
        raise ValueError(f"Expected clean dev_val size 292, got {len(clean_dev_val)}")

    # Apply exact frozen TELCO_CONCEPT rule to B200: all rows with tenure > 29.0 inverted
    b200_flips = b200_df["tenure"] > 29.0
    total_affected_tenure_gt_29 = int(b200_flips.sum())
    if total_affected_tenure_gt_29 != 97:
        raise ValueError(f"Expected exactly 97 affected records in B200, got {total_affected_tenure_gt_29}")

    b200_post = b200_df.copy()
    b200_post.loc[b200_flips, "Churn"] = b200_post.loc[b200_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    b200_post_class_counts = b200_post["Churn"].value_counts().to_dict()
    if b200_post_class_counts != {"Yes": 124, "No": 76}:
        raise ValueError(f"Unexpected post-drift B200 class distribution: {b200_post_class_counts}")

    # Apply TELCO_CONCEPT to clean dev_val for reference calculations
    dev_flips = clean_dev_val["tenure"] > 29.0
    clean_dev_val_post = clean_dev_val.copy()
    clean_dev_val_post.loc[dev_flips, "Churn"] = clean_dev_val_post.loc[dev_flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    # Evaluation partition feature signatures for leakage audits (string normalized)
    eval_signatures = {
        "D_source_train": set(tuple(str(v) for v in r) for r in df_source_train[feat_cols].itertuples(index=False)),
        "D_source_eval": set(tuple(str(v) for v in r) for r in df_source_eval[feat_cols].itertuples(index=False)),
        "D_dev_val": set(tuple(str(v) for v in r) for r in clean_dev_val[feat_cols].itertuples(index=False)),
        "D_target_eval": set(tuple(str(v) for v in r) for r in df_target_eval[feat_cols].itertuples(index=False)),
    }

    # Verify B200 has 0 feature overlap with any evaluation partition
    for part_name, sig_set in eval_signatures.items():
        overlap = len(b200_signatures.intersection(sig_set))
        if overlap != 0:
            raise ValueError(f"Leakage detected: B200 overlaps with {part_name} by {overlap} records!")

    # Load frozen source preprocessor (unrefitted)
    source_prep: TabularDataPreprocessor = joblib.load(source_prep_path)
    bundle_b200 = source_prep.transform(b200_post)

    gen_input = bundle_b200.generator_df.copy()
    gen_input["Churn"] = b200_post["Churn"].values
    discrete_cols = bundle_b200.generator_discrete_columns + ["Churn"]
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    b200_training_fingerprint = compute_dataset_fingerprint(gen_input)

    # -------------------------------------------------------------------------
    # STEP 2: Protocol Pre-Registration with Reconciled Benchmark Values
    # -------------------------------------------------------------------------
    protocol = {
        "protocol_name": "Batch_5J_Generator_Feasibility",
        "audit_version": "Batch_5J_Audit",
        "dataset_id": "telco_churn",
        "drift_scenario": "TELCO_CONCEPT",
        "hardware_environment": {
            "device": "CPU",
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "torch_version": "2.10.0+cpu",
        },
        "training_data_provenance": {
            "source_dataset_fingerprint": canon_manifest.get("train_dataset_fingerprint", ""),
            "b200_input_fingerprint": b200_training_fingerprint,
            "b200_row_count": len(gen_input),
            "b200_feature_count": gen_input.shape[1],
            "b200_affected_tenure_gt_29_flips": 97,
            "b200_class_distribution": b200_post_class_counts,
            "source_preprocessor_path": str(source_prep_path),
            "source_preprocessor_refitted": False,
        },
        "strategy": {
            "chosen_strategy": "Strategy A: Joint Generation (Features + Post-Drift Target Churn)",
            "rejected_strategy": "Strategy B: Class-Specific Generation",
            "justification": (
                "Post-drift B200 contains only 76 minority ('No') and 124 majority ('Yes') records. "
                "Splitting 200 samples across 19 features into class-specific pools causes severe sample "
                "starvation, extreme mode collapse, and numerical instability. Joint generation trains on all "
                "200 records and enables direct empirical evaluation of macroscopic conditional associations."
            ),
        },
        "generator_configurations": {
            "TVAE": {
                "epochs": 150,
                "discrete_columns_count": len(discrete_cols),
                "device": "cpu",
                "sample_count": 200,
            },
            "CTGAN": {
                "epochs": 150,
                "discrete_columns_count": len(discrete_cols),
                "device": "cpu",
                "sample_count": 200,
            },
            "TabDDPM": {
                "epochs": 20,
                "batch_size": 64,
                "num_timesteps": 50,
                "learning_rate": 0.001,
                "discrete_columns_count": len(discrete_cols),
                "device": "cpu",
                "sample_count": 200,
            },
        },
        "validation_tiers": {
            "Tier_1_Structural": "Schema matching, correct column count/order, 0 NaNs, 0 Infs, valid categories",
            "Tier_2_Hard_Constraints": "Physical ranges (tenure in [0, 72], non-negative charges), unique candidates, 0 exact B200 copies, 0 forbidden signature overlaps",
            "Tier_3_Empirical_Plausibility": "Billing ratio screen in [0.5, 1.5], continuous KS-test, categorical TVD, DCR p05 nearest-neighbor privacy",
        },
        "frozen_benchmark_references_reconciled": {
            "B0_Frozen_Source": {
                "f1_dev": 0.4055,
                "f1_source": 0.6075,
                "notes": "Verified from telco_authentic_controls_dev_metrics_corrected.json on clean D_dev_val (N=292)",
            },
            "B2_B200_K200_ROS": {
                "f1_dev": 0.4497,
                "f1_source": 0.5890,
                "notes": "Primary non-synthetic Random OverSampling control arm",
            },
            "B4_B200_Target_Only": {
                "f1_dev": 0.7660,
                "f1_source": 0.3492,
                "notes": "Reconciled from 0.3089 transcription error; verified from telco_authentic_controls_source_retention.json",
            },
        },
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # -------------------------------------------------------------------------
    # STEP 3: Audit Frozen Candidate CSV Artifacts
    # -------------------------------------------------------------------------
    generators = ["TVAE", "CTGAN", "TabDDPM"]
    generator_results: dict[str, Any] = {}
    validation_summaries: dict[str, Any] = {}
    candidate_hashes: dict[str, str] = {}

    for gen_name in generators:
        csv_filename = f"batch05j_synthetic_candidates_{gen_name.lower()}.csv"
        csv_path = results_dir / csv_filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")

        file_hash = compute_file_sha256(csv_path)
        candidate_hashes[gen_name] = file_hash

        df_sample = pd.read_csv(csv_path)
        if len(df_sample) != 200 or df_sample.shape[1] != 20:
            raise ValueError(f"Candidate shape mismatch for {gen_name}: {df_sample.shape}")

        # Three-tier candidate validation
        val_summary = evaluate_candidate_tiers(
            df_synth=df_sample,
            df_train_b200=gen_input,
            meta=meta,
            eval_signatures=eval_signatures,
            continuous_cols=continuous_cols,
            discrete_cols=discrete_cols,
            clean_dev_val=clean_dev_val_post,
        )
        validation_summaries[gen_name] = val_summary

        generator_results[gen_name] = {
            "status": "AUDITED_FROZEN",
            "candidate_file": csv_filename,
            "csv_artifact": csv_filename,
            "candidate_file_sha256": file_hash,
            "candidates_requested": 200,
            "candidates_returned": len(df_sample),
            "tier1_structural_valid_count": val_summary["independent_pass_counts"]["tier1_structural_valid"],
            "tier2_physical_valid_count": val_summary["independent_pass_counts"]["tier2_physical_valid"],
            "tier2_unique_hard_valid_count": val_summary["sequential_funnel_counts"]["stage3_tier2_unique_hard_valid"],
            "tier3_plausible_unique_count": val_summary["sequential_funnel_counts"]["stage4_tier3_plausible_unique"],
            "class_distribution": val_summary["descriptive_feature_label_relationships"]["class_distribution"],
            "churn_rate_tenure_gt_29_pct": val_summary["descriptive_feature_label_relationships"]["affected_slice_tenure_gt_29"]["churn_rate_pct"],
            "churn_rate_tenure_le_29_pct": val_summary["descriptive_feature_label_relationships"]["preserved_slice_tenure_le_29"]["churn_rate_pct"],
            "conditional_difference_delta_pct": val_summary["descriptive_feature_label_relationships"]["conditional_difference_delta_pct"],
            "mean_ks_statistic": val_summary["tier3_empirical_plausibility"]["mean_ks_statistic"],
            "mean_tvd": val_summary["tier3_empirical_plausibility"]["mean_tvd"],
            "dcr_p05": val_summary["tier3_empirical_plausibility"]["dcr_p05"],
            "near_duplicates_under_0_01": val_summary["tier3_empirical_plausibility"]["near_duplicates_under_0_01"],
        }

    # Save audited protocol, summary, and validation JSONs
    with open(results_dir / "batch05j_generator_feasibility_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    with open(results_dir / "batch05j_generator_feasibility_summary.json", "w", encoding="utf-8") as f:
        json.dump(generator_results, f, indent=2)

    with open(results_dir / "batch05j_validation_tier_summary.json", "w", encoding="utf-8") as f:
        json.dump(validation_summaries, f, indent=2)

    return {
        "status": "SUCCESS",
        "results": generator_results,
        "protocol": protocol,
        "candidate_hashes": candidate_hashes,
    }


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=" * 70)
    print("SPRINT 06 — BATCH 5J-AUDIT: CANDIDATE & PROTOCOL AUDIT")
    print("=" * 70)
    out = audit_frozen_candidates()
    print("\nAudited Summary Results:")
    print(json.dumps(out["results"], indent=2))
    print(f"\nAudit completed in {time.perf_counter() - t_start:.2f}s.")
