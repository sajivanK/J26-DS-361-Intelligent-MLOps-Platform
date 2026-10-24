"""Unit tests for Sprint 06 Batch 5J / Batch 5J-Audit: Post-Drift Synthetic Generator Feasibility & Audit."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)
from component3.src.experiments.run_generator_feasibility import (
    evaluate_candidate_tiers,
    compute_tvd,
    compute_file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPONENT3_ROOT = REPO_ROOT / "component3"
RESULTS_DIR = COMPONENT3_ROOT / "results" / "generators"


def test_input_b200_isolation_and_provenance():
    """Verify B200 input provenance, post-drift label transformation, and zero partition leakage."""
    canon_manifest_path = COMPONENT3_ROOT / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    assert canon_manifest_path.is_file(), "Canonical split manifest must exist"

    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    with open(canon_manifest_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    b200 = splits.nested_adapt_budgets[200].copy().reset_index(drop=True)
    assert len(b200) == 200, "B200 must contain exactly 200 adaptation records"

    # Verify post-drift TELCO_CONCEPT inversion: all 97 rows with tenure > 29.0 inverted
    flips = b200["tenure"] > 29.0
    assert flips.sum() == 97, "Exactly 97 records must have tenure > 29.0"

    raw_gt29_no = int((b200.loc[flips, "Churn"] == "No").sum())
    raw_gt29_yes = int((b200.loc[flips, "Churn"] == "Yes").sum())
    assert raw_gt29_no == 84, "Exactly 84 records in tenure > 29 were No (flipped to Yes)"
    assert raw_gt29_yes == 13, "Exactly 13 records in tenure > 29 were Yes (flipped to No)"

    b200_post = b200.copy()
    b200_post.loc[flips, "Churn"] = b200_post.loc[flips, "Churn"].map({"No": "Yes", "Yes": "No"})

    post_gt29_yes = int((b200_post.loc[flips, "Churn"] == "Yes").sum())
    post_gt29_no = int((b200_post.loc[flips, "Churn"] == "No").sum())
    assert post_gt29_yes == 84
    assert post_gt29_no == 13

    class_dist = b200_post["Churn"].value_counts().to_dict()
    assert class_dist == {"Yes": 124, "No": 76}, "Post-drift class distribution must be exactly 124 Yes, 76 No"

    # Verify zero feature signature overlap with evaluation partitions
    feat_cols = meta.feature_columns
    b200_sigs = set(tuple(str(v) for v in r) for r in b200[feat_cols].itertuples(index=False))

    src_train_sigs = set(tuple(str(v) for v in r) for r in splits.source_train[feat_cols].itertuples(index=False))
    src_eval_sigs = set(tuple(str(v) for v in r) for r in splits.source_eval[feat_cols].itertuples(index=False))
    tgt_eval_sigs = set(tuple(str(v) for v in r) for r in splits.target_eval[feat_cols].itertuples(index=False))

    assert len(b200_sigs.intersection(src_train_sigs)) == 0, "B200 overlaps with source_train"
    assert len(b200_sigs.intersection(src_eval_sigs)) == 0, "B200 overlaps with source_eval"
    assert len(b200_sigs.intersection(tgt_eval_sigs)) == 0, "B200 overlaps with target_eval"


def test_batch05j_artifacts_exist_and_conform():
    """Verify Batch 5J protocol, summary, tier metrics, and candidate CSV artifacts."""
    protocol_path = RESULTS_DIR / "batch05j_generator_feasibility_protocol.json"
    summary_path = RESULTS_DIR / "batch05j_generator_feasibility_summary.json"
    tier_path = RESULTS_DIR / "batch05j_validation_tier_summary.json"

    assert protocol_path.is_file(), "Protocol JSON must exist"
    assert summary_path.is_file(), "Summary JSON must exist"
    assert tier_path.is_file(), "Tier validation JSON must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for gen in ["TVAE", "CTGAN", "TabDDPM"]:
        assert gen in summary, f"{gen} must be in summary"
        gen_data = summary[gen]
        assert gen_data["status"] in ["COMPLETED", "AUDITED_FROZEN"]
        assert gen_data["candidates_requested"] == 200
        assert gen_data["candidates_returned"] == 200
        assert gen_data["tier1_structural_valid_count"] == 200

        csv_path = RESULTS_DIR / gen_data["candidate_file"]
        assert csv_path.is_file(), f"Candidate CSV for {gen} must exist"
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 200, f"CSV for {gen} must have 200 rows"
        assert df_csv.shape[1] == 20, f"CSV for {gen} must have 20 columns"


def test_reconciled_benchmark_reference_values():
    """Verify authoritative frozen benchmark numbers from Batch 5I artifacts."""
    exp_dir = COMPONENT3_ROOT / "results" / "experimental_models"
    with open(exp_dir / "telco_authentic_controls_dev_metrics_corrected.json", "r") as f:
        dev_metrics = json.load(f)
    with open(exp_dir / "telco_authentic_controls_source_retention.json", "r") as f:
        ret_metrics = json.load(f)

    # B0 reference: 0.4055 on D_dev_val (N=292), 0.6075 on D_source_eval (N=493)
    assert np.isclose(dev_metrics["B0"]["overall"]["positive_f1"], 0.4055)
    assert np.isclose(ret_metrics["B0"]["source_f1_positive"], 0.6075)

    # B2_B200_K200 reference: 0.4497 on D_dev_val, 0.5890 on D_source_eval
    assert np.isclose(dev_metrics["B2_B200_K200"]["overall"]["positive_f1"], 0.4497)
    assert np.isclose(ret_metrics["B2_B200_K200"]["source_f1_positive"], 0.5890)

    # B4_B200 reference: 0.7660 on D_dev_val, 0.3492 on D_source_eval (reconciled from 0.3089 transcription error)
    assert np.isclose(dev_metrics["B4_B200"]["overall"]["positive_f1"], 0.7660)
    assert np.isclose(ret_metrics["B4_B200"]["source_f1_positive"], 0.3492)


def test_candidate_filtering_funnel_and_independent_counts():
    """Verify sequential funnel vs independent candidate filter counts on frozen CSVs."""
    summary_path = RESULTS_DIR / "batch05j_generator_feasibility_summary.json"
    tier_path = RESULTS_DIR / "batch05j_validation_tier_summary.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(tier_path, "r", encoding="utf-8") as f:
        tiers = json.load(f)

    # Sequential Plausible Unique Survivors (Stage 4)
    assert summary["TVAE"]["tier3_plausible_unique_count"] == 65
    assert summary["CTGAN"]["tier3_plausible_unique_count"] == 30
    assert summary["TabDDPM"]["tier3_plausible_unique_count"] == 42

    # Independent billing ratio pass counts (across all 200 candidates)
    assert tiers["TVAE"]["independent_pass_counts"]["tier3_billing_ratio_valid"] == 70
    assert tiers["CTGAN"]["independent_pass_counts"]["tier3_billing_ratio_valid"] == 32
    assert tiers["TabDDPM"]["independent_pass_counts"]["tier3_billing_ratio_valid"] == 49


def test_descriptive_conditional_relationships_audit():
    """Verify descriptive conditional churn rate calculations without deterministic rule gating."""
    summary_path = RESULTS_DIR / "batch05j_generator_feasibility_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # TabDDPM: accurately captured concept shift
    assert np.isclose(summary["TabDDPM"]["churn_rate_tenure_gt_29_pct"], 86.46, atol=0.05)
    assert np.isclose(summary["TabDDPM"]["churn_rate_tenure_le_29_pct"], 41.35, atol=0.05)
    assert np.isclose(summary["TabDDPM"]["conditional_difference_delta_pct"], 45.11, atol=0.05)

    # CTGAN: under-captured concept shift
    assert np.isclose(summary["CTGAN"]["churn_rate_tenure_gt_29_pct"], 62.50, atol=0.05)
    assert np.isclose(summary["CTGAN"]["churn_rate_tenure_le_29_pct"], 59.87, atol=0.05)
    assert np.isclose(summary["CTGAN"]["conditional_difference_delta_pct"], 2.63, atol=0.05)

    # TVAE: majority class bias
    assert np.isclose(summary["TVAE"]["churn_rate_tenure_gt_29_pct"], 100.0, atol=0.05)
    assert np.isclose(summary["TVAE"]["churn_rate_tenure_le_29_pct"], 69.87, atol=0.05)
    assert np.isclose(summary["TVAE"]["conditional_difference_delta_pct"], 30.13, atol=0.05)


def test_three_tier_validation_logic_synthetic():
    """Unit test for candidate tier evaluation logic using synthetic fixtures."""
    meta = get_dataset_registry().get("telco_churn")
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    discrete_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "Churn"
    ]

    # Create dummy training data
    train_data = {
        "tenure": [10.0, 40.0],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": [500.0, 2800.0],
        "Churn": ["No", "Yes"],
    }
    for c in discrete_cols[:-1]:
        train_data[c] = ["A", "B"]
    df_train = pd.DataFrame(train_data)

    # Candidate 1: completely valid
    c1 = df_train.iloc[0:1].copy()
    c1["tenure"] = 12.0
    c1["TotalCharges"] = 600.0  # ratio = 600 / (12*50) = 1.0 (valid)

    # Candidate 2: negative tenure (violates Tier 2)
    c2 = c1.copy()
    c2["tenure"] = -5.0

    # Candidate 3: invalid billing ratio (violates Tier 3)
    c3 = c1.copy()
    c3["tenure"] = 10.0
    c3["TotalCharges"] = 5000.0  # ratio = 5000 / (10*50) = 10.0 (invalid)

    df_test = pd.concat([c1, c2, c3], ignore_index=True)

    result = evaluate_candidate_tiers(
        df_synth=df_test,
        df_train_b200=df_train,
        meta=meta,
        eval_signatures={},
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
    )

    assert result["tier1_structural"]["structural_valid_count"] == 3
    assert result["sequential_funnel_counts"]["stage2_tier2_hard_physical_pass"] == 2  # c2 rejected
    assert result["sequential_funnel_counts"]["stage4_tier3_plausible_unique"] == 1  # only c1 passes all


def test_tvd_computation():
    """Verify categorical TVD metric calculation."""
    p = {"A": 60, "B": 40}
    q = {"A": 50, "B": 50}
    tvd = compute_tvd(p, q, total_p=100, total_q=100)
    assert pytest.approx(tvd, 0.001) == 0.10
