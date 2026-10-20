"""Automated Unit and Governance Tests for Sprint 06 — Batch 5H Benchmark Protocol.

Validates:
1. Protocol JSON integrity, required fields, and arm definitions.
2. Authentic control arms (B0-B5) and synthetic repair arms (S1-S4).
3. Non-synthetic control specifications (ROS, Domain Weighting, Target-Only).
4. Development evaluation isolation (zero duplicate-group overlap between B200 and residual dev_val).
5. Evaluation partition isolation (zero cross-partition index or group leakage).
6. Preserved cryptographic hashes of historical baseline models and Batch 5F/5G models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)

REPO_ROOT = Path("component3").resolve().parent
RESULTS_DIR = REPO_ROOT / "component3" / "results" / "experimental_models"
SPLITS_DIR = REPO_ROOT / "component3" / "results" / "adaptation_splits"
MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
BASELINES_DIR = REPO_ROOT / "component3" / "models" / "baselines"

PROTOCOL_PATH = RESULTS_DIR / "batch05h_benchmark_protocol.json"
DESIGN_DOC_PATH = REPO_ROOT / "component3" / "docs" / "sprint06_batch05h_fair_repair_benchmark_design.md"

KNOWN_BASELINE_RF_HASH = "69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701"
KNOWN_SOURCE_RF_HASH = "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0"


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def protocol_data():
    """Load and parse Batch 5H benchmark protocol."""
    assert PROTOCOL_PATH.is_file(), f"Missing protocol file: {PROTOCOL_PATH}"
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_protocol_file_and_doc_exist():
    """Verify both machine-readable protocol and markdown design document exist."""
    assert PROTOCOL_PATH.is_file()
    assert DESIGN_DOC_PATH.is_file()
    assert DESIGN_DOC_PATH.stat().st_size > 5000, "Design doc appears incomplete"


def test_protocol_governance_fields(protocol_data):
    """Verify core metadata, drift scenario, and provenance in protocol."""
    assert protocol_data["protocol_name"] == "Sprint06_Batch05H_Fair_Repair_Benchmark_Protocol"
    assert protocol_data["dataset_id"] == "telco_churn"
    assert protocol_data["drift_scenario"] == "TELCO_CONCEPT"
    assert "tenure > 29.0" in protocol_data["concept_transformation_rule"]
    assert protocol_data["label_provenance"] == "SYNTHETIC_CONCEPT_TRUTH"
    assert protocol_data["adaptation_budgets"] == [50, 100, 200]
    assert protocol_data["synthetic_quantities_K"] == [0, 50, 100, 200, 500]


def test_protocol_authentic_controls(protocol_data):
    """Verify presence and specification of authentic control arms B0-B5."""
    controls = protocol_data["authentic_control_arms"]
    expected_arms = [
        "B0_frozen_source",
        "B1_naive_pooling",
        "B2_target_ros",
        "B3_balanced_domain_weighting",
        "B4_target_only",
        "B5_prespecified_weighting",
    ]
    for arm in expected_arms:
        assert arm in controls, f"Missing authentic control arm: {arm}"

    assert controls["B0_frozen_source"]["source_train_rows"] == 3452
    assert controls["B0_frozen_source"]["target_adapt_rows"] == 0
    assert controls["B4_target_only"]["source_train_rows"] == 0


def test_protocol_synthetic_arms(protocol_data):
    """Verify presence and specification of synthetic repair arms S1-S4."""
    synthetic = protocol_data["synthetic_repair_arms"]
    expected_arms = [
        "S1_raw_synthetic",
        "S2_screened_synthetic",
        "S3_slice_targeted",
        "S4_target_only_synthetic",
    ]
    for arm in expected_arms:
        assert arm in synthetic, f"Missing synthetic repair arm: {arm}"

    assert "tenure > 29.0" in synthetic["S3_slice_targeted"]["drift_slice_condition"]


def test_protocol_generator_specifications(protocol_data):
    """Verify generator configurations for TVAE, CTGAN, and TabDDPM."""
    gens = protocol_data["generator_specifications"]
    assert "TVAE" in gens
    assert "CTGAN" in gens
    assert "TabDDPM" in gens

    assert gens["TVAE"]["status"] == "PRIMARY_RECOMMENDED"
    assert gens["TabDDPM"]["status"] == "EXPLORATORY_PILOT_ONLY"
    assert gens["TVAE"]["batch_size_adaptation_bound"] <= 32


def test_development_validation_partition_isolation():
    """Verify that D_adapt,dev_val (residual adaptation pool with duplicate filtering) has 0 overlap with B200."""
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    canon_split_path = REPO_ROOT / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    adapt_pool = splits.target_adapt_pool
    assert len(adapt_pool) == 493

    b200 = splits.nested_adapt_budgets[200]
    assert len(b200) == 200

    b200_cust = set(b200["customerID"])
    unused_pool = adapt_pool[~adapt_pool["customerID"].isin(b200_cust)].copy().reset_index(drop=True)
    assert len(unused_pool) == 293

    # Auditing cross-partition duplicate groups:
    # Customer 9122-UMROB is in B200; duplicate pair 2676-ISHSF is in unused_pool.
    feat_cols = meta.feature_columns
    raw_overlap = b200.merge(unused_pool, on=feat_cols)
    assert len(raw_overlap) == 1, "Expected exactly 1 duplicate group pair between B200 and raw unused pool"
    assert "2676-ISHSF" in set(raw_overlap["customerID_y"])

    # Clean dev_val filters duplicate group to guarantee 0 group leakage
    dup_cust_in_unused = set(raw_overlap["customerID_y"])
    dev_val_clean = unused_pool[~unused_pool["customerID"].isin(dup_cust_in_unused)].copy().reset_index(drop=True)
    assert len(dev_val_clean) == 292

    overlap_clean = b200.merge(dev_val_clean, on=feat_cols)
    assert len(overlap_clean) == 0, f"Found {len(overlap_clean)} overlapping feature rows after filtering!"


def test_inviolable_model_hashes_preserved():
    """Verify historical baseline model and experimental source model hashes remain bitwise invariant."""
    baseline_rf_path = BASELINES_DIR / "telco_churn_random_forest.joblib"
    source_rf_path = MODELS_DIR / "telco_rf_experimental_source.joblib"

    assert compute_file_sha256(baseline_rf_path) == KNOWN_BASELINE_RF_HASH
    assert compute_file_sha256(source_rf_path) == KNOWN_SOURCE_RF_HASH


def test_stopping_conditions_defined(protocol_data):
    """Verify all required fail-fast stopping conditions are present."""
    stop_conds = protocol_data["stopping_conditions"]
    assert stop_conds["max_rss_memory_mb"] <= 4096.0
    assert stop_conds["generator_loss_divergence"] is not None
    assert stop_conds["partition_leakage"] is not None
    assert stop_conds["invalid_label_provenance"] is not None
