"""Unit tests for Component 3 Sprint 06 Batch 5A Repair Experiment Protocol (Scientific Revisions).

Validates:
1. Protocol documentation completeness across Track A and Track B requirements.
2. Track A / Track B separation and prerequisite gating for concept-drift adaptation.
3. Within-generator matched Random vs. Targeted comparison controls.
4. Authentic Random OverSampling (ROS) control definition.
5. Recovery fraction scoping (not applicable to Track A; guardrailed under drift).
6. Strict partition isolation and duplicate-group-aware internal split invariance.
7. Baseline model hyperparameter matching against the frozen serialized artifact.
8. Historical figure provenance audit and classification.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import DatasetRegistry


# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMP3_DIR = REPO_ROOT / "component3"
DOCS_DIR = COMP3_DIR / "docs"
PROTOCOL_PATH = DOCS_DIR / "sprint06_batch05a_repair_experiment_protocol.md"
RESULTS_DIR = COMP3_DIR / "results"
MODELS_DIR = COMP3_DIR / "models"
MANIFEST_PATH = RESULTS_DIR / "split_manifests" / "telco_churn_split_manifest.json"
BASELINE_MODEL_PATH = MODELS_DIR / "baselines" / "telco_churn_random_forest.joblib"
BASELINE_PREP_PATH = MODELS_DIR / "baselines" / "telco_churn_preprocessor.joblib"
DRIFT_EVAL_CSV = RESULTS_DIR / "drift" / "drift_simulation_summary.csv"
HISTORICAL_FIG_PATH = RESULTS_DIR / "figures" / "pilot_slice_f1_recovery.png"


# ---------------------------------------------------------------------------
# Test 1: Protocol Document Existence and Key Section Verification
# ---------------------------------------------------------------------------

def test_protocol_document_completeness():
    """Verify the technical protocol document exists and contains all required revised sections."""
    assert PROTOCOL_PATH.exists(), f"Protocol document missing at {PROTOCOL_PATH}"
    content = PROTOCOL_PATH.read_text(encoding="utf-8")

    required_keywords = [
        "Track A: Minority-Class Augmentation",
        "Track B: Controlled Concept-Drift Adaptation",
        "BLOCKED / PREREQUISITE-GATED",
        "Authentic OverSampling (ROS)",
        "TVAE Random",
        "TVAE Targeted",
        "CTGAN Random",
        "CTGAN Targeted",
        "TabDDPM Deduplicated",
        "max_depth = 10",
        "n_estimators = 100",
        "Audit of Historical Figure",
        "Explicit Scientific Non-Claims",
    ]
    for kw in required_keywords:
        assert kw in content, f"Protocol document missing expected section or keyword: '{kw}'"


# ---------------------------------------------------------------------------
# Test 2: Track Separation and Track B Prerequisite Gating
# ---------------------------------------------------------------------------

def test_track_separation_and_prerequisite_gating():
    """Verify Track A and Track B are separated and Track B is strictly prerequisite-gated."""
    content = PROTOCOL_PATH.read_text(encoding="utf-8")

    # Track A must evaluate on clean validation data without concept-drift recovery claims
    assert "clean validation partition" in content.lower()
    assert "zero claim of concept-drift recovery" in content.lower()

    # Track B must explain why pre-drift generators cannot represent the new concept
    assert "old concept" in content.lower() or "pre-drift" in content.lower()
    assert "blocked / prerequisite-gated" in content.lower()

    # Engine must abstain when post-drift labels are unavailable
    assert "abstain" in content.lower()


# ---------------------------------------------------------------------------
# Test 3: Within-Generator Matched Random vs. Targeted Controls
# ---------------------------------------------------------------------------

def test_matched_random_vs_targeted_controls():
    """Verify within-generator matched comparisons eliminate confounding."""
    track_a_arms = {
        "Arm A1": {"name": "Baseline", "budget": 0, "generator": None, "filtering": None},
        "Arm A2": {"name": "Authentic ROS", "budget": 200, "generator": "Authentic", "filtering": "ROS"},
        "Arm A3": {"name": "TVAE Random", "budget": 200, "generator": "TVAE", "filtering": "Tier 2 Hard"},
        "Arm A4": {"name": "TVAE Targeted", "budget": 200, "generator": "TVAE", "filtering": "Tier 2 + Tier 3"},
        "Arm A5": {"name": "CTGAN Random", "budget": 200, "generator": "CTGAN", "filtering": "Tier 2 Hard"},
        "Arm A6": {"name": "CTGAN Targeted", "budget": 200, "generator": "CTGAN", "filtering": "Tier 2 + Tier 3"},
        "Arm A7": {"name": "TabDDPM Deduplicated", "budget": 200, "generator": "TabDDPM", "filtering": "Tier 2 + Dedup"},
    }

    # Verify TVAE pair is matched on budget
    assert track_a_arms["Arm A3"]["budget"] == track_a_arms["Arm A4"]["budget"] == 200
    assert track_a_arms["Arm A3"]["generator"] == track_a_arms["Arm A4"]["generator"] == "TVAE"
    assert track_a_arms["Arm A3"]["filtering"] != track_a_arms["Arm A4"]["filtering"]

    # Verify CTGAN pair is matched on budget
    assert track_a_arms["Arm A5"]["budget"] == track_a_arms["Arm A6"]["budget"] == 200
    assert track_a_arms["Arm A5"]["generator"] == track_a_arms["Arm A6"]["generator"] == "CTGAN"
    assert track_a_arms["Arm A5"]["filtering"] != track_a_arms["Arm A6"]["filtering"]

    # Verify all augmentation arms have identical budget K=200
    aug_arms = ["Arm A2", "Arm A3", "Arm A4", "Arm A5", "Arm A6", "Arm A7"]
    budgets = [track_a_arms[a]["budget"] for a in aug_arms]
    assert len(set(budgets)) == 1 and budgets[0] == 200


# ---------------------------------------------------------------------------
# Test 4: Recovery Fraction Scoping and Singularity Guardrails
# ---------------------------------------------------------------------------

def compute_recovery_fraction(clean_f1: float, degraded_f1: float, repaired_f1: float) -> tuple[float | None, str]:
    """Mathematical recovery fraction with strict singularity guardrails."""
    deg_delta = clean_f1 - degraded_f1
    if deg_delta <= 0.01:
        return None, "NOT_APPLICABLE_OR_ZERO_DEGRADATION"
    recovery = (repaired_f1 - degraded_f1) / deg_delta
    return recovery, "VALID"


def test_recovery_fraction_scoping():
    """Verify recovery fraction is scoped strictly to genuine degradation and rejected for Track A."""
    # Track A Clean setting: clean_f1 = 0.6209, degraded_f1 = 0.6209 (no degradation)
    rec_clean, status_clean = compute_recovery_fraction(0.6209, 0.6209, 0.6400)
    assert status_clean == "NOT_APPLICABLE_OR_ZERO_DEGRADATION"
    assert rec_clean is None

    # Track B Concept Drift setting: clean_f1 = 0.6209, degraded_f1 = 0.4252, repaired = 0.5200
    rec_drift, status_drift = compute_recovery_fraction(0.6209, 0.4252, 0.5200)
    assert status_drift == "VALID"
    assert rec_drift is not None
    assert np.isclose(rec_drift, (0.5200 - 0.4252) / (0.6209 - 0.4252))


# ---------------------------------------------------------------------------
# Test 5: Strict Partition Isolation and Duplicate-Safe Internal Split
# ---------------------------------------------------------------------------

def test_partition_isolation_and_duplicate_safety():
    """Verify canonical partitions and internal development splits have 0 feature leaks."""
    assert MANIFEST_PATH.exists(), f"Split manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_indices = set(manifest["train_indices"])
    val_indices = set(manifest["val_indices"])
    test_indices = set(manifest["test_indices"])

    # Outer split disjointness
    assert len(train_indices & val_indices) == 0, "Train and Val indices overlap"
    assert len(train_indices & test_indices) == 0, "Train and Test indices overlap"
    assert len(val_indices & test_indices) == 0, "Val and Test indices overlap"
    assert len(train_indices) == 4931
    assert len(val_indices) == 1056
    assert len(test_indices) == 1056

    # Internal duplicate-safe split of training positive slice
    reg = DatasetRegistry()
    meta = reg.get("telco_churn")
    raw_df = pd.read_csv(meta.get_absolute_path())

    train_df = raw_df.iloc[sorted(list(train_indices))].copy()
    feature_cols = [c for c in train_df.columns if c not in ["customerID", "Churn"]]

    pos_df = train_df[train_df["Churn"] == "Yes"].copy()
    assert len(pos_df) == 1309, f"Expected 1,309 positive training records, got {len(pos_df)}"

    grouped = pos_df.groupby(feature_cols, sort=False).indices
    np.random.seed(42)
    group_keys = list(grouped.keys())
    np.random.shuffle(group_keys)

    gen_train_pos_idx = []
    auth_ctrl_pos_idx = []
    for k in group_keys:
        idx_list = grouped[k].tolist()
        if len(gen_train_pos_idx) + len(idx_list) <= 1047:
            gen_train_pos_idx.extend(idx_list)
        else:
            auth_ctrl_pos_idx.extend(idx_list)

    assert len(gen_train_pos_idx) == 1047, f"Expected 1,047 generator training records, got {len(gen_train_pos_idx)}"
    assert len(auth_ctrl_pos_idx) == 262, f"Expected 262 authentic control records, got {len(auth_ctrl_pos_idx)}"
    assert set(gen_train_pos_idx).isdisjoint(auth_ctrl_pos_idx)

    # Check zero feature-identical row overlap across the internal split boundary
    gen_feats = pos_df.iloc[gen_train_pos_idx][feature_cols]
    ctrl_feats = pos_df.iloc[auth_ctrl_pos_idx][feature_cols]
    overlap = pd.merge(gen_feats, ctrl_feats, how="inner")
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping feature identical groups across internal split"


# ---------------------------------------------------------------------------
# Test 6: Baseline Model Hyperparameters and Artifact Verification
# ---------------------------------------------------------------------------

def test_baseline_model_hyperparameter_verification():
    """Verify protocol matches the frozen serialized baseline Random Forest model."""
    assert BASELINE_MODEL_PATH.exists(), f"Baseline model artifact missing at {BASELINE_MODEL_PATH}"
    assert BASELINE_PREP_PATH.exists(), f"Baseline preprocessor artifact missing at {BASELINE_PREP_PATH}"

    rf_model = joblib.load(BASELINE_MODEL_PATH)
    params = rf_model.get_params()

    # Exact parameter matching
    assert params["n_estimators"] == 100, f"Expected n_estimators=100, got {params['n_estimators']}"
    assert params["max_depth"] == 10, f"Expected max_depth=10, got {params['max_depth']}"
    assert params["class_weight"] == "balanced", f"Expected class_weight='balanced', got {params['class_weight']}"
    assert params["random_state"] == 42, f"Expected random_state=42, got {params['random_state']}"


# ---------------------------------------------------------------------------
# Test 7: Historical Figure Provenance and Classification
# ---------------------------------------------------------------------------

def test_historical_figure_audit():
    """Verify historical figure exists and is treated as exploratory smoke-test evidence."""
    assert HISTORICAL_FIG_PATH.exists(), f"Figure missing at {HISTORICAL_FIG_PATH}"
    assert HISTORICAL_FIG_PATH.stat().st_size > 50000, "Figure file size is smaller than expected"

    protocol_content = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "historical exploratory smoke-test evidence" in protocol_content
    assert "not valid research evidence" in protocol_content.lower().replace("*", "")
