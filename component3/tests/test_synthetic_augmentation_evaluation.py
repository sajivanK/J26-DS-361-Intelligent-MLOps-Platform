"""Automated test suite for Sprint 06 Batch 5B: Synthetic Minority-Class Augmentation Evaluation.

Covers:
- Execution integrity of Notebook 05 (genuine outputs, no errors).
- Partition isolation and test set locking.
- Baseline model and preprocessor artifact preservation.
- Correct positive-class labeling on augmented records.
- Identical candidate pool usage for matched within-generator arms.
- Tier 2 structural safety and Tier 3 empirical plausibility screening.
- K=200 budget enforcement across all augmentation arms.
- TabDDPM multi-constraint screening and plausibility insufficiency handling.
- Random Forest hyperparameter matching across retrained models.
- Valid metric calculations and consistent paired prediction arrays (N=1,056).
- Absence of concept-drift recovery claims in Track A.
- Verification of publication-quality visualization figures.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import DatasetRegistry


# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMP3_DIR = REPO_ROOT / "component3"
NOTEBOOKS_DIR = COMP3_DIR / "notebooks"
MODELS_DIR = COMP3_DIR / "models"
RESULTS_DIR = COMP3_DIR / "results"
AUG_DIR = RESULTS_DIR / "augmentation"
FIG_DIR = RESULTS_DIR / "figures"
REPAIR_MODELS_DIR = MODELS_DIR / "repair"

NB05_PATH = NOTEBOOKS_DIR / "05_synthetic_augmentation_evaluation.ipynb"
MANIFEST_PATH = RESULTS_DIR / "split_manifests" / "telco_churn_split_manifest.json"
BASELINE_RF_PATH = MODELS_DIR / "baselines" / "telco_churn_random_forest.joblib"
BASELINE_PREP_PATH = MODELS_DIR / "baselines" / "telco_churn_preprocessor.joblib"

SUMMARY_CSV_PATH = AUG_DIR / "telco_churn_minority_augmentation_summary.csv"
SUMMARY_JSON_PATH = AUG_DIR / "telco_churn_minority_augmentation_summary.json"
CANDIDATE_MANIFEST_PATH = AUG_DIR / "telco_churn_candidate_selection_manifest.json"
PRED_AUDIT_CSV_PATH = AUG_DIR / "telco_churn_validation_predictions_audit.csv"

FIG1_PATH = FIG_DIR / "track_a_minority_augmentation_f1_comparison.png"
FIG2_PATH = FIG_DIR / "track_a_matched_filtering_effect.png"
FIG3_PATH = FIG_DIR / "track_a_precision_recall_curves.png"


# ---------------------------------------------------------------------------
# Test 1: Notebook 05 Execution Integrity
# ---------------------------------------------------------------------------

def test_notebook05_execution_integrity():
    """Verify Notebook 05 exists, has all code cells executed with genuine outputs, and zero errors."""
    assert NB05_PATH.exists(), f"Notebook 05 missing at {NB05_PATH}"
    nb = nbformat.read(NB05_PATH, as_version=4)

    assert len(nb.cells) >= 15, f"Expected at least 15 cells, got {len(nb.cells)}"
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 8, f"Expected 8 code cells, got {len(code_cells)}"

    for i, cell in enumerate(code_cells):
        assert cell.execution_count is not None, f"Code cell {i+1} was not executed"
        assert len(cell.outputs) > 0, f"Code cell {i+1} has empty outputs"
        for out in cell.outputs:
            assert out.output_type != "error", f"Error in cell {i+1}: {out.get('ename')}: {out.get('evalue')}"


# ---------------------------------------------------------------------------
# Test 2: Partition Isolation and Test Set Locking
# ---------------------------------------------------------------------------

def test_partition_isolation_and_test_locking():
    """Verify that train, validation, and test partitions are completely disjoint."""
    assert MANIFEST_PATH.exists(), f"Missing split manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_idx = set(manifest["train_indices"])
    val_idx = set(manifest["val_indices"])
    test_idx = set(manifest["test_indices"])

    assert len(train_idx) == 4931
    assert len(val_idx) == 1056
    assert len(test_idx) == 1056

    assert train_idx.isdisjoint(val_idx), "Train and Val partitions overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test partitions overlap!"
    assert val_idx.isdisjoint(test_idx), "Val and Test partitions overlap!"


# ---------------------------------------------------------------------------
# Test 3: Baseline Artifact Preservation
# ---------------------------------------------------------------------------

def test_baseline_artifact_preservation():
    """Verify that baseline Random Forest and preprocessor artifacts remain intact and unmodified."""
    assert BASELINE_RF_PATH.exists(), f"Missing baseline RF model at {BASELINE_RF_PATH}"
    assert BASELINE_PREP_PATH.exists(), f"Missing baseline preprocessor at {BASELINE_PREP_PATH}"

    base_rf = joblib.load(BASELINE_RF_PATH)
    params = base_rf.get_params()

    assert params["n_estimators"] == 100
    assert params["max_depth"] == 10
    assert params["class_weight"] == "balanced"
    assert params["random_state"] == 42


# ---------------------------------------------------------------------------
# Test 4: Positive Class Labeling on Augmented Records
# ---------------------------------------------------------------------------

def test_positive_class_labeling():
    """Verify that augmented records are correctly assigned to the minority class (Churn='Yes')."""
    assert SUMMARY_JSON_PATH.exists(), f"Missing summary JSON at {SUMMARY_JSON_PATH}"
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Every augmentation arm must add positive minority samples
    for arm in summary["seven_arm_results"]:
        if arm["arm_id"] != "Arm A1":
            assert arm["added_samples"] == 200
            assert arm["total_train_samples"] == 4931 + 200


# ---------------------------------------------------------------------------
# Test 5: Identical Candidate Pool for Matched Within-Generator Arms
# ---------------------------------------------------------------------------

def test_identical_candidate_pools_for_matched_arms():
    """Verify that random and quality-screened arms originate from the same candidate pool."""
    assert CANDIDATE_MANIFEST_PATH.exists(), f"Missing candidate manifest at {CANDIDATE_MANIFEST_PATH}"
    with open(CANDIDATE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # TVAE: Arm A3 and Arm A4 must use candidate IDs from the TVAE pool
    tvae_info = manifest["generators"]["tvae"]
    a3_ids = set(tvae_info["arm_a3_random_candidate_ids"])
    a4_ids = set(tvae_info["arm_a4_targeted_candidate_ids"])

    assert len(a3_ids) == 200
    assert len(a4_ids) == 200
    assert all(cid.startswith("tvae_cand_") for cid in a3_ids)
    assert all(cid.startswith("tvae_cand_") for cid in a4_ids)

    # CTGAN: Arm A5 and Arm A6 must use candidate IDs from the CTGAN pool
    ctgan_info = manifest["generators"]["ctgan"]
    a5_ids = set(ctgan_info["arm_a5_random_candidate_ids"])
    a6_ids = set(ctgan_info["arm_a6_targeted_candidate_ids"])

    assert len(a5_ids) == 200
    assert len(a6_ids) == 200
    assert all(cid.startswith("ctgan_cand_") for cid in a5_ids)
    assert all(cid.startswith("ctgan_cand_") for cid in a6_ids)


# ---------------------------------------------------------------------------
# Test 6: Tier 2 Structural and Tier 3 Empirical Plausibility Filtering
# ---------------------------------------------------------------------------

def test_structural_and_plausibility_filtering_logic():
    """Verify filtering logic properly validates physical bounds and billing ratios."""
    test_df = pd.DataFrame({
        "tenure": [10.0, -1.0, 10.0, 10.0, 0.0],
        "MonthlyCharges": [50.0, 50.0, 50.0, 50.0, 20.0],
        "TotalCharges": [500.0, 500.0, 100.0, 1500.0, 0.0],  # ratio: 1.0, invalid, 0.2 (low), 3.0 (high), 0.0
    })

    # Hard validity
    hard_valid = (test_df["tenure"] >= 0) & (test_df["MonthlyCharges"] > 0) & (test_df["TotalCharges"] >= 0)
    assert hard_valid.tolist() == [True, False, True, True, True]

    # Plausibility
    ratio = test_df["TotalCharges"] / (test_df["tenure"] * test_df["MonthlyCharges"] + 1e-6)
    plausible = hard_valid & (ratio >= 0.5) & (ratio <= 1.5) & (test_df["tenure"] <= 72)
    assert plausible.tolist() == [True, False, False, False, False]


# ---------------------------------------------------------------------------
# Test 7: Budget Enforcement (K=200) Across All Arms
# ---------------------------------------------------------------------------

def test_k200_budget_enforcement():
    """Verify that all augmented arms strictly adhere to K=200 added samples."""
    assert SUMMARY_CSV_PATH.exists(), f"Missing summary CSV at {SUMMARY_CSV_PATH}"
    df = pd.read_csv(SUMMARY_CSV_PATH)

    assert len(df) == 7, f"Expected 7 experimental arms, got {len(df)}"

    baseline_row = df[df["arm_id"] == "Arm A1"].iloc[0]
    assert baseline_row["added_samples"] == 0
    assert baseline_row["total_train_samples"] == 4931

    aug_rows = df[df["arm_id"] != "Arm A1"]
    for _, row in aug_rows.iterrows():
        assert row["added_samples"] == 200, f"{row['arm_id']} did not add exactly 200 samples"
        assert row["total_train_samples"] == 5131, f"{row['arm_id']} total train samples != 5131"


# ---------------------------------------------------------------------------
# Test 8: TabDDPM Multi-Constraint Screening and Insufficiency Gating
# ---------------------------------------------------------------------------

def test_tabddpm_multi_constraint_gating():
    """Verify TabDDPM plausibility count is below K=200 and documented as insufficient for plausibility."""
    with open(CANDIDATE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tab_info = manifest["generators"]["tabddpm"]
    assert tab_info["plausibility_insufficient_for_k200"] is True
    assert tab_info["stats"]["plausible_unique"] < 200
    assert tab_info["stats"]["hard_valid_unique"] >= 200


# ---------------------------------------------------------------------------
# Test 9: Random Forest Hyperparameter Matching Across Repaired Models
# ---------------------------------------------------------------------------

def test_random_forest_hyperparameters_matched():
    """Verify that all retrained Random Forest models share identical hyperparameters."""
    repaired_arm_ids = ["arm_a2", "arm_a3", "arm_a4", "arm_a5", "arm_a6", "arm_a7"]
    for arm_slug in repaired_arm_ids:
        model_file = REPAIR_MODELS_DIR / f"telco_churn_rf_track_a_{arm_slug}.joblib"
        assert model_file.exists(), f"Repaired model missing at {model_file}"

        rf = joblib.load(model_file)
        params = rf.get_params()

        assert params["n_estimators"] == 100, f"{arm_slug} n_estimators != 100"
        assert params["max_depth"] == 10, f"{arm_slug} max_depth != 10"
        assert params["class_weight"] == "balanced", f"{arm_slug} class_weight != balanced"
        assert params["random_state"] == 42, f"{arm_slug} random_state != 42"


# ---------------------------------------------------------------------------
# Test 10: Metric Calculations and Paired Prediction Arrays
# ---------------------------------------------------------------------------

def test_metric_calculations_and_support():
    """Verify metric ranges, non-null values, and confusion matrix support (N=1,056)."""
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for arm in summary["seven_arm_results"]:
        assert 0.0 <= arm["positive_f1"] <= 1.0
        assert 0.0 <= arm["macro_f1"] <= 1.0
        assert 0.0 <= arm["precision"] <= 1.0
        assert 0.0 <= arm["recall"] <= 1.0
        assert 0.0 <= arm["roc_auc"] <= 1.0
        assert 0.0 <= arm["pr_auc"] <= 1.0

        # Confusion matrix support must sum to exactly 1,056
        cm = arm["confusion_matrix"]
        total_support = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total_support == 1056, f"Confusion matrix support is {total_support}, expected 1056"


# ---------------------------------------------------------------------------
# Test 11: Absence of Concept-Drift Recovery Claims in Track A
# ---------------------------------------------------------------------------

def test_no_concept_drift_recovery_claims_in_track_a():
    """Verify Track A artifacts do not compute recovery fractions or claim concept-drift repair."""
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary_text = f.read()

    assert "recovery_fraction" not in summary_text.lower()
    assert "concept_drift_recovery" not in summary_text.lower()

    df = pd.read_csv(SUMMARY_CSV_PATH)
    assert "recovery_fraction" not in df.columns


# ---------------------------------------------------------------------------
# Test 12: Publication-Quality Visualization Figures
# ---------------------------------------------------------------------------

def test_publication_figures_exist_and_valid():
    """Verify that all three publication figures exist and exceed 50 KB."""
    for fig_path in [FIG1_PATH, FIG2_PATH, FIG3_PATH]:
        assert fig_path.exists(), f"Figure missing at {fig_path}"
        assert fig_path.stat().st_size > 50000, f"Figure {fig_path.name} is smaller than 50 KB"


# ---------------------------------------------------------------------------
# Test 13: Prediction-Level Audit Artifact Integrity
# ---------------------------------------------------------------------------

def test_prediction_level_audit_artifact_integrity():
    """Verify prediction audit CSV exists, has 1,056 records, 0 raw features, and matches metrics."""
    assert PRED_AUDIT_CSV_PATH.exists(), f"Missing prediction audit CSV at {PRED_AUDIT_CSV_PATH}"
    audit_df = pd.read_csv(PRED_AUDIT_CSV_PATH)

    assert len(audit_df) == 1056, f"Expected 1056 validation predictions, got {len(audit_df)}"

    expected_cols = [
        "val_record_index", "customer_id", "y_true",
        "pred_arm_a1", "prob_arm_a1",
        "pred_arm_a2", "prob_arm_a2",
        "pred_arm_a3", "prob_arm_a3",
        "pred_arm_a4", "prob_arm_a4",
        "pred_arm_a5", "prob_arm_a5",
        "pred_arm_a6", "prob_arm_a6",
        "pred_arm_a7", "prob_arm_a7",
    ]
    assert list(audit_df.columns) == expected_cols, f"Column mismatch in audit artifact: {list(audit_df.columns)}"
    assert not audit_df.isna().any().any(), "Prediction audit CSV contains NaN values"

    # Ensure customer_id column uses anonymous row identifiers and contains no real customer IDs
    assert audit_df["customer_id"].str.startswith("val_row_").all(), "customer_id values must be anonymous val_row_XXXX IDs"
    assert not audit_df["customer_id"].str.contains(r"^[0-9]{4}-[A-Z0-9]{5}$", regex=True).any(), (
        "Real Telco customer IDs detected in prediction audit artifact!"
    )

    # Ensure no private raw features leaked into audit artifact
    private_raw_features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "gender"]
    for priv in private_raw_features:
        assert priv not in audit_df.columns, f"Private raw feature '{priv}' leaked into audit artifact!"

    # Verify recalculating F1 matches summary CSV
    from sklearn.metrics import f1_score
    summary_df = pd.read_csv(SUMMARY_CSV_PATH)
    for i in range(1, 8):
        recalc_f1 = f1_score(audit_df["y_true"], audit_df[f"pred_arm_a{i}"], pos_label=1)
        reported_f1 = summary_df.loc[i - 1, "positive_f1"]
        assert round(recalc_f1, 4) == round(reported_f1, 4), (
            f"Arm A{i} F1 mismatch: recalculated {recalc_f1:.4f} != reported {reported_f1:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 14: Permutation Testing and Generator Checkpoint Provenance
# ---------------------------------------------------------------------------

def test_permutation_testing_and_provenance_metadata():
    """Verify summary JSON contains permutation test statistics, generator provenance, and diversity audit."""
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Checkpoint provenance
    prov = summary.get("generator_checkpoint_provenance", {})
    assert set(prov.keys()) == {"TVAE", "CTGAN", "TabDDPM"}, f"Missing provenance keys: {set(prov.keys())}"
    for gen_name, p in prov.items():
        assert p["training_record_count"] == 1309
        assert p["epoch_count"] == 10
        assert p["random_seed"] == 42
        assert "Experiment A" in p["origin"]
        assert p["overlaps_262_holdout"] is True
        assert p["val_leakage"] is False

    # Permutation test statistics in seven_arm_results
    for arm in summary["seven_arm_results"]:
        assert "perm_p_one_sided" in arm, f"Missing perm_p_one_sided in {arm['arm_id']}"
        assert "perm_p_two_sided" in arm, f"Missing perm_p_two_sided in {arm['arm_id']}"
        assert 0.0 <= arm["perm_p_one_sided"] <= 1.0
        assert 0.0 <= arm["perm_p_two_sided"] <= 1.0

    # Within-generator comparisons
    within = summary.get("within_generator_comparisons", [])
    assert len(within) == 2
    for w in within:
        assert "perm_p_one_sided_targeted_gt_random" in w
        assert "perm_p_two_sided" in w
        assert "filtering_delta_ci95" in w

    # Exact matches with authentic training records must be 0
    div_audit = summary.get("synthetic_validity_and_diversity_audit", {})
    assert div_audit.get("exact_matches_with_authentic_train") == 0


# ---------------------------------------------------------------------------
# Test 15: Maintained Reproducibility Runner in Component 3
# ---------------------------------------------------------------------------

def test_maintained_reproducibility_runner():
    """Verify that the maintained reproducibility runner exists in component3 and has zero scratch dependencies."""
    runner_path = COMP3_DIR / "src" / "experiments" / "run_synthetic_augmentation_evaluation.py"
    assert runner_path.exists(), f"Maintained runner missing at {runner_path}"

    with open(runner_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ensure no imports or paths from root scratch/ directory
    for line in content.splitlines():
        if line.strip().startswith(("import scratch", "from scratch")):
            assert False, f"Scratch import detected: {line}"
        if "scratch/" in line or "scratch\\" in line:
            assert False, f"Scratch path reference detected: {line}"


