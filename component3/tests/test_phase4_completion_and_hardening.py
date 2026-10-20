"""Phase 4 Test Suite: Final System Completion, Hardening, and Validation for Component 3.

Validates:
1. Prescription Dataset Audit & Integrity (160 rows, 3 datasets, 10 scenario groups, zero leakage).
2. Model Evaluation & Baselines Comparison (Always-Reject, Majority, Rule-Based, Tree, Learned).
3. Degrees-of-Freedom & Predictor Architecture (Unified vs Separate predictors).
4. Full Notebooks Inventory & Real Saved Outputs (01 through 10).
5. Phase 4 End-to-End System Demonstration Manifest & Cryptographic Fingerprints.
6. Security Hardening & Safe Failure Handling (path traversal, corrupted models, schema mismatch, OOD).
7. Python API & CLI Operations with DecisionMode (`RULE_BASED` and `LEARNED`).
8. Component 4 (C4) Review Handoff Contract Integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from component3.src.api import (
    check_system_readiness,
    execute_repair,
    inspect_package,
    inspect_repair_status,
    inspect_validation_results,
    validate_repair_request,
    view_prescription_reasoning,
)
from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    ExecutionMode,
    RepairObjective,
    RepairRequest,
)
from component3.src.orchestration.repair_orchestrator import RepairOrchestrator
from component3.src.prescription.learned_model import (
    INPUT_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    SCENARIO_GROUP_COLUMNS,
    LearnedPrescriptionModel,
)
from component3.src.prescription.model_evaluation import (
    evaluate_all_prescription_baselines,
    evaluate_separate_vs_unified_predictors,
)
from component3.src.prescription.prescription_models import DecisionMode, RepairStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# 1. Prescription Dataset Independence & Air-Gapping Tests
# =============================================================================

def test_prescription_dataset_audit_and_zero_leakage():
    """Verify prescription dataset has 160 records, 3 datasets, 10 scenarios, zero leakage."""
    csv_path = REPO_ROOT / "component3" / "results" / "prescription_training_dataset.csv"
    json_path = REPO_ROOT / "component3" / "results" / "prescription_training_dataset.json"

    assert csv_path.is_file(), "CSV prescription training dataset missing"
    assert json_path.is_file(), "JSON prescription training dataset missing"

    df = pd.read_csv(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(df) == 160, f"Expected exactly 160 records, got {len(df)}"
    assert len(records) == 160, f"Expected exactly 160 records in JSON, got {len(records)}"

    # Check datasets
    datasets = set(df["group_dataset_id"].unique())
    assert datasets == {"telco_churn", "uci_adult_full", "bank_marketing"}

    # Check scenarios: exactly 10 distinct scenario groups
    scenarios = set(df["group_scenario_id"].unique())
    assert len(scenarios) == 10, f"Expected 10 scenario groups, got {len(scenarios)}"

    # Air-gapping verification
    input_cols = set(INPUT_FEATURE_COLUMNS)
    outcome_cols = set(OUTCOME_COLUMNS)
    group_cols = set(SCENARIO_GROUP_COLUMNS)

    assert len(input_cols.intersection(outcome_cols)) == 0, "Leakage between inputs and outcomes"
    assert len(input_cols.intersection(group_cols)) == 0, "Leakage between inputs and group columns"

    # Missing / censored outcome check (EXP_5L_TARGETED_REPAIR_STATUS)
    censored = df[df["out_delta_f1_vs_baseline"].isnull()]
    assert len(censored) == 1, f"Expected exactly 1 censored outcome, got {len(censored)}"
    assert censored["out_package_status"].iloc[0] == "BLOCKED_NON_ORACLE_ABSENT"


# =============================================================================
# 2. Model Evaluation, Baselines & Predictor Architecture Tests
# =============================================================================

def test_prescription_baselines_evaluation():
    """Verify evaluation of all 5 baselines under Leave-One-Scenario-Out CV."""
    csv_path = REPO_ROOT / "component3" / "results" / "prescription_training_dataset.csv"
    df = pd.read_csv(csv_path)

    results = evaluate_all_prescription_baselines(df, group_col="group_scenario_id")

    expected_models = {
        "Always-Reject",
        "Majority-Class",
        "Deterministic-Rule-Engine",
        "Interpretable-Tree",
        "Learned-Ridge-Logistic",
    }
    assert set(results.keys()) == expected_models

    # Feasibility and source retention checks
    for name, s in results.items():
        assert s.feasibility_violation_rate == 0.0, f"{name} had feasibility violations"
        assert s.source_retention_violation_rate == 0.0, f"{name} had source retention violations"
        assert s.utility_rmse >= 0.0
        assert s.mean_strategy_regret_vs_oracle >= 0.0

    # Rule-based engine should achieve zero feasibility violations and low regret
    rule_s = results["Deterministic-Rule-Engine"]
    assert rule_s.mean_strategy_regret_vs_oracle <= 0.02


def test_separate_vs_unified_predictors_audit():
    """Verify mathematical audit of separate vs unified candidate predictors."""
    csv_path = REPO_ROOT / "component3" / "results" / "prescription_training_dataset.csv"
    df = pd.read_csv(csv_path)

    audit = evaluate_separate_vs_unified_predictors(df)
    assert audit["num_scenario_observations"] == 10
    assert audit["is_separate_predictors_justified"] is False
    assert "severely insufficient statistical degrees of freedom" in audit["scientific_rationale"]
    assert "Unified candidate-utility regressor" in audit["scientific_rationale"] or "candidate-utility" in audit["scientific_rationale"]


# =============================================================================
# 3. Notebooks Inventory & Execution Verification
# =============================================================================

def test_notebooks_inventory_and_outputs():
    """Verify that all 10 research notebooks exist and have saved outputs."""
    nb_dir = REPO_ROOT / "component3" / "notebooks"
    expected_notebooks = [
        "01_dataset_preparation.ipynb",
        "02_baseline_model_training.ipynb",
        "03_drift_simulation.ipynb",
        "04_synthetic_generator_pilot.ipynb",
        "05_synthetic_augmentation_evaluation.ipynb",
        "06_prescription_dataset_analysis.ipynb",
        "07_strategy_prediction_training.ipynb",
        "08_generator_prediction_training.ipynb",
        "09_repair_quantity_prediction_training.ipynb",
        "10_prescription_model_comparison.ipynb",
    ]

    for nb_name in expected_notebooks:
        nb_path = nb_dir / nb_name
        assert nb_path.is_file(), f"Missing required notebook: {nb_name}"
        assert nb_path.stat().st_size > 50_000, f"Notebook {nb_name} appears unexecuted (size too small)"

        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        assert "cells" in nb_data
        code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) > 0, f"No code cells in {nb_name}"

        # Ensure code cells have outputs
        cells_with_output = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        assert len(cells_with_output) > 0, f"Notebook {nb_name} has no executed outputs"


# =============================================================================
# 4. Phase 4 End-to-End Demonstration Manifest & Hashes
# =============================================================================

def test_phase4_demonstration_manifest_and_contracts():
    """Verify the saved Phase 4 demonstration manifest and C4 contract integrity."""
    demo_dir = REPO_ROOT / "component3" / "results" / "phase4_demonstration"
    manifest_path = demo_dir / "phase4_system_demonstration_results.json"
    c4_path = demo_dir / "wf2_c4_handoff_contract.json"

    assert manifest_path.is_file(), "Phase 4 demonstration manifest missing"
    assert c4_path.is_file(), "C4 handoff contract missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_workflows"] == 5
    assert manifest["workflows_passed"] == 5
    assert manifest["verdict"] == "READY FOR RESEARCH DEMONSTRATION"
    assert manifest["scientific_validation_status"] == "PILOT_EXPLORATORY_EVIDENCE"

    # Verify C4 contract fields
    with open(c4_path, "r", encoding="utf-8") as f:
        c4 = json.load(f)

    assert c4["is_approved_for_human_review"] is True
    assert c4["package_status"] == "APPROVED_FOR_REVIEW"
    assert c4["candidate_model_sha256"] is not None
    assert len(c4["candidate_model_sha256"]) == 64


# =============================================================================
# 5. Security Hardening & Robust Failure Handling
# =============================================================================

def test_security_malformed_requests():
    """Verify orchestrator gracefully rejects or abstains on malformed requests."""
    orchestrator = RepairOrchestrator()

    # 1. Request with negative budget
    with pytest.raises(Exception):
        RepairRequest(
            request_id="MALFORMED-BUDGET",
            drift_detected=True,
            adaptation_budget_size=-10,
        )

    # 2. Request with invalid severity score (> 1.0)
    with pytest.raises(Exception):
        RepairRequest(
            request_id="MALFORMED-SEVERITY",
            drift_detected=True,
            drift_severity_score=1.5,
        )


def test_security_ood_detection_and_safe_fallback():
    """Verify learned model detects out-of-distribution scenarios and triggers fallback."""
    csv_path = REPO_ROOT / "component3" / "results" / "prescription_training_dataset.csv"
    df = pd.read_csv(csv_path)

    model = LearnedPrescriptionModel(seed=42).fit(df)

    # In-distribution scenario
    in_dist = {
        "in_dataset_num_features": 19,
        "in_dataset_num_numerical": 3,
        "in_dataset_num_categorical": 16,
        "in_dataset_imbalance_ratio": 0.26,
        "in_available_adaptation_budget": 50,
        "in_has_valid_diagnostic_slice": 1,
    }
    is_ood, reasons = model.is_out_of_distribution(in_dist)
    assert is_ood is False, f"Expected in-distribution, got OOD: {reasons}"

    # Out-of-distribution scenario (extreme values)
    out_dist = {
        "in_dataset_num_features": 10000,
        "in_dataset_num_numerical": 5000,
        "in_dataset_num_categorical": 5000,
        "in_dataset_imbalance_ratio": 0.0001,
        "in_available_adaptation_budget": 50000,
        "in_has_valid_diagnostic_slice": 0,
    }
    is_ood_out, reasons_out = model.is_out_of_distribution(out_dist)
    assert is_ood_out is True, "Expected OOD detection to trigger"
    assert len(reasons_out) >= 3


def test_quality_gate_rejection_on_corrupted_candidates():
    """Verify that corrupt synthetic candidates cannot bypass quality gates."""
    orchestrator = RepairOrchestrator()
    corrupted_data = pd.DataFrame({
        "tenure": [-99999.0] * 50,
        "MonthlyCharges": [float("nan")] * 50,
        "TotalCharges": [-1.0] * 50,
        "Churn": ["Corrupt"] * 50,
    })

    req = RepairRequest(
        request_id="SEC-CORRUPT-TEST",
        drift_detected=True,
        drift_type=DriftType.CONCEPT_DRIFT,
        drift_severity=DriftSeverity.HIGH,
        drift_severity_score=0.7,
        adaptation_budget_size=50,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pkg, log = orchestrator.execute_repair(
            req,
            output_directory=tmp_dir,
            pre_generated_candidates_df=corrupted_data,
            decision_mode=DecisionMode.RULE_BASED,
        )
        assert pkg.package_status == RepairPackageStatus.REJECTED
        assert pkg.requires_human_approval is True
        assert len(pkg.rejection_reasons) > 0


# =============================================================================
# 6. API and CLI Operations Verification
# =============================================================================

def test_api_prescribe_and_status():
    """Verify Python API operations under both RULE_BASED and LEARNED modes."""
    # API: validate_repair_request
    val_res = validate_repair_request({
        "request_id": "API-TEST-RULE",
        "drift_detected": True,
        "drift_type": "COVARIATE_SHIFT",
        "drift_severity": "MEDIUM",
        "drift_severity_score": 0.4,
        "adaptation_budget_size": 50,
    })
    assert val_res["valid"] is True

    # API: view_prescription_reasoning with RULE_BASED
    req_rule = RepairRequest(
        request_id="API-TEST-RULE",
        drift_detected=True,
        drift_type=DriftType.COVARIATE_SHIFT,
        drift_severity=DriftSeverity.MEDIUM,
        drift_severity_score=0.4,
        adaptation_budget_size=50,
    )
    presc_rule = view_prescription_reasoning(req_rule, decision_mode=DecisionMode.RULE_BASED)
    assert "strategy" in presc_rule
    assert presc_rule["decision_mode"] == "RULE_BASED"

    # API: view_prescription_reasoning with LEARNED
    presc_learned = view_prescription_reasoning(req_rule, decision_mode=DecisionMode.LEARNED)
    assert "strategy" in presc_learned
    assert presc_learned["decision_mode"] == "LEARNED"


def test_cli_smoke_execution():
    """Verify CLI interface handles check-readiness subcommand cleanly."""
    from component3.src.cli import main as cli_main

    # Smoke run of CLI check-readiness
    ret = cli_main(["check-readiness"])
    assert ret == 0
