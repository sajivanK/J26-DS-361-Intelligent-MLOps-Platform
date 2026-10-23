"""Comprehensive hardening failure and simulated consumer test suite for Component 3.

Validates that Component 3 handles all edge cases, failure conditions, security boundaries,
and artifact integrity checks gracefully with explicit error messages and honest contract outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from component3.src.contracts.external_contracts import (
    validate_three_section_contract,
    validate_c4_flat_contract,
    DependencyResolutionError,
    ContractValidationError,
)
from component3.src.contracts.safe_slice_evaluator import (
    evaluate_slice_rule,
    UnsafeSliceRuleError,
)
from component3.src.contracts.repair_package import (
    RepairPackage,
    RepairPackageStatus,
)
from component3.src.adapters.c4_retraining_adapter import C4RetrainingAdapter
from component3.src.generators.generator_selector import GeneratorSelector, GeneratorExecutionError
from component3.src.generators.plugin_manager import GeneratorPluginManager, GeneratorPluginError
from component3.src.utils.artifact_storage import (
    ArtifactStorageManager,
    compute_file_sha256,
    verify_file_integrity,
    resolve_portable_uri,
    to_portable_relative_uri,
)
from component3.src.api import (
    validate_external_contract,
    export_c4_flat_contract,
    execute_contract_repair,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def valid_telco_base_payload() -> dict:
    """Return a baseline valid 3-section payload for Telco Churn."""
    return {
        "c1_evidence": {
            "predicted_harm": 0.15,
            "confidence_interval": [0.08, 0.22],
            "drift_type": "covariate_shift",
            "drift_severity": "HIGH",
            "affected_features": ["Contract", "MonthlyCharges", "tenure"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "Contract",
            "root_cause_category": "covariate_shift",
            "affected_slice_rule": "Contract == 'Month-to-month' and tenure <= 12",
            "affected_features_ranked": ["Contract", "MonthlyCharges", "tenure"],
            "affected_value_ranges": {"Contract": ["Month-to-month"], "tenure": [0, 12]},
            "slice_support": 0.28,
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.15,
            "temporal_window": 30,
        },
        "data_context": {
            "dataset_id": "telco_churn",
            "model_id": "telco_rf_experimental_source",
            "reference_dataset_uri": "component3/data/raw/telco_churn.csv",
            "current_dataset_uri": "component3/data/raw/telco_churn.csv",
            "target_column": "Churn",
            "schema_version": "1.0",
            "label_status": "AVAILABLE",
            "preprocessing_uri": "component3/models/experimental/telco_preprocessor_experimental_source.joblib",
        },
    }


# =============================================================================
# 1. Dependency Resolution & Missing Artifact Tests
# =============================================================================

def test_missing_reference_dataset_raises_dependency_error(valid_telco_base_payload: dict, tmp_path: Path):
    payload = dict(valid_telco_base_payload)
    payload["data_context"] = dict(payload["data_context"])
    payload["data_context"]["reference_dataset_uri"] = "component3/data/raw/nonexistent_reference.csv"

    with pytest.raises(DependencyResolutionError, match="Reference dataset not found"):
        execute_contract_repair(request=payload, output_directory=tmp_path)


def test_missing_current_dataset_raises_dependency_error(valid_telco_base_payload: dict, tmp_path: Path):
    payload = dict(valid_telco_base_payload)
    payload["data_context"] = dict(payload["data_context"])
    payload["data_context"]["current_dataset_uri"] = "component3/data/raw/nonexistent_current.csv"

    with pytest.raises(DependencyResolutionError, match="Current drifted dataset not found"):
        execute_contract_repair(request=payload, output_directory=tmp_path)


def test_missing_source_model_raises_dependency_error(valid_telco_base_payload: dict, tmp_path: Path):
    payload = dict(valid_telco_base_payload)
    payload["data_context"] = dict(payload["data_context"])
    payload["data_context"]["model_id"] = "nonexistent_model_id_v999"

    with pytest.raises(DependencyResolutionError, match="nonexistent_model_id_v999"):
        execute_contract_repair(request=payload, output_directory=tmp_path)


def test_missing_external_uci_adult_artifacts_reported_without_fallback(tmp_path: Path):
    adult_payload = {
        "c1_evidence": {
            "predicted_harm": 0.08,
            "confidence_interval": [0.03, 0.13],
            "drift_type": "covariate_shift",
            "drift_severity": "MODERATE",
            "affected_features": ["capital-gain", "education-num"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "capital-gain",
            "root_cause_category": "covariate_shift",
            "affected_slice_rule": "age > 40 and education_num > 12",
            "affected_features_ranked": ["capital-gain", "education-num"],
            "affected_value_ranges": {},
            "slice_support": 0.15,
            "diagnosis_stability": 0.80,
            "drift_impact_score": 0.10,
            "temporal_window": 14,
        },
        "data_context": {
            "dataset_id": "uci_adult",
            "model_id": "champion-v12",
            "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
            "current_dataset_uri": "shared/uci_adult/data/current.csv",
            "target_column": "income",
            "schema_version": "1.0",
            "label_status": "AVAILABLE",
        },
    }
    with pytest.raises(DependencyResolutionError) as exc_info:
        execute_contract_repair(request=adult_payload, output_directory=tmp_path)

    assert "Component 3 will not silently substitute another" in str(exc_info.value)


# =============================================================================
# 2. Schema Validation & Input Boundary Tests
# =============================================================================

def test_wrong_target_column_rejected(valid_telco_base_payload: dict):
    payload = dict(valid_telco_base_payload)
    payload["data_context"] = dict(payload["data_context"])
    payload["data_context"]["target_column"] = ""

    ok, errors, _ = validate_three_section_contract(payload)
    assert not ok
    assert any("target_column" in err for err in errors)


def test_unsupported_drift_type_rejected(valid_telco_base_payload: dict):
    payload = dict(valid_telco_base_payload)
    payload["c1_evidence"] = dict(payload["c1_evidence"])
    payload["c1_evidence"]["drift_type"] = "MALICIOUS_UNSUPPORTED_DRIFT"

    ok, errors, _ = validate_three_section_contract(payload)
    assert not ok
    assert any("drift_type" in err for err in errors)


def test_invalid_confidence_interval_rejected(valid_telco_base_payload: dict):
    payload = dict(valid_telco_base_payload)
    payload["c1_evidence"] = dict(payload["c1_evidence"])
    payload["c1_evidence"]["confidence_interval"] = [0.25, 0.05]  # lower > upper

    ok, errors, _ = validate_three_section_contract(payload)
    assert not ok
    assert any("confidence_interval" in err for err in errors)


# =============================================================================
# 3. Safe Slice Evaluator & Security Tests
# =============================================================================

def test_slice_rule_code_injection_prevention():
    df = pd.DataFrame({"age": [25, 45, 60], "income": [30000, 80000, 120000]})

    malicious_rules = [
        "__import__('os').system('dir')",
        "eval('1 + 1') == 2",
        "open('test.txt', 'w')",
        "exec('x = 1')",
        "__builtins__.__import__('os')",
    ]

    for rule in malicious_rules:
        with pytest.raises(UnsafeSliceRuleError):
            evaluate_slice_rule(df, rule)


def test_slice_rule_empty_matching():
    df = pd.DataFrame({"age": [20, 30, 40], "Contract": ["One year", "Two year", "Two year"]})
    rule = "Contract == 'Month-to-month'"
    mask = evaluate_slice_rule(df, rule)
    assert len(mask) == len(df)
    assert not mask.any()  # 0 rows matched


def test_slice_rule_syntax_error_handled():
    df = pd.DataFrame({"age": [20, 30, 40]})
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df, "age == == 20")


# =============================================================================
# 4. Generator Plugins & Feasibility Tests
# =============================================================================

def test_unregistered_generator_selection():
    mgr = GeneratorPluginManager()
    with pytest.raises(GeneratorPluginError, match="Unknown generator"):
        mgr.get_generator_class("unsupported_gan_v99")


def test_insufficient_sample_size_fails_compatibility():
    selector = GeneratorSelector()
    small_df = pd.DataFrame({"colA": [1, 2, 3], "colB": ["x", "y", "z"]})
    compat, reason = selector.check_compatibility(
        generator_name="tvae",
        df=small_df,
        categorical_columns=["colB"],
        numerical_columns=["colA"],
    )
    assert not compat
    assert "require at least 30 rows" in reason


# =============================================================================
# 5. Output Contract & Handoff Integrity Tests
# =============================================================================

def test_c4_flat_contract_validation_rejects_missing_fields():
    invalid_c4 = {
        "repair_dataset_uri": "component3/data/repairs/sample.csv",
        "repair_package_status": "APPROVED",
        # missing generator, n_synthetic_rows, etc.
    }
    ok, errors = validate_c4_flat_contract(invalid_c4)
    assert not ok
    assert len(errors) > 0


def test_c4_flat_contract_validation_rejects_out_of_range():
    invalid_c4 = {
        "repair_dataset_uri": "component3/data/repairs/sample.csv",
        "repair_package_status": "APPROVED",
        "generator": "TVAE",
        "n_synthetic_rows": 100,
        "augmentation_ratio": 1.5,  # > 1.0 invalid
        "fidelity_score": 0.90,
        "utility_score": 0.85,
        "privacy_status": "PASS",
        "target_coverage": 0.95,
        "prescription_confidence": 0.88,
        "repair_version": "c3-001",
        "repair_timestamp": "2026-10-21T10:00:00Z",
    }
    ok, errors = validate_c4_flat_contract(invalid_c4)
    assert not ok
    assert any("augmentation_ratio" in err for err in errors)


def test_no_drift_honest_output_contract():
    pkg = RepairPackage(
        request_id="REQ-NODRIFT-001",
        dataset_id="telco_churn",
        model_id="telco_rf_experimental_source",
        execution_status="COMPLETED",
        package_status=RepairPackageStatus.ABSTAINED_NO_REPAIR,
        chosen_strategy="NO_REPAIR",
        rejection_reasons=("No significant drift detected.",),
    )
    adapter = C4RetrainingAdapter()
    flat = adapter.to_c4_flat_contract(pkg)
    assert flat["repair_package_status"] == "REJECTED"
    assert flat["n_synthetic_rows"] == 0
    assert flat["augmentation_ratio"] == 0.0
    assert flat["fidelity_score"] == 0.0
    assert flat["utility_score"] == 0.0
    assert flat["target_coverage"] == 0.0

    ok, errors = validate_c4_flat_contract(flat)
    assert ok, f"Expected valid schema for rejected package: {errors}"


# =============================================================================
# 6. Artifact Storage Portability & SHA-256 Verification Tests
# =============================================================================

def test_atomic_write_and_integrity_verification(tmp_path: Path):
    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    dest = tmp_path / "test_data.csv"

    saved_p, sha = ArtifactStorageManager.atomic_write_dataframe(df, dest)
    assert saved_p.is_file()
    assert len(sha) == 64

    # Verify integrity passes
    ok, msg = verify_file_integrity(saved_p, sha)
    assert ok

    # Tamper with file content
    with open(saved_p, "a", encoding="utf-8") as f:
        f.write("tampered,row\n")

    # Verify integrity fails
    ok_tampered, msg_tampered = verify_file_integrity(saved_p, sha)
    assert not ok_tampered
    assert "mismatch" in msg_tampered


def test_portable_uri_resolution(tmp_path: Path):
    test_file = tmp_path / "artifact.csv"
    test_file.write_text("a,b\n1,2\n")

    rel_uri = to_portable_relative_uri(test_file, base_root=tmp_path)
    assert "\\" not in rel_uri

    resolved = resolve_portable_uri(rel_uri, base_root=tmp_path)
    assert resolved.is_file()


def test_missing_portable_uri_raises_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="could not be resolved"):
        resolve_portable_uri("nonexistent/artifact.csv", base_root=tmp_path)


# =============================================================================
# 7. Simulated Component 4 Consumer Integration Test
# =============================================================================

def test_simulated_c4_consumer_pipeline(tmp_path: Path, valid_telco_base_payload: dict):
    """Simulates Component 4 reading the exported JSON contract and candidate CSV."""
    # 1. Generate a mock approved repair package with real artifacts
    cand_df = pd.DataFrame({
        "Contract": ["Month-to-month"] * 50,
        "tenure": [6] * 50,
        "MonthlyCharges": [75.0] * 50,
        "Churn": ["Yes"] * 25 + ["No"] * 25,
    })
    csv_path = tmp_path / "c3_repair_candidates.csv"
    cand_df.to_csv(csv_path, index=False)
    cand_hash = compute_file_sha256(csv_path)

    pkg = RepairPackage(
        request_id="REQ-SIM-C4-001",
        run_id="RUN-SIM-001",
        package_id="PKG-SIM-001",
        dataset_id="telco_churn",
        model_id="telco_rf_experimental_source",
        chosen_strategy="TARGETED_SYNTHETIC_REPAIR",
        package_status=RepairPackageStatus.APPROVED_FOR_REVIEW,
        quantity_selected=50,
        generator_info={"generator": "TVAE"},
        candidate_dataset_path=str(csv_path),
        candidate_dataset_sha256=cand_hash,
        utility_metrics={"training_rows": 200, "target_candidate_f1": 0.84},
        fidelity_indicators={"fidelity_score": 0.89},
        privacy_risk_indicators={"status": "PASS"},
        validation_report={"target_coverage": 0.94},
        downstream_handoff_metadata={"evidence_strength": "CONFIRMED_BENCHMARK"},
    )

    c4_json_path = tmp_path / "c4_handoff.json"
    adapter = C4RetrainingAdapter()
    exported_contract = adapter.export_flat_contract(pkg, output_path=c4_json_path)

    # 2. Simulated C4 consumer actions:
    assert c4_json_path.is_file()
    with open(c4_json_path, "r", encoding="utf-8") as f:
        consumer_payload = json.load(f)

    # Validate against C4 schema
    is_valid, validation_errors = validate_c4_flat_contract(consumer_payload)
    assert is_valid, f"C4 contract validation failed: {validation_errors}"

    # Check status gate
    assert consumer_payload["repair_package_status"] == "APPROVED"
    assert consumer_payload["privacy_status"] == "PASS"

    # Resolve and load repair dataset
    dataset_uri = consumer_payload["repair_dataset_uri"]
    resolved_dataset_path = Path(dataset_uri)
    if not resolved_dataset_path.is_absolute():
        resolved_dataset_path = REPO_ROOT / resolved_dataset_path

    assert resolved_dataset_path.is_file()
    retraining_df = pd.read_csv(resolved_dataset_path)
    assert len(retraining_df) == consumer_payload["n_synthetic_rows"]
    assert "Churn" in retraining_df.columns


def test_output_directory_blocked_raises_os_error(tmp_path: Path):
    """Verify that execution fails cleanly when output directory cannot be created."""
    blocked_file = tmp_path / "blocked_dir"
    blocked_file.write_text("existing_file_blocks_directory_creation")

    # Test artifact storage directly; this failure does not require a model run.
    with pytest.raises((OSError, FileExistsError, NotADirectoryError)):
        ArtifactStorageManager.atomic_write_dataframe(pd.DataFrame({"x": [1]}), blocked_file / "child_dir/data.csv")

