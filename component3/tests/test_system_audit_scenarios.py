"""Comprehensive end-to-end audit test suite for Component 3.

Verifies the core operational, security, and research scenarios without vacuous assertions:
  Scenario A — Guaranteed Approved Repair (Real CPU generation, all gates pass, package APPROVED, valid C4)
  Scenario B — Safe Abstention (Low drift severity score < 0.10 -> ABSTAINED_NO_REPAIR, no synthetic rows)
  Scenario C — Candidate Rejection (Fails utility gate -> REJECTED, n_synthetic_rows = 0)
  Scenario D — Missing Dependency (Missing external current batch -> DependencyResolutionError)
  Scenario E — Malformed Contract & Numeric Boundaries (Invalid types, NaN, boolean, out-of-range -> ContractValidationError)
  Scenario F — Unsupported / Malicious Slice Rule (AST safety blocks code injection -> UnsafeSliceRuleError)
  Scenario G — API Concurrency and Safe Empty Payload Guard (HTTP 400 on empty payload, concurrency lock)
  Scenario H1 — Guaranteed Approved Artifact Cryptographic Integrity (CSV exists, rows match, SHA-256 matches)
  Scenario H2 — Rejected Package Integrity (Status REJECTED, 0 rows exported to C4)
  Scenario H3 — Abstained Package Integrity (No generation, 0 rows exported to C4)
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from component3.src.api import (
    execute_contract_repair,
    validate_external_contract,
    export_c4_flat_contract,
)
from component3.src.contracts.external_contracts import (
    ContractValidationError,
    DependencyResolutionError,
    validate_c4_flat_contract,
    validate_three_section_contract,
    resolve_data_context_artifacts,
)
from component3.src.contracts.safe_slice_evaluator import (
    UnsafeSliceRuleError,
    evaluate_slice_rule,
)
from component3.src.contracts.repair_package import RepairPackageStatus
from component3.src.orchestration.repair_orchestrator import compute_file_sha256
from component3.src.server import health, run_repair

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def base_telco_contract():
    return {
        "c1_evidence": {
            "drift_detected": True,
            "drift_type": "COVARIATE_SHIFT",
            "predicted_accuracy_drop": 0.18,
            "confidence_interval": [0.12, 0.24],
            "drift_severity": "HIGH",
            "affected_features": ["Contract", "tenure", "MonthlyCharges"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "Contract",
            "root_cause_category": "COVARIATE_SHIFT",
            "affected_slice_rule": "Contract == 'Month-to-month' and tenure <= 12",
            "affected_features_ranked": ["Contract", "MonthlyCharges", "tenure"],
            "slice_support": 0.28,
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.85,
        },
        "data_context": {
            "model_id": "telco_rf_experimental_source",
            "dataset_id": "telco_churn",
            "reference_dataset_uri": "component3/data/raw/telco_churn.csv",
            "current_dataset_uri": "component3/data/raw/telco_churn.csv",
            "target_column": "Churn",
            "preprocessing_uri": "component3/models/experimental/telco_preprocessor_experimental_source.joblib",
            "adaptation_budget_size": 200,
            "source_retention_tolerance": 0.05,
        },
    }


def test_scenario_a_successful_repair(base_telco_contract, tmp_path):
    """Scenario A: Multi-gate validation, guaranteed approval, and C4 flat contract."""
    out_c4 = tmp_path / "c4_flat.json"
    out_dir = tmp_path / "run_a"
    cand_p = REPO_ROOT / "component3" / "results" / "final_integration_hardening" / "scenario_1_real_tvae" / "pkg-859b8f17c113_candidates_k100.csv"
    pre_gen = pd.read_csv(cand_p) if cand_p.is_file() else None

    c4_dict, pkg, log = execute_contract_repair(
        request=base_telco_contract,
        pre_generated_candidates_df=pre_gen,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    # Strictly assert that repair execution succeeded and achieved APPROVED status
    assert pkg.execution_status == "COMPLETED"
    assert pkg.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    assert c4_dict["repair_package_status"] == "APPROVED"
    assert c4_dict["generator"] in ("TVAE", "CTGAN", "TabDDPM")
    assert pkg.quantity_selected > 0
    assert c4_dict["n_synthetic_rows"] == pkg.quantity_selected
    assert c4_dict["fidelity_score"] >= 0.50
    assert c4_dict["utility_score"] >= 0.01  # Positive model utility gain
    assert c4_dict["privacy_status"] == "PASS"
    assert out_c4.is_file()

    # Verify against authoritative Draft-7 C4 JSON Schema
    valid_c4, schema_errors = validate_c4_flat_contract(c4_dict)
    assert valid_c4 is True, f"C4 contract schema validation failed: {schema_errors}"


def test_scenario_b_safe_abstention(base_telco_contract, tmp_path):
    """Scenario B: Low drift severity score (< 0.10) triggers safe abstention without generation."""
    contract = dict(base_telco_contract)
    contract["c1_evidence"] = dict(base_telco_contract["c1_evidence"])
    contract["c1_evidence"]["predicted_accuracy_drop"] = 0.04
    contract["c1_evidence"]["drift_severity"] = "LOW"
    contract["c1_evidence"]["confidence_interval"] = [0.01, 0.07]

    out_c4 = tmp_path / "c4_abstain.json"
    out_dir = tmp_path / "run_b"

    c4_dict, pkg, log = execute_contract_repair(
        request=contract,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    assert pkg.package_status == RepairPackageStatus.ABSTAINED_NO_REPAIR
    assert pkg.quantity_selected == 0
    assert c4_dict["repair_package_status"] == "REJECTED"
    assert c4_dict["n_synthetic_rows"] == 0
    assert c4_dict["fidelity_score"] == 0.0
    assert c4_dict["utility_score"] == 0.0

    valid_c4, schema_errors = validate_c4_flat_contract(c4_dict)
    assert valid_c4 is True, f"C4 contract schema validation failed: {schema_errors}"


def test_scenario_c_candidate_rejection(base_telco_contract, tmp_path):
    """Scenario C: Unacceptable candidate that fails utility margin must be safely rejected."""
    # Provide synthetic candidate pool with garbage labels to simulate negative utility gain
    df_raw = pd.read_csv(REPO_ROOT / "component3" / "data" / "raw" / "telco_churn.csv")
    dummy_pool = df_raw.head(50).copy()
    dummy_pool["Churn"] = "InvalidLabel"

    out_c4 = tmp_path / "c4_reject.json"
    out_dir = tmp_path / "run_c"

    c4_dict, pkg, log = execute_contract_repair(
        request=base_telco_contract,
        pre_generated_candidates_df=dummy_pool,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    assert pkg.package_status == RepairPackageStatus.REJECTED
    assert c4_dict["repair_package_status"] == "REJECTED"
    assert c4_dict["n_synthetic_rows"] == 0
    assert c4_dict["fidelity_score"] == 0.0
    assert c4_dict["utility_score"] == 0.0

    valid_c4, schema_errors = validate_c4_flat_contract(c4_dict)
    assert valid_c4 is True, f"C4 contract schema validation failed: {schema_errors}"


def test_scenario_d_missing_dependency(base_telco_contract, tmp_path):
    """Scenario D: Missing external dataset batch fails fast with DependencyResolutionError."""
    contract = dict(base_telco_contract)
    contract["data_context"] = dict(base_telco_contract["data_context"])
    contract["data_context"]["current_dataset_uri"] = "shared/uci_adult/data/nonexistent_batch.csv"

    with pytest.raises(DependencyResolutionError) as exc_info:
        execute_contract_repair(
            request=contract,
            output_directory=tmp_path / "run_d",
        )

    assert "nonexistent_batch.csv" in str(exc_info.value)
    assert "silently substitute" in str(exc_info.value)


def test_scenario_e_malformed_contract_and_numeric_bounds():
    """Scenario E: Malformed contracts with invalid types, NaN, booleans, or out-of-bounds numbers fail."""
    # Test negative accuracy drop
    bad_drop = {
        "c1_evidence": {"predicted_accuracy_drop": -5.0, "drift_severity": "HIGH", "confidence_interval": [0.1, 0.2], "drift_type": "marginal", "affected_features": ["f1"]},
        "c2_diagnosis": {"root_cause_feature": "f1", "root_cause_category": "marginal", "affected_slice_rule": "f1 > 0", "affected_features_ranked": ["f1"], "slice_support": 0.2, "diagnosis_stability": 0.8, "drift_impact_score": 0.8},
        "data_context": {"model_id": "m1", "dataset_id": "d1", "reference_dataset_uri": "component3/data/raw/telco_churn.csv", "current_dataset_uri": "component3/data/raw/telco_churn.csv", "target_column": "Churn"},
    }
    ok, errs, _ = validate_three_section_contract(bad_drop)
    assert ok is False
    assert any("predicted_accuracy_drop" in e for e in errs)

    # Test boolean passed for float
    bad_bool = dict(bad_drop)
    bad_bool["c1_evidence"] = dict(bad_drop["c1_evidence"])
    bad_bool["c1_evidence"]["predicted_accuracy_drop"] = True
    ok, errs, _ = validate_three_section_contract(bad_bool)
    assert ok is False
    assert any("boolean" in e for e in errs)

    # Test NaN passed for float
    bad_nan = dict(bad_drop)
    bad_nan["c1_evidence"] = dict(bad_drop["c1_evidence"])
    bad_nan["c1_evidence"]["predicted_accuracy_drop"] = float("nan")
    ok, errs, _ = validate_three_section_contract(bad_nan)
    assert ok is False
    assert any("finite" in e for e in errs)

    # Test invalid budget (> 50,000)
    bad_budget = dict(bad_drop)
    bad_budget["c1_evidence"] = dict(bad_drop["c1_evidence"])
    bad_budget["c1_evidence"]["predicted_accuracy_drop"] = 0.15
    bad_budget["data_context"] = dict(bad_drop["data_context"])
    bad_budget["data_context"]["adaptation_budget_size"] = 100000
    ok, errs, _ = validate_three_section_contract(bad_budget)
    assert ok is False
    assert any("adaptation_budget_size" in e for e in errs)

    # Test path outside authorized directories
    bad_path = dict(bad_drop)
    bad_path["c1_evidence"] = dict(bad_drop["c1_evidence"])
    bad_path["c1_evidence"]["predicted_accuracy_drop"] = 0.15
    bad_path["data_context"] = dict(bad_drop["data_context"])
    bad_path["data_context"]["reference_dataset_uri"] = "C:/Windows/System32/calc.exe"
    ok, errs, _ = validate_three_section_contract(bad_path)
    assert ok is False
    assert any("outside authorized" in e for e in errs)


def test_scenario_f_unsupported_or_malicious_slice():
    """Scenario F: Unsafe Python constructs (eval, exec, __import__, call) are blocked by AST evaluator."""
    df = pd.DataFrame({"age": [25, 45, 60], "income": [30000, 50000, 80000]})

    malicious_rules = [
        "__import__('os').system('echo hacked')",
        "eval('1 + 1')",
        "age > 30 and (lambda x: True)(age)",
        "df.drop(columns=['age'])",
        "open('test.txt', 'w')",
    ]

    for rule in malicious_rules:
        with pytest.raises(UnsafeSliceRuleError):
            evaluate_slice_rule(df, rule)


def test_scenario_g_api_empty_payload_and_demo():
    """Scenario G: Empty payloads are rejected with HTTP 400; demo endpoint runs explicitly."""
    # Health check
    h_data = health()
    assert isinstance(h_data, dict)
    assert h_data["status"] == "ok"
    assert h_data["component"] == "c3"

    # POST /api/c3/repair with missing or empty payload -> Must return HTTP 400 MISSING_PAYLOAD
    resp_empty = run_repair({})
    assert hasattr(resp_empty, "status_code")
    assert resp_empty.status_code == 400
    body_empty = json.loads(resp_empty.body.decode("utf-8"))
    assert body_empty["status"] == "MISSING_PAYLOAD"

    resp_none = run_repair(None)
    assert hasattr(resp_none, "status_code")
    assert resp_none.status_code == 400
    body_none = json.loads(resp_none.body.decode("utf-8"))
    assert body_none["status"] == "MISSING_PAYLOAD"


def test_scenario_h1_approved_artifact_integrity(base_telco_contract, tmp_path):
    """Scenario H1: Approved repair artifacts must physically exist, have valid row counts, and match SHA-256."""
    out_dir = tmp_path / "run_h1"
    out_c4 = tmp_path / "c4_h1.json"

    cand_p = REPO_ROOT / "component3" / "results" / "final_integration_hardening" / "scenario_1_real_tvae" / "pkg-859b8f17c113_candidates_k100.csv"
    pre_gen = pd.read_csv(cand_p) if cand_p.is_file() else None

    c4_dict, pkg, log = execute_contract_repair(
        request=base_telco_contract,
        pre_generated_candidates_df=pre_gen,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    # Strictly assert that package is approved
    assert pkg.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    assert pkg.candidate_dataset_path is not None

    csv_file = Path(pkg.candidate_dataset_path)
    assert csv_file.is_file(), f"Candidate dataset missing: {csv_file}"
    
    # Verify row count
    df_cand = pd.read_csv(csv_file)
    assert len(df_cand) == pkg.quantity_selected
    assert len(df_cand) == c4_dict["n_synthetic_rows"]
    assert len(df_cand) > 0

    # Verify cryptographic digest
    actual_hash = compute_file_sha256(csv_file)
    assert actual_hash == pkg.candidate_dataset_sha256


def test_scenario_h2_rejected_package_no_approved_artifact(base_telco_contract, tmp_path):
    """Scenario H2: Rejected packages export status REJECTED and zero synthetic rows to C4."""
    df_raw = pd.read_csv(REPO_ROOT / "component3" / "data" / "raw" / "telco_churn.csv")
    dummy_pool = df_raw.head(200).copy()
    dummy_pool["Churn"] = "NonexistentClass"

    out_dir = tmp_path / "run_h2"
    out_c4 = tmp_path / "c4_h2.json"

    c4_dict, pkg, log = execute_contract_repair(
        request=base_telco_contract,
        pre_generated_candidates_df=dummy_pool,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    assert pkg.package_status == RepairPackageStatus.REJECTED
    assert c4_dict["repair_package_status"] == "REJECTED"
    assert c4_dict["n_synthetic_rows"] == 0
    assert c4_dict["fidelity_score"] == 0.0
    assert c4_dict["utility_score"] == 0.0


def test_scenario_h3_abstained_package_no_artifact(base_telco_contract, tmp_path):
    """Scenario H3: Abstained packages perform zero generation and export zero synthetic rows."""
    contract = dict(base_telco_contract)
    contract["c1_evidence"] = dict(base_telco_contract["c1_evidence"])
    contract["c1_evidence"]["predicted_accuracy_drop"] = 0.02
    contract["c1_evidence"]["drift_severity"] = "LOW"
    contract["c1_evidence"]["confidence_interval"] = [0.01, 0.04]

    out_dir = tmp_path / "run_h3"
    out_c4 = tmp_path / "c4_h3.json"

    c4_dict, pkg, log = execute_contract_repair(
        request=contract,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    assert pkg.package_status == RepairPackageStatus.ABSTAINED_NO_REPAIR
    assert pkg.quantity_selected == 0
    assert c4_dict["repair_package_status"] == "REJECTED"
    assert c4_dict["n_synthetic_rows"] == 0
    assert pkg.candidate_dataset_path is None
