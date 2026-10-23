"""Comprehensive Integration Tests for External Data Contracts and Boundary Adapters.

Tests:
1. Exact Adult 3-section input contract validation.
2. Missing Adult artifact dependency resolution (no silent Telco substitution).
3. C1 Drift Adapter: predicted accuracy drop, confidence interval, interaction drift, derived detection.
4. C2 Diagnosis Adapter: affected_slice_rule, operator normalization, ranked features, AST safe evaluation.
5. C4 Flat Output Contract: exact 12-field schema validation, team validation helper compliance.
6. Real-data end-to-end execution through canonical execute_contract_repair() entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from component3.src.adapters.c1_drift_adapter import C1DriftAdapter
from component3.src.adapters.c2_diagnosis_adapter import C2DiagnosisAdapter
from component3.src.adapters.c4_retraining_adapter import C4RetrainingAdapter
from component3.src.api import (
    _parse_request,
    execute_contract_repair,
    export_c4_flat_contract,
    validate_external_contract,
    validate_repair_request,
)
from component3.src.contracts.external_contracts import (
    ContractValidationError,
    DependencyResolutionError,
    validate_c4_flat_contract,
    validate_three_section_contract,
)
from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
from component3.src.contracts.repair_request import DriftSeverity, DriftType, RepairRequest
from component3.src.contracts.safe_slice_evaluator import (
    UnsafeSliceRuleError,
    evaluate_slice_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def exact_adult_payload() -> dict:
    """Exact 3-section contract payload provided by the team."""
    return {
        "c1_evidence": {
            "predicted_accuracy_drop": 14.2,
            "confidence_interval": [10.9, 17.5],
            "drift_type": "interaction",
            "drift_severity": "high",
            "affected_features": [
                "education_num",
                "occupation"
            ]
        },
        "c2_diagnosis": {
            "root_cause_feature": "education_num",
            "root_cause_category": "interaction",
            "affected_slice_rule": "education_num > 12 AND occupation == 'Tech-support'",
            "affected_features_ranked": [
                "education_num",
                "occupation",
                "age"
            ],
            "affected_value_ranges": {
                "education_num": [12, 16],
                "occupation": [
                    "Tech-support",
                    "Prof-specialty"
                ]
            },
            "slice_support": 0.18,
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.72,
            "temporal_window": 3
        },
        "data_context": {
            "model_id": "champion-v12",
            "dataset_id": "adult-v1",
            "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
            "current_dataset_uri": "shared/uci_adult/data/current.csv",
            "target_column": "income",
            "schema_version": "adult-schema-v1",
            "label_status": "AVAILABLE",
            "preprocessing_uri": "shared/uci_adult/preprocessing.py"
        }
    }


@pytest.fixture
def real_telco_contract_payload() -> dict:
    """Executable 3-section payload referencing actual existing Telco artifacts."""
    return {
        "c1_evidence": {
            "predicted_accuracy_drop": 14.2,
            "confidence_interval": [10.9, 17.5],
            "drift_type": "interaction",
            "drift_severity": "high",
            "affected_features": [
                "tenure",
                "MonthlyCharges"
            ]
        },
        "c2_diagnosis": {
            "root_cause_feature": "Contract",
            "root_cause_category": "covariate_shift",
            "affected_slice_rule": "Contract == 'Month-to-month' and tenure <= 12",
            "affected_features_ranked": [
                "Contract",
                "MonthlyCharges",
                "tenure"
            ],
            "affected_value_ranges": {
                "Contract": ["Month-to-month"],
                "tenure": [0, 12]
            },
            "slice_support": 0.28,
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.85,
            "temporal_window": 2
        },
        "data_context": {
            "model_id": "telco_rf_experimental_source",
            "dataset_id": "telco_churn",
            "reference_dataset_uri": "component3/data/raw/telco_churn.csv",
            "current_dataset_uri": "component3/data/raw/telco_churn.csv",
            "target_column": "Churn",
            "schema_version": "1.0.0",
            "label_status": "AVAILABLE",
            "adaptation_budget_size": 200,
            "source_retention_tolerance": 0.05
        }
    }


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_exact_adult_input_schema_validation(exact_adult_payload):
    """Verify that the exact Adult 3-section contract passes schema validation."""
    ok, errors, container = validate_three_section_contract(exact_adult_payload)
    assert ok is True
    assert len(errors) == 0
    assert container is not None

    # C1 evidence
    c1 = container.c1_evidence
    assert c1.predicted_accuracy_drop == 14.2
    assert c1.predicted_harm_proportion == 0.142
    assert c1.confidence_interval == (10.9, 17.5)
    assert c1.confidence_interval_proportions == (0.109, 0.175)
    assert c1.drift_type == "interaction"
    assert c1.drift_severity == "high"
    assert c1.is_drift_detected is True
    assert c1.affected_features == ("education_num", "occupation")

    # C2 diagnosis
    c2 = container.c2_diagnosis
    assert c2.root_cause_feature == "education_num"
    assert c2.normalized_slice_rule == "education_num > 12 and occupation == 'Tech-support'"
    assert c2.affected_features_ranked == ("education_num", "occupation", "age")
    assert c2.slice_support == 0.18
    assert c2.diagnosis_stability == 0.85
    assert c2.drift_impact_score == 0.72

    # Data context
    dc = container.data_context
    assert dc.model_id == "champion-v12"
    assert dc.dataset_id == "adult-v1"
    assert dc.has_authentic_labels is True


def test_missing_adult_artifacts_dependency_resolution(exact_adult_payload):
    """Verify explicit DependencyResolutionError when referenced Adult artifacts are absent."""
    with pytest.raises(DependencyResolutionError) as exc_info:
        _parse_request(exact_adult_payload)

    err_msg = str(exc_info.value)
    # Must report missing file and must NOT silently use Telco defaults
    assert "not found at URI" in err_msg or "neither registered in ModelRegistry nor found on disk" in err_msg
    assert "telco_churn" not in err_msg


def test_c1_drift_adapter_all_semantics():
    """Test C1 adapter field normalization, severity mapping, and derived detection."""
    adapter = C1DriftAdapter(default_dataset_id="telco_churn")

    # 1. Percentage accuracy drop -> proportion score
    payload_pct = {
        "predicted_accuracy_drop": 14.2,
        "drift_type": "interaction",
        "drift_severity": "high",
        "affected_features": ["f1", "f2"],
    }
    req1 = adapter.adapt(payload_pct)
    assert req1.drift_detected is True
    assert req1.drift_type == DriftType.INTERACTION
    assert req1.drift_severity == DriftSeverity.HIGH
    assert req1.drift_severity_score == 0.142

    # 2. Predicted harm as proportion [0, 1]
    payload_prop = {
        "predicted_harm": 0.08,
        "drift_type": "combined",
        "drift_severity": "moderate",
        "features_drifted": ["col_a"],
    }
    req2 = adapter.adapt(payload_prop)
    assert req2.drift_detected is True
    assert req2.drift_severity == DriftSeverity.MEDIUM
    assert req2.drift_severity_score == 0.08

    # 3. Explicit no drift
    payload_none = {
        "drift_type": "none",
        "drift_severity": "none",
        "predicted_harm": 0.0,
    }
    req3 = adapter.adapt(payload_none)
    assert req3.drift_detected is False
    assert req3.drift_severity == DriftSeverity.NONE
    assert req3.drift_type == DriftType.NONE


def test_c2_diagnosis_adapter_safe_ast_evaluation():
    """Test C2 adapter slice parsing and restricted AST evaluator mask generation."""
    adapter = C2DiagnosisAdapter()
    df_test = pd.DataFrame({
        "education_num": [10, 13, 14, 11],
        "occupation": ["Sales", "Tech-support", "Craft-repair", "Tech-support"],
        "age": [25, 35, 45, 55],
    })

    c2_payload = {
        "root_cause_feature": "education_num",
        "root_cause_category": "interaction",
        "affected_slice_rule": "education_num > 12 AND occupation == 'Tech-support'",
        "affected_features_ranked": ["education_num", "occupation"],
        "slice_support": 0.25,
        "diagnosis_stability": 0.88,
    }

    slice_def = adapter.adapt(c2_payload)
    assert slice_def.slice_predicate == "education_num > 12 and occupation == 'Tech-support'"
    assert "education_num" in slice_def.affected_features

    # Evaluate mask safely
    mask = slice_def.evaluate_mask(df_test)
    assert mask.tolist() == [False, True, False, False]


def test_safe_slice_evaluator_rejects_unsafe_constructs():
    """Verify that restricted AST evaluator rejects arbitrary code, function calls, and syntax errors."""
    df_dummy = pd.DataFrame({"age": [20, 30], "income": [50, 100]})

    # Unsafe function call
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df_dummy, "__import__('os').system('echo pwned')")

    # Unsafe attribute access
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df_dummy, "age.__class__.__bases__")

    # Unknown column
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df_dummy, "nonexistent_col > 10")

    # Syntax error
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df_dummy, "age >")


def test_c4_flat_contract_export_matches_shared_schema():
    """Verify that C4 flat export matches all 12 fields in shared/contracts/c3_output_schema.json."""
    pkg = RepairPackage(
        package_id="PKG-AUDIT-001",
        request_id="REQ-AUDIT-001",
        run_id="RUN-AUDIT-001",
        chosen_strategy="SYNTHETIC_WEIGHTED_HYBRID",
        package_status=RepairPackageStatus.APPROVED_FOR_REVIEW,
        quantity_selected=500,
        candidate_dataset_path="component3/results/runs/run_test/synthetic.csv",
        validation_report={"target_coverage": 0.93},
        fidelity_indicators={"fidelity_score": 0.91, "mean_ks_statistic": 0.09},
        privacy_risk_indicators={"status": "PASS"},
        utility_metrics={"target_candidate_f1": 0.87, "training_rows": 1666},
    )

    c4_dict = export_c4_flat_contract(pkg, repair_version="c3-001")

    # 1. Component 3 validator
    ok, errs = validate_c4_flat_contract(c4_dict)
    assert ok is True, f"Validation errors: {errs}"

    # 2. Team shared schema check
    schema_path = REPO_ROOT / "shared" / "contracts" / "c3_output_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    required_fields = [f for f, v in schema["output_fields"].items() if v.get("required", False)]
    assert len(required_fields) == 12

    for req_f in required_fields:
        assert req_f in c4_dict, f"Missing required field: {req_f}"

    # Enum checks
    assert c4_dict["repair_package_status"] == "APPROVED"
    assert c4_dict["generator"] in ("CTGAN", "TVAE", "TabDDPM")
    assert c4_dict["privacy_status"] in ("PASS", "FAIL")

    # Bounded float ranges
    for flt_col in ("augmentation_ratio", "fidelity_score", "utility_score", "target_coverage", "prescription_confidence"):
        val = c4_dict[flt_col]
        assert 0.0 <= val <= 1.0, f"{flt_col} out of range [0, 1]: {val}"


def test_real_data_contract_execution_end_to_end(real_telco_contract_payload, tmp_path):
    """Execute complete end-to-end contract workflow using available real data with guaranteed approval."""
    out_c4_file = tmp_path / "c4_output.json"
    out_dir = tmp_path / "run_out"
    cand_p = REPO_ROOT / "component3" / "results" / "final_integration_hardening" / "scenario_1_real_tvae" / "pkg-859b8f17c113_candidates_k100.csv"
    pre_gen = pd.read_csv(cand_p) if cand_p.is_file() else None

    c4_dict, pkg, log = execute_contract_repair(
        request=real_telco_contract_payload,
        pre_generated_candidates_df=pre_gen,
        output_c4_path=out_c4_file,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    # Strictly verify execution outcome without accepting rejection
    assert pkg.execution_status == "COMPLETED"
    assert pkg.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    assert c4_dict["repair_package_status"] == "APPROVED"
    assert pkg.quantity_selected > 0
    assert c4_dict["n_synthetic_rows"] == pkg.quantity_selected

    # Verify flat C4 output
    assert out_c4_file.is_file()
    assert c4_dict["generator"] in ("TVAE", "CTGAN", "TabDDPM")
    assert 0.0 <= c4_dict["fidelity_score"] <= 1.0
    assert 0.0 <= c4_dict["utility_score"] <= 1.0
    assert c4_dict["privacy_status"] in ("PASS", "FAIL")

    # Validate against shared contract
    ok, errs = validate_c4_flat_contract(c4_dict)
    assert ok is True, f"C4 contract validation errors: {errs}"


def test_upstream_orchestration_contract_format(exact_adult_payload):
    """Verify that validate_three_section_contract accepts upstream orchestration wrapping and aliases."""
    upstream_payload = {
        "upstream": {
            "c1": exact_adult_payload["c1_evidence"],
            "c2_to_c3": exact_adult_payload["c2_diagnosis"],
            "context": {
                # Notice reference_dataset_uri and current_dataset_uri are omitted here,
                # but present in c1 pass-through
                "dataset_id": "adult-v1",
                "target_column": "income",
                "label_status": "AVAILABLE",
            },
        }
    }
    # Add pass-through URIs into c1
    upstream_payload["upstream"]["c1"]["model_id"] = "adult-xgb-v1"
    upstream_payload["upstream"]["c1"]["reference_dataset_uri"] = "shared/uci_adult/data/reference.csv"
    upstream_payload["upstream"]["c1"]["current_dataset_uri"] = "shared/uci_adult/data/current.csv"

    ok, errors, container = validate_three_section_contract(upstream_payload)
    assert ok is True, f"Validation failed with errors: {errors}"
    assert container is not None
    assert container.c1_evidence.predicted_accuracy_drop == 14.2
    assert container.c2_diagnosis.root_cause_feature == "education_num"
    assert container.data_context.model_id == "adult-xgb-v1"
    assert container.data_context.reference_dataset_uri == "shared/uci_adult/data/reference.csv"
    assert container.data_context.current_dataset_uri == "shared/uci_adult/data/current.csv"


def test_real_c1_output_schema_compatibility():
    """Verify contract parsing and adaptation from actual shared/contracts/c1_output_schema.json format."""
    c1_real = {
        "evidence_version": "c1-001",
        "model_id": "adult-xgb-v1",
        "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
        "current_dataset_uri": "shared/uci_adult/data/reference.csv",
        "drift_type": "marginal",
        "drift_severity": "high",
        "predicted_harm": 0.11,
        "harm_interval": [0.07, 0.15],
        "harm_metric": "accuracy_drop",
        "affected_features": ["hours_per_week", "age"],
        "detector_scores": {"ks": 0.29, "chi2": 0.05},
    }
    c2_real = {
        "root_cause_feature": "hours_per_week",
        "root_cause_category": "marginal",
        "affected_slice_rule": "age > 45 and hours_per_week < 25",
        "affected_features_ranked": ["hours_per_week", "age", "occupation"],
        "affected_value_ranges": {"age": [46, 90], "hours_per_week": [1, 24]},
        "slice_support": 0.18,
        "diagnosis_stability": 0.84,
        "drift_impact_score": 0.72,
        "temporal_window": 3,
    }
    context_real = {
        "model_id": "adult-xgb-v1",
        "dataset_id": "adult-v1",
        "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
        "current_dataset_uri": "shared/uci_adult/data/reference.csv",
        "target_column": "income",
        "schema_version": "1.0",
        "label_status": "AVAILABLE",
    }
    payload = {
        "c1_evidence": c1_real,
        "c2_diagnosis": c2_real,
        "data_context": context_real,
    }

    ok, errors, container = validate_three_section_contract(payload)
    assert ok is True, f"Contract validation failed: {errors}"
    assert container is not None
    assert container.c1_evidence.predicted_accuracy_drop == 0.11
    assert container.c1_evidence.confidence_interval == (0.07, 0.15)
    assert container.c1_evidence.predicted_harm_proportion == 0.11
    assert container.c1_evidence.is_drift_detected is True
    assert container.c2_diagnosis.root_cause_feature == "hours_per_week"


def test_numeric_boundary_rejection_nan_inf_bool():
    """Verify that boolean, NaN, Inf, and out-of-range numeric values are strictly rejected."""
    base_valid = {
        "c1_evidence": {
            "predicted_accuracy_drop": 0.15,
            "confidence_interval": [0.10, 0.20],
            "drift_type": "marginal",
            "drift_severity": "high",
            "affected_features": ["col_a", "col_b"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "col_a",
            "root_cause_category": "marginal",
            "affected_slice_rule": "col_a > 10",
            "affected_features_ranked": ["col_a", "col_b"],
            "slice_support": 0.25,
            "diagnosis_stability": 0.80,
            "drift_impact_score": 0.70,
            "temporal_window": 2,
        },
        "data_context": {
            "model_id": "telco_rf_experimental_source",
            "dataset_id": "telco_churn",
            "reference_dataset_uri": "component3/data/raw/telco_churn.csv",
            "current_dataset_uri": "component3/data/raw/telco_churn.csv",
            "target_column": "Churn",
            "adaptation_budget_size": 200,
            "source_retention_tolerance": 0.05,
        },
    }

    # 1. Boolean passed for float in c1
    bad1 = json.loads(json.dumps(base_valid))
    bad1["c1_evidence"]["predicted_accuracy_drop"] = True
    ok, errs, _ = validate_three_section_contract(bad1)
    assert ok is False
    assert any("boolean" in e for e in errs)

    # 2. Boolean passed for float in c2
    bad2 = json.loads(json.dumps(base_valid))
    bad2["c2_diagnosis"]["slice_support"] = False
    ok, errs, _ = validate_three_section_contract(bad2)
    assert ok is False
    assert any("boolean" in e for e in errs)

    # 3. Non-finite NaN passed in c1
    bad3 = json.loads(json.dumps(base_valid))
    bad3["c1_evidence"]["predicted_accuracy_drop"] = float("nan")
    ok, errs, _ = validate_three_section_contract(bad3)
    assert ok is False
    assert any("finite" in e for e in errs)

    # 4. Out of range budget (< 1)
    bad4 = json.loads(json.dumps(base_valid))
    bad4["data_context"]["adaptation_budget_size"] = 0
    ok, errs, _ = validate_three_section_contract(bad4)
    assert ok is False
    assert any("adaptation_budget_size" in e for e in errs)

    # 5. Out of range retention tolerance (> 0.50)
    bad5 = json.loads(json.dumps(base_valid))
    bad5["data_context"]["source_retention_tolerance"] = 0.80
    ok, errs, _ = validate_three_section_contract(bad5)
    assert ok is False
    assert any("source_retention_tolerance" in e for e in errs)

    # 6. Duplicate features
    bad6 = json.loads(json.dumps(base_valid))
    bad6["c1_evidence"]["affected_features"] = ["col_a", "col_a"]
    ok, errs, _ = validate_three_section_contract(bad6)
    assert ok is False
    assert any("duplicate" in e for e in errs)


def test_path_authorization_security_rejections():
    """Verify that unauthorized absolute paths and traversal attempts are rejected."""
    unauth_payload = {
        "c1_evidence": {
            "predicted_accuracy_drop": 0.15,
            "confidence_interval": [0.10, 0.20],
            "drift_type": "marginal",
            "drift_severity": "high",
            "affected_features": ["col_a"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "col_a",
            "root_cause_category": "marginal",
            "affected_slice_rule": "col_a > 10",
            "affected_features_ranked": ["col_a"],
            "slice_support": 0.25,
            "diagnosis_stability": 0.80,
            "drift_impact_score": 0.70,
        },
        "data_context": {
            "model_id": "telco_rf_experimental_source",
            "dataset_id": "telco_churn",
            "reference_dataset_uri": "C:/Windows/System32/drivers/etc/hosts",
            "current_dataset_uri": "component3/data/raw/telco_churn.csv",
            "target_column": "Churn",
        },
    }

    ok, errs, _ = validate_three_section_contract(unauth_payload)
    assert ok is False
    assert any("outside authorized" in e for e in errs)


