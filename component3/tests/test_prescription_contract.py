"""Unit tests for Component 3 Prescription Experiment Data Contract."""

import pandas as pd
import pytest

from component3.src.contracts.prescription_contract import (
    POST_GENERATION_OUTCOME_FIELDS,
    PRE_DECISION_FEATURE_WHITELIST,
    PrescriptionExperimentRecord,
    filter_leak_free_dataset,
    validate_prescription_record,
)


@pytest.fixture
def valid_contract_record_dict():
    return {
        "experiment_id": "exp_uci_adult_01",
        # Section A: Dataset & Model Metadata
        "dataset_id": "uci_adult",
        "dataset_version": "1.0.0",
        "baseline_model_family": "xgboost",
        "baseline_model_version": "1.0.0",
        "total_features_count": 12,
        "numerical_features_count": 5,
        "categorical_features_count": 7,
        "numerical_feature_ratio": 0.4167,
        "dataset_missingness_rate": 0.012,
        "class_imbalance_ratio": 0.241,
        "reference_training_rows": 24000,
        # Section B: Pre-Decision Drift Context
        "drift_scenario": "S04",
        "drift_type": "categorical_distribution_shift",
        "drift_severity": "moderate",
        "measured_drift_severity_score": 0.39,
        "affected_features": "race",
        "affected_feature_count": 1,
        "slice_support": 0.06,
        "diagnosis_stability": 0.75,
        "slice_rule": "race == 2",
        "pre_repair_overall_f1": 0.7195,
        "pre_repair_slice_f1": 0.6667,
        # Section C: Candidate Repair Configuration
        "generator": "CTGAN",
        "repair_quantity": 500,
        "epochs": 5,
        "generator_learning_rate": 0.0002,
        "seed": 42,
        # Section D: Post-Generation Outcomes
        "target_coverage": 1.0,
        "structural_status": "PASS",
        "fidelity_score": 0.784,
        "mean_ks_statistic": 0.41,
        "mean_total_variation_distance": 0.22,
        "correlation_frobenius_norm": 0.54,
        "duplicate_record_ratio": 0.0,
        "privacy_status": "PASS",
        "dcr_p05": 0.3708,
        "dcr_mean": 0.482,
        "exact_match_count": 0,
        "near_duplicate_count": 0,
        "utility_score": 0.8629,
        "delta_overall_f1": -0.012,
        "delta_affected_slice_f1": -0.018,
        "delta_non_affected_f1": -0.0117,
        "delta_baseline_slice_f1": -0.040,
        "training_seconds": 3.2,
        "sampling_seconds": 0.05,
        "validation_seconds": 0.7,
        "total_seconds": 3.95,
        "decision": "REVIEW",
        "package_status": "REJECTED",
        # Section E: Research Targets & Governance
        "prescriptive_viability_label": 1,
        "repair_improvement_label": 0,
        "label_source": "teacher_pseudo_label",
        "label_authorized": True,
        "simulation_only": True,
        "evaluation_methodology": "offline_retrained_delta",
        "experiment_provenance": "component3_automated_runner",
    }


def test_valid_record_passes_contract(valid_contract_record_dict):
    assert validate_prescription_record(valid_contract_record_dict) is True
    record_obj = PrescriptionExperimentRecord(**valid_contract_record_dict)
    assert record_obj.experiment_id == "exp_uci_adult_01"


def test_missing_pre_decision_feature_fails_contract(valid_contract_record_dict):
    corrupt_dict = valid_contract_record_dict.copy()
    del corrupt_dict["slice_support"]  # Missing required pre-decision context
    with pytest.raises(ValueError) as exc_info:
        validate_prescription_record(corrupt_dict)
    assert "slice_support" in str(exc_info.value)


def test_out_of_bounds_metric_fails_contract(valid_contract_record_dict):
    corrupt_dict = valid_contract_record_dict.copy()
    corrupt_dict["target_coverage"] = 1.25  # Impossible coverage > 1.0
    with pytest.raises(ValueError) as exc_info:
        validate_prescription_record(corrupt_dict)
    assert "target_coverage must be in [0, 1]" in str(exc_info.value)


def test_extract_pre_decision_features_prevents_leakage(valid_contract_record_dict):
    record_obj = PrescriptionExperimentRecord(**valid_contract_record_dict)
    pre_features = record_obj.extract_pre_decision_features()

    # Must contain all whitelist features
    for feat in PRE_DECISION_FEATURE_WHITELIST:
        assert feat in pre_features

    # Must NEVER contain post-generation outcome features
    for post_feat in POST_GENERATION_OUTCOME_FIELDS:
        assert post_feat not in pre_features
        assert post_feat not in pre_features.keys()


def test_filter_leak_free_dataset_removes_all_outcomes(valid_contract_record_dict):
    df = pd.DataFrame([valid_contract_record_dict])
    clean_df = filter_leak_free_dataset(df)

    # Post generation columns must be strictly absent
    for post_feat in POST_GENERATION_OUTCOME_FIELDS:
        assert post_feat not in clean_df.columns

    # Pre decision features must be retained
    assert "drift_scenario" in clean_df.columns
    assert "generator" in clean_df.columns
    assert "repair_quantity" in clean_df.columns
    assert "prescriptive_viability_label" in clean_df.columns


def test_baseline_test_f1_and_outcomes_prohibited_from_whitelist():
    """Audit 3 Regression Test: baseline_test_f1 and post-generation metrics are forbidden from whitelist."""
    from component3.src.contracts.prescription_contract import PROHIBITED_PRESCRIPTION_INPUTS

    assert "baseline_test_f1" in PROHIBITED_PRESCRIPTION_INPUTS
    assert "test_f1" in PROHIBITED_PRESCRIPTION_INPUTS
    assert "test_accuracy" in PROHIBITED_PRESCRIPTION_INPUTS

    # Ensure zero intersection with PRE_DECISION_FEATURE_WHITELIST
    intersection = set(PRE_DECISION_FEATURE_WHITELIST) & PROHIBITED_PRESCRIPTION_INPUTS
    assert len(intersection) == 0, f"Leakage detected: {intersection}"

