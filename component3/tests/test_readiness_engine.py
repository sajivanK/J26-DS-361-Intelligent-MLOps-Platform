"""Permanent tests for the Component 3 Repair Readiness Engine."""

import json
import pytest

from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import (
    load_reference_dataset,
    load_current_dataset,
    load_c1_evidence,
    load_c2_diagnosis,
)
from component3.src.readiness.readiness_engine import (
    assess_repair_readiness,
)
from component3.src.readiness.readiness_models import (
    ReadinessDecision,
    ReadinessReason,
    ReadinessStatus,
)


@pytest.fixture(scope="module")
def readiness_resources():
    """Load shared configuration and reference data once."""
    config = load_config()
    reference = load_reference_dataset(config)
    return {
        "config": config,
        "reference": reference,
    }


@pytest.mark.parametrize("scenario", [f"S{i:02d}" for i in range(8)])
def test_readiness_across_drift_scenarios(readiness_resources, scenario):
    """Verify readiness decision consistency across S00-S07 scenarios."""
    config = readiness_resources["config"]
    reference = readiness_resources["reference"]
    current = load_current_dataset(config, scenario)
    c1 = load_c1_evidence(config, scenario)
    c2 = load_c2_diagnosis(config, scenario)

    decision = assess_repair_readiness(
        c1_payload=c1,
        c2_payload=c2,
        reference_df=reference,
        current_df=current,
        config=config,
        label_provenance=None,
    )

    assert isinstance(decision, ReadinessDecision)
    assert decision.status in ReadinessStatus
    assert decision.supervised_retraining_ready is False
    assert 0 <= decision.affected_slice_rows <= decision.current_rows

    if scenario == "S00":
        assert decision.status == ReadinessStatus.BLOCKED
        assert ReadinessReason.NO_DRIFT_DETECTED in decision.reason_codes
        assert decision.generation_ready is False
    else:
        assert decision.status == ReadinessStatus.REVIEW
        assert ReadinessReason.LABEL_PROVENANCE_UNKNOWN in decision.reason_codes
        assert decision.generation_ready is True


def test_readiness_decision_json_serialization(readiness_resources):
    """Verify that ReadinessDecision serializes cleanly to JSON."""
    config = readiness_resources["config"]
    reference = readiness_resources["reference"]
    current = load_current_dataset(config, "S05")
    c1 = load_c1_evidence(config, "S05")
    c2 = load_c2_diagnosis(config, "S05")

    decision = assess_repair_readiness(
        c1_payload=c1,
        c2_payload=c2,
        reference_df=reference,
        current_df=current,
        config=config,
        label_provenance=None,
    )

    data = decision.to_dict()
    assert isinstance(data, dict)
    assert data["status"] == "REVIEW"
    assert "LABEL_PROVENANCE_UNKNOWN" in data["reason_codes"]
    assert data["generation_ready"] is True
    assert data["supervised_retraining_ready"] is False
    assert data["reference_rows"] == 26048
    assert data["current_rows"] == 6000
    assert data["affected_slice_rows"] == 342
    assert data["diagnosis_stability"] == 0.88

    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    assert deserialized == data


from copy import deepcopy
from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule


@pytest.fixture(scope="module")
def s05_resources(readiness_resources):
    """Reusable S05 evidence for negative tests."""
    config = readiness_resources["config"]
    return {
        "config": config,
        "reference": readiness_resources["reference"],
        "current": load_current_dataset(config, "S05"),
        "c1": load_c1_evidence(config, "S05"),
        "c2": load_c2_diagnosis(config, "S05"),
    }


@pytest.mark.parametrize(
    "case_name, expected_status, expected_reason",
    [
        (
            "missing_config",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "invalid_c1",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "contradictory_support",
            ReadinessStatus.BLOCKED,
            ReadinessReason.CROSS_COMPONENT_INCONSISTENCY,
        ),
        (
            "small_reference",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INSUFFICIENT_REFERENCE_ROWS,
        ),
        (
            "small_current",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INSUFFICIENT_CURRENT_ROWS,
        ),
        (
            "insufficient_slice",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INSUFFICIENT_AFFECTED_SLICE,
        ),
        (
            "low_stability",
            ReadinessStatus.REVIEW,
            ReadinessReason.LOW_DIAGNOSIS_STABILITY,
        ),
    ],
)
def test_readiness_failure_conditions(
    s05_resources, case_name, expected_status, expected_reason
):
    """Verify that defective or insufficient inputs are safely blocked or reviewed."""
    config = s05_resources["config"]
    reference = s05_resources["reference"]
    current = s05_resources["current"]
    c1 = s05_resources["c1"]
    c2 = s05_resources["c2"]

    test_config = config
    test_reference = reference
    test_current = current
    test_c1 = c1
    test_c2 = c2

    if case_name == "missing_config":
        test_config = deepcopy(config)
        test_config.pop("readiness")
    elif case_name == "invalid_c1":
        test_c1 = deepcopy(c1)
        test_c1["drift_type"] = "invalid_drift_type"
    elif case_name == "contradictory_support":
        test_c2 = deepcopy(c2)
        test_c2["slice_support"] = 0.99
    elif case_name == "small_reference":
        test_reference = reference.head(100).copy()
    elif case_name == "small_current":
        test_current = current.head(100).copy()
        test_c2 = deepcopy(c2)
        mask = evaluate_slice_rule(test_current, test_c2["affected_slice_rule"])
        test_c2["slice_support"] = float(mask.mean())
    elif case_name == "insufficient_slice":
        test_config = deepcopy(config)
        test_config["readiness"]["min_affected_slice_rows"] = 1000
    elif case_name == "low_stability":
        test_config = deepcopy(config)
        test_config["readiness"]["min_diagnosis_stability"] = 0.95

    decision = assess_repair_readiness(
        c1_payload=test_c1,
        c2_payload=test_c2,
        reference_df=test_reference,
        current_df=test_current,
        config=test_config,
        label_provenance=None,
    )

    assert decision.status == expected_status
    assert expected_reason in decision.reason_codes
    assert decision.generation_ready is False
    assert decision.supervised_retraining_ready is False


@pytest.mark.parametrize(
    "setting_name, invalid_value",
    [
        ("min_reference_rows", -500),
        ("min_current_rows", True),
        ("min_affected_slice_rows", "50"),
        ("min_affected_slice_rows", 0),
        ("min_diagnosis_stability", 1.5),
        ("min_diagnosis_stability", -0.1),
        ("min_diagnosis_stability", float("nan")),
        ("min_diagnosis_stability", float("inf")),
        ("require_trusted_labels", "False"),
        ("allow_unlabeled_current_data", "False"),
    ],
)
def test_invalid_readiness_configuration(
    s05_resources,
    setting_name,
    invalid_value,
):
    """Invalid readiness settings must fail safely."""
    config = deepcopy(s05_resources["config"])
    config["readiness"][setting_name] = invalid_value
    decision = assess_repair_readiness(
        c1_payload=deepcopy(s05_resources["c1"]),
        c2_payload=deepcopy(s05_resources["c2"]),
        reference_df=s05_resources["reference"],
        current_df=s05_resources["current"],
        config=config,
        label_provenance=None,
    )
    assert decision.status == ReadinessStatus.BLOCKED
    assert (
        ReadinessReason.INVALID_INPUT_CONTRACT
        in decision.reason_codes
    )
    assert decision.generation_ready is False
    assert decision.supervised_retraining_ready is False
    assert len(decision.warnings) > 0


@pytest.mark.parametrize(
    "case_name, expected_status, expected_reason",
    [
        (
            "missing_c1",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "missing_c2",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "invalid_c1_type",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "missing_slice_rule",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "unsafe_slice_rule",
            ReadinessStatus.BLOCKED,
            ReadinessReason.INVALID_INPUT_CONTRACT,
        ),
        (
            "category_mismatch_warning",
            ReadinessStatus.REVIEW,
            ReadinessReason.CROSS_COMPONENT_INCONSISTENCY,
        ),
    ],
)
def test_malformed_inputs_and_warnings(
    s05_resources,
    case_name,
    expected_status,
    expected_reason,
):
    """Invalid evidence and unresolved warnings must fail safely."""
    config = deepcopy(s05_resources["config"])
    c1 = deepcopy(s05_resources["c1"])
    c2 = deepcopy(s05_resources["c2"])
    if case_name == "missing_c1":
        c1 = None
    elif case_name == "missing_c2":
        c2 = None
    elif case_name == "invalid_c1_type":
        c1 = []
    elif case_name == "missing_slice_rule":
        c2.pop("affected_slice_rule")
    elif case_name == "unsafe_slice_rule":
        c2["affected_slice_rule"] = (
            "__import__('os').system('echo unsafe')"
        )
    elif case_name == "category_mismatch_warning":
        c2["root_cause_category"] = "interaction"
    decision = assess_repair_readiness(
        c1_payload=c1,
        c2_payload=c2,
        reference_df=s05_resources["reference"],
        current_df=s05_resources["current"],
        config=config,
        label_provenance=None,
    )
    assert decision.status == expected_status
    assert expected_reason in decision.reason_codes
    assert decision.generation_ready is False
    assert decision.supervised_retraining_ready is False
    if case_name == "category_mismatch_warning":
        assert len(decision.warnings) > 0


@pytest.mark.parametrize(
    "case_name, provenance, allow_unlabeled, expected_status, expected_generation, expected_retraining",
    [
        (
            "missing_provenance",
            None,
            True,
            ReadinessStatus.REVIEW,
            True,
            False,
        ),
        (
            "labels_unavailable",
            {"labels_available": False},
            True,
            ReadinessStatus.REVIEW,
            True,
            False,
        ),
        (
            "unverified_labels",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": False,
            },
            True,
            ReadinessStatus.REVIEW,
            True,
            False,
        ),
        (
            "untrusted_labels",
            {
                "labels_available": True,
                "trusted": False,
                "source_verified": True,
            },
            True,
            ReadinessStatus.REVIEW,
            True,
            False,
        ),
        (
            "verified_trusted_labels",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": True,
            },
            True,
            ReadinessStatus.READY,
            True,
            True,
        ),
        (
            "unlabeled_data_disallowed",
            None,
            False,
            ReadinessStatus.BLOCKED,
            False,
            False,
        ),
        (
            "verified_labels_unlabeled_disallowed",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": True,
            },
            False,
            ReadinessStatus.READY,
            True,
            True,
        ),
    ],
)
def test_label_provenance_safety(
    s05_resources,
    case_name,
    provenance,
    allow_unlabeled,
    expected_status,
    expected_generation,
    expected_retraining,
):
    """Supervised retraining requires explicitly verified label provenance."""
    config = deepcopy(s05_resources["config"])
    config["readiness"]["allow_unlabeled_current_data"] = allow_unlabeled
    decision = assess_repair_readiness(
        c1_payload=deepcopy(s05_resources["c1"]),
        c2_payload=deepcopy(s05_resources["c2"]),
        reference_df=s05_resources["reference"],
        current_df=s05_resources["current"],
        config=config,
        label_provenance=deepcopy(provenance),
    )
    assert decision.status == expected_status, case_name
    assert decision.generation_ready is expected_generation, case_name
    assert (
        decision.supervised_retraining_ready is expected_retraining
    ), case_name


# =============================================================================
# Sprint 03 — Task 07: Permanent Failure-Handling & Decision Priority Tests
# =============================================================================

from unittest.mock import patch


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Simulated internal failure"),
        AttributeError("Simulated unexpected attribute error"),
        TypeError("Simulated validator type error"),
    ],
)
def test_cross_component_unexpected_exception_fails_closed(
    s05_resources, failure
):
    """Unexpected validator exceptions must return BLOCKED."""
    with patch(
        "component3.src.readiness.readiness_engine."
        "validate_cross_component_consistency",
        side_effect=failure,
    ):
        result = assess_repair_readiness(
            c1_payload=deepcopy(s05_resources["c1"]),
            c2_payload=deepcopy(s05_resources["c2"]),
            reference_df=s05_resources["reference"],
            current_df=s05_resources["current"],
            config=deepcopy(s05_resources["config"]),
            label_provenance=None,
        )
    assert result.status == ReadinessStatus.BLOCKED
    assert ReadinessReason.CROSS_COMPONENT_INCONSISTENCY in result.reason_codes
    assert result.generation_ready is False
    assert result.supervised_retraining_ready is False
    assert len(result.warnings) > 0


@pytest.mark.parametrize(
    "malformed_result",
    [
        None,
        [],
        {},
        {"valid": "True"},
        {"valid": 1},
        {"valid": True, "errors": "not a list"},
        {"valid": True, "warnings": "not a list"},
        {"valid": True, "warnings": [123]},
        {"valid": False, "errors": [None]},
    ],
)
def test_malformed_consistency_result_fails_closed(
    s05_resources, malformed_result
):
    """Malformed validator responses must never authorize repair."""
    with patch(
        "component3.src.readiness.readiness_engine."
        "validate_cross_component_consistency",
        return_value=malformed_result,
    ):
        result = assess_repair_readiness(
            c1_payload=deepcopy(s05_resources["c1"]),
            c2_payload=deepcopy(s05_resources["c2"]),
            reference_df=s05_resources["reference"],
            current_df=s05_resources["current"],
            config=deepcopy(s05_resources["config"]),
            label_provenance=None,
        )
    assert result.status == ReadinessStatus.BLOCKED
    assert ReadinessReason.CROSS_COMPONENT_INCONSISTENCY in result.reason_codes
    assert result.generation_ready is False
    assert result.supervised_retraining_ready is False
    assert len(result.warnings) > 0


def test_critical_failure_takes_priority_over_review(s05_resources):
    """Critical row insufficiency must take priority and not downgrade to REVIEW."""
    bad_config = deepcopy(s05_resources["config"])
    bad_config["readiness"]["min_current_rows"] = 7000
    bad_config["readiness"]["min_diagnosis_stability"] = 0.95

    decision = assess_repair_readiness(
        c1_payload=deepcopy(s05_resources["c1"]),
        c2_payload=deepcopy(s05_resources["c2"]),
        reference_df=s05_resources["reference"],
        current_df=s05_resources["current"],
        config=bad_config,
        label_provenance=None,
    )

    assert decision.status == ReadinessStatus.BLOCKED
    assert ReadinessReason.INSUFFICIENT_CURRENT_ROWS in decision.reason_codes
    assert decision.generation_ready is False
    assert decision.supervised_retraining_ready is False


def test_warning_triggered_review_blocks_autonomous_repair(s05_resources):
    """Cross-component consistency warnings must force REVIEW and block repair."""
    with patch(
        "component3.src.readiness.readiness_engine."
        "validate_cross_component_consistency",
        return_value={
            "valid": True,
            "errors": [],
            "warnings": ["Advisory: drift pattern may be ambiguous"],
        },
    ):
        result = assess_repair_readiness(
            c1_payload=deepcopy(s05_resources["c1"]),
            c2_payload=deepcopy(s05_resources["c2"]),
            reference_df=s05_resources["reference"],
            current_df=s05_resources["current"],
            config=deepcopy(s05_resources["config"]),
            label_provenance={
                "labels_available": True,
                "trusted": True,
                "source_verified": True,
            },
        )
    assert result.status == ReadinessStatus.REVIEW
    assert ReadinessReason.CROSS_COMPONENT_INCONSISTENCY in result.reason_codes
    assert result.generation_ready is False
    assert result.supervised_retraining_ready is False
    assert len(result.warnings) > 0
