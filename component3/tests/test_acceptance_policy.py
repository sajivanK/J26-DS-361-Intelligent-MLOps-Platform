"""Unit tests for repair acceptance and rejection policy engine."""

import pytest

from component3.src.validation.acceptance_policy import (
    AcceptancePolicy,
    evaluate_repair_acceptance,
)
from component3.src.validation.validation_models import (
    AcceptanceDecision,
    ValidationStageResult,
    ValidationStatus,
)


@pytest.fixture
def pass_stages():
    return {
        "structural": ValidationStageResult(stage="structural", status=ValidationStatus.PASS),
        "fidelity": ValidationStageResult(stage="fidelity", status=ValidationStatus.PASS),
        "privacy": ValidationStageResult(stage="privacy", status=ValidationStatus.PASS),
        "utility": ValidationStageResult(stage="utility", status=ValidationStatus.PASS),
    }


def test_all_stages_pass_accepts(pass_stages):
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy=pass_stages["privacy"],
        utility=pass_stages["utility"],
    )
    assert res.decision == AcceptanceDecision.ACCEPT
    assert res.package_status == "APPROVED"
    assert res.supervised_retraining_approved is True
    assert len(res.reasons) == 0


def test_structural_fail_rejects(pass_stages):
    fail_structural = ValidationStageResult(
        stage="structural",
        status=ValidationStatus.FAIL,
        reasons=("Target coverage below threshold",),
    )
    res = evaluate_repair_acceptance(
        structural=fail_structural,
        fidelity=pass_stages["fidelity"],
        privacy=pass_stages["privacy"],
        utility=pass_stages["utility"],
    )
    assert res.decision == AcceptanceDecision.REJECT
    assert res.package_status == "REJECTED"
    assert any("Structural validation failed" in r for r in res.reasons)


def test_privacy_fail_rejects(pass_stages):
    fail_privacy = ValidationStageResult(
        stage="privacy",
        status=ValidationStatus.FAIL,
        reasons=("Exact match detected",),
    )
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy=fail_privacy,
        utility=pass_stages["utility"],
    )
    assert res.decision == AcceptanceDecision.REJECT
    assert res.package_status == "REJECTED"
    assert any("Privacy validation failed" in r for r in res.reasons)


def test_fidelity_fail_rejects_in_production(pass_stages):
    fail_fidelity = ValidationStageResult(
        stage="fidelity",
        status=ValidationStatus.FAIL,
        reasons=("KS statistic exceeded",),
    )
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=fail_fidelity,
        privacy=pass_stages["privacy"],
        utility=pass_stages["utility"],
        mode="production",
    )
    assert res.decision == AcceptanceDecision.REJECT
    assert res.package_status == "REJECTED"


def test_fidelity_fail_reviews_in_exploratory(pass_stages):
    fail_fidelity = ValidationStageResult(
        stage="fidelity",
        status=ValidationStatus.FAIL,
        reasons=("KS statistic exceeded",),
    )
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=fail_fidelity,
        privacy=pass_stages["privacy"],
        utility=pass_stages["utility"],
        mode="exploratory",
    )
    assert res.decision == AcceptanceDecision.REVIEW
    assert res.package_status == "REJECTED"


def test_utility_fail_rejects_in_production(pass_stages):
    fail_utility = ValidationStageResult(
        stage="utility",
        status=ValidationStatus.FAIL,
        reasons=("Negative transfer detected",),
    )
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy=pass_stages["privacy"],
        utility=fail_utility,
        mode="production",
    )
    assert res.decision == AcceptanceDecision.REJECT
    assert res.package_status == "REJECTED"


def test_utility_not_evaluated_results_in_review(pass_stages):
    not_eval_utility = ValidationStageResult(
        stage="utility",
        status=ValidationStatus.NOT_EVALUATED,
        reasons=("Labels unauthorized",),
    )
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy=pass_stages["privacy"],
        utility=not_eval_utility,
    )
    assert res.decision == AcceptanceDecision.REVIEW
    assert res.package_status == "REJECTED"
    assert res.supervised_retraining_approved is False


def test_missing_stages_prevent_accept(pass_stages):
    # Only structural passed; other stages None
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
    )
    assert res.decision == AcceptanceDecision.REVIEW
    assert res.package_status == "REJECTED"
    assert any("Privacy validation has not been performed" in r for r in res.reasons)


def test_malformed_structural_input_rejects():
    # Pass invalid object
    res = evaluate_repair_acceptance(structural=None)  # type: ignore
    assert res.decision == AcceptanceDecision.REJECT
    assert res.package_status == "REJECTED"


def test_oop_interface_and_invalid_modes(pass_stages):
    policy = AcceptancePolicy(mode="production")
    res = policy.decide(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy=pass_stages["privacy"],
        utility=pass_stages["utility"],
    )
    assert res.decision == AcceptanceDecision.ACCEPT
    assert isinstance(res.to_dict(), dict)

    with pytest.raises(ValueError):
        AcceptancePolicy(mode="invalid_mode")


def test_malformed_fidelity_stage_rejects(pass_stages):
    res = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity="invalid_string_stage",  # type: ignore
        privacy=pass_stages["privacy"],
        utility=pass_stages["utility"],
    )
    assert res.decision == AcceptanceDecision.REJECT
    assert res.package_status == "REJECTED"
    assert any("Fidelity validation result is malformed" in r for r in res.reasons)


def test_malformed_privacy_and_utility_stages_reject(pass_stages):
    res_priv = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy={"status": "PASS"},  # type: ignore
        utility=pass_stages["utility"],
    )
    assert res_priv.decision == AcceptanceDecision.REJECT
    assert any("Privacy validation result is malformed" in r for r in res_priv.reasons)

    res_util = evaluate_repair_acceptance(
        structural=pass_stages["structural"],
        fidelity=pass_stages["fidelity"],
        privacy=pass_stages["privacy"],
        utility=12345,  # type: ignore
    )
    assert res_util.decision == AcceptanceDecision.REJECT
    assert any("Utility validation result is malformed" in r for r in res_util.reasons)

