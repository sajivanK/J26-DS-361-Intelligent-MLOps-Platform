"""End-to-end integration tests for Component 3 core modules."""

import json
from copy import deepcopy
import pytest
from pandas.testing import assert_frame_equal

from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import (
    load_reference_dataset,
    load_current_dataset,
    load_c1_evidence,
    load_c2_diagnosis,
    load_integration_contracts,
)
from component3.src.contracts.input_validator import (
    validate_c1_input,
    validate_c2_input,
    validate_cross_component_consistency,
)
from component3.src.contracts.safe_slice_evaluator import extract_affected_slice
from component3.src.readiness.readiness_engine import assess_repair_readiness
from component3.src.readiness.readiness_models import (
    ReadinessDecision,
    ReadinessReason,
    ReadinessStatus,
)


@pytest.fixture(scope="module")
def integration_resources():
    """Load configuration, reference data, and contracts once."""
    config = load_config()
    reference = load_reference_dataset(config)
    c1_contract, c2_contract = load_integration_contracts(config)
    return {
        "config": config,
        "reference": reference,
        "c1_contract": c1_contract,
        "c2_contract": c2_contract,
    }


@pytest.mark.parametrize("scenario_id", [f"S{i:02d}" for i in range(8)])
def test_end_to_end_readiness_pipeline(integration_resources, scenario_id):
    """Verify complete end-to-end integration workflow across S00-S07."""
    resources = integration_resources
    config = resources["config"]
    reference = resources["reference"]

    current = load_current_dataset(config, scenario_id)
    c1 = load_c1_evidence(config, scenario_id)
    c2 = load_c2_diagnosis(config, scenario_id)

    reference_before = reference.copy(deep=True)
    current_before = current.copy(deep=True)
    c1_before = deepcopy(c1)
    c2_before = deepcopy(c2)

    # 1. Validate official input contracts.
    c1_result = validate_c1_input(c1, resources["c1_contract"])
    c2_result = validate_c2_input(
        c2, resources["c2_contract"], reference
    )
    assert c1_result["valid"] is True
    assert c2_result["valid"] is True

    # 2. Check cross-component consistency.
    consistency = validate_cross_component_consistency(
        c1, c2, current
    )
    assert consistency["valid"] is True

    # 3. Assess readiness using the same loaded inputs.
    decision = assess_repair_readiness(
        c1_payload=c1,
        c2_payload=c2,
        reference_df=reference,
        current_df=current,
        config=config,
        label_provenance=None,
        c1_contract=resources["c1_contract"],
        c2_contract=resources["c2_contract"],
    )

    # 4. Verify decision serialization.
    serialized = decision.to_dict()
    json.dumps(serialized)
    assert serialized["status"] == decision.status.value
    assert serialized["reference_rows"] == len(reference)
    assert serialized["current_rows"] == len(current)

    # 5. Extract the affected slice independently.
    affected_slice = extract_affected_slice(
        current, c2["affected_slice_rule"]
    )
    assert decision.affected_slice_rows == len(affected_slice)

    # 6. Verify safety decisions.
    if scenario_id == "S00":
        assert decision.status == ReadinessStatus.BLOCKED
        assert ReadinessReason.NO_DRIFT_DETECTED in decision.reason_codes
        assert len(affected_slice) == 0
        assert decision.generation_ready is False
    else:
        assert decision.status == ReadinessStatus.REVIEW
        assert ReadinessReason.LABEL_PROVENANCE_UNKNOWN in decision.reason_codes
        assert decision.generation_ready is True

    # Missing verified provenance must never enable supervised retraining.
    assert decision.supervised_retraining_ready is False

    # 7. Verify inputs were not modified.
    assert_frame_equal(reference, reference_before)
    assert_frame_equal(current, current_before)
    assert c1 == c1_before
    assert c2 == c2_before


from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule


# Sprint 03 Task 08 — Negative core integration scenarios
@pytest.fixture(scope="module")
def negative_integration_resources():
    """Load real saved S05 resources for negative integration tests."""
    config = load_config()
    return {
        "config": config,
        "reference": load_reference_dataset(config),
        "current": load_current_dataset(config, "S05"),
        "c1": load_c1_evidence(config, "S05"),
        "c2": load_c2_diagnosis(config, "S05"),
    }


@pytest.mark.parametrize(
    "failure_case, expected_reason",
    [
        ("missing_c1_field", ReadinessReason.INVALID_INPUT_CONTRACT),
        ("missing_c2_slice_rule", ReadinessReason.INVALID_INPUT_CONTRACT),
        ("incorrect_slice_support", ReadinessReason.CROSS_COMPONENT_INCONSISTENCY),
        ("unsafe_slice_expression", ReadinessReason.INVALID_INPUT_CONTRACT),
        ("insufficient_current_data", ReadinessReason.INSUFFICIENT_CURRENT_ROWS),
        ("invalid_readiness_config", ReadinessReason.INVALID_INPUT_CONTRACT),
    ],
)
def test_invalid_inputs_fail_closed(
    negative_integration_resources, failure_case, expected_reason
):
    """Invalid inputs must not authorize repair or mutate source data."""
    resources = negative_integration_resources
    config = deepcopy(resources["config"])
    reference = resources["reference"].copy(deep=True)
    current = resources["current"].copy(deep=True)
    c1 = deepcopy(resources["c1"])
    c2 = deepcopy(resources["c2"])

    if failure_case == "missing_c1_field":
        c1.pop("drift_type")
    elif failure_case == "missing_c2_slice_rule":
        c2.pop("affected_slice_rule")
    elif failure_case == "incorrect_slice_support":
        c2["slice_support"] = 0.99
    elif failure_case == "unsafe_slice_expression":
        c2["affected_slice_rule"] = "__import__('os').system('echo unsafe')"
    elif failure_case == "insufficient_current_data":
        current = current.head(100).copy()
        mask = evaluate_slice_rule(current, c2["affected_slice_rule"])
        c2["slice_support"] = float(mask.mean())
    elif failure_case == "invalid_readiness_config":
        config["readiness"]["min_current_rows"] = -1

    # Capture the exact inputs the readiness engine will receive.
    reference_before = reference.copy(deep=True)
    current_before = current.copy(deep=True)
    c1_before = deepcopy(c1)
    c2_before = deepcopy(c2)

    # Now execute the readiness engine.

    decision = assess_repair_readiness(
        c1_payload=c1,
        c2_payload=c2,
        reference_df=reference,
        current_df=current,
        config=config,
        label_provenance=None,
    )

    assert decision.status == ReadinessStatus.BLOCKED
    assert expected_reason in decision.reason_codes
    assert decision.generation_ready is False
    assert decision.supervised_retraining_ready is False

    # Verify original datasets and payloads remain untouched
    assert_frame_equal(reference, reference_before)
    assert_frame_equal(current, current_before)
    assert c1 == c1_before
    assert c2 == c2_before

