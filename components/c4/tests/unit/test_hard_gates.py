import json
from pathlib import Path
from typing import Any

import pytest

from c4.contracts import UpstreamInput
from c4.risk import run_hard_gates


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "valid_upstream_input.json"
)


def load_payload() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_upstream() -> UpstreamInput:
    return UpstreamInput.model_validate(load_payload())


def test_all_hard_gates_pass_for_valid_input() -> None:
    upstream = load_upstream()

    all_passed, gates = run_hard_gates(upstream)

    assert all_passed is True
    assert gates["repair_approved"]["passed"] is True
    assert gates["privacy_check"]["passed"] is True
    assert gates["diagnosis_stability"]["passed"] is True
    assert gates["schema_compatible"]["passed"] is True
    assert gates["rollback_ready"]["passed"] is True
    assert gates["overall"] == {
        "value": "PASS",
        "passed": True,
    }


@pytest.mark.parametrize(
    ("section", "field", "value", "failed_gate"),
    [
        (
            "c3_repair_package",
            "repair_package_status",
            "REJECTED",
            "repair_approved",
        ),
        (
            "c3_repair_package",
            "privacy_status",
            "FAIL",
            "privacy_check",
        ),
        (
            "c2_diagnosis",
            "diagnosis_stability",
            0.49,
            "diagnosis_stability",
        ),
    ],
)
def test_failed_upstream_gate_causes_overall_failure(
    section: str,
    field: str,
    value: Any,
    failed_gate: str,
) -> None:
    payload = load_payload()
    payload[section][field] = value
    upstream = UpstreamInput.model_validate(payload)

    all_passed, gates = run_hard_gates(upstream)

    assert all_passed is False
    assert gates[failed_gate]["passed"] is False
    assert gates["overall"] == {
        "value": "FAIL",
        "passed": False,
    }


def test_diagnosis_stability_boundary_passes() -> None:
    payload = load_payload()
    payload["c2_diagnosis"]["diagnosis_stability"] = 0.50
    upstream = UpstreamInput.model_validate(payload)

    all_passed, gates = run_hard_gates(upstream)

    assert all_passed is True
    assert gates["diagnosis_stability"]["passed"] is True


@pytest.mark.parametrize(
    ("schema_compatible", "rollback_ready", "failed_gate"),
    [
        (False, True, "schema_compatible"),
        (True, False, "rollback_ready"),
    ],
)
def test_failed_runtime_gate_causes_overall_failure(
    schema_compatible: bool,
    rollback_ready: bool,
    failed_gate: str,
) -> None:
    upstream = load_upstream()

    all_passed, gates = run_hard_gates(
        upstream,
        schema_compatible=schema_compatible,
        rollback_ready=rollback_ready,
    )

    assert all_passed is False
    assert gates[failed_gate]["passed"] is False
    assert gates["overall"] == {
        "value": "FAIL",
        "passed": False,
    }