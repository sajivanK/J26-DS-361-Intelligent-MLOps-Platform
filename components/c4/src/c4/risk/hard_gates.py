from typing import Any

from c4.contracts import UpstreamInput


GateResults = dict[str, dict[str, Any]]


def run_hard_gates(
    upstream: UpstreamInput,
    *,
    schema_compatible: bool = True,
    rollback_ready: bool = True,
) -> tuple[bool, GateResults]:
    """
    Evaluate mandatory deterministic release gates.

    Every gate must pass before C4 can continue to candidate retraining,
    forecasting, conformal calibration, and release-policy evaluation.

    During independent development, schema compatibility and rollback
    readiness default to True. During integration, these values must come
    from real runtime checks.
    """
    gates: GateResults = {
        "repair_approved": {
            "value": upstream.c3_repair_package.repair_package_status,
            "passed": (
                upstream.c3_repair_package.repair_package_status == "APPROVED"
            ),
        },
        "privacy_check": {
            "value": upstream.c3_repair_package.privacy_status,
            "passed": upstream.c3_repair_package.privacy_status == "PASS",
        },
        "diagnosis_stability": {
            "value": upstream.c2_diagnosis.diagnosis_stability,
            "passed": upstream.c2_diagnosis.diagnosis_stability >= 0.50,
        },
        "schema_compatible": {
            "value": schema_compatible,
            "passed": schema_compatible,
        },
        "rollback_ready": {
            "value": rollback_ready,
            "passed": rollback_ready,
        },
    }

    all_passed = all(bool(result["passed"]) for result in gates.values())

    gates["overall"] = {
        "value": "PASS" if all_passed else "FAIL",
        "passed": all_passed,
    }

    return all_passed, gates