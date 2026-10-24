"""Live upstream gate assessment. No training, forecasts or release approval."""
from __future__ import annotations

from typing import Any

from c4.contracts.platform_adapter import adapt_platform_upstream


def assess_pipeline_gates(upstream: dict[str, Any]) -> dict[str, Any]:
    adapted = adapt_platform_upstream(upstream)
    repair = adapted.inputs.c3_repair_package
    diagnosis = adapted.inputs.c2_diagnosis

    checks = {
        "repair_approved": repair.repair_package_status == "APPROVED",
        "privacy_check": repair.privacy_status == "PASS",
        "diagnosis_stability": diagnosis.diagnosis_stability >= 0.5,
    }
    failures = [name for name, passed in checks.items() if not passed]

    return {
        "run_id": adapted.context.run_id,
        "status": (
            "UPSTREAM_GATES_FAILED"
            if failures else "RUNTIME_CHECKS_REQUIRED"
        ),
        "decision": "ESCALATE" if failures else None,
        "reason": (
            "Mandatory upstream checks failed: " + ", ".join(failures)
            if failures else
            "Upstream checks passed. Candidate, schema and rollback "
            "verification are still required."
        ),
        "upstream_checks": checks,
        "failed_checks": failures,
        "runtime_checks": {
            "schema_compatible": None,
            "candidate_artifact_valid": None,
            "rollback_ready": None,
        },
        "forecast_status": "NOT_EVALUATED",
        "N1_executed": False,
        "N2_executed": False,
        "production_release_authorized": False,
        "audit_envelope": adapted.audit_envelope,
    }
