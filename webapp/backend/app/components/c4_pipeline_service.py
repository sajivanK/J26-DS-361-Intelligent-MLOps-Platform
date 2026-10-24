"""Live Layer A decisions. No candidate training or forecast execution."""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


class PipelineUnavailable(RuntimeError):
    pass


class PipelineInputError(ValueError):
    pass


POLICY_VERSION = "c4-live-upstream-gates-v1"
MARGINS = ("global_margin", "affected_slice_margin", "non_affected_margin")
EVIDENCE = (
    "validation_global_f1_gain",
    "validation_affected_slice_recovery",
    "validation_non_affected_margin",
)


def _number(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
    )


def evaluate_pipeline(upstream):
    started = perf_counter()
    if not isinstance(upstream, dict):
        raise PipelineInputError("Expected an upstream JSON object.")

    try:
        json.dumps(upstream, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise PipelineInputError("Upstream must contain finite JSON values.") from exc

    for name in ("context", "c1", "c2_to_c4", "c3"):
        if not isinstance(upstream.get(name), dict):
            raise PipelineInputError(f"Missing upstream object: {name}.")

    context = upstream["context"]
    c1 = upstream["c1"]
    c2 = upstream["c2_to_c4"]
    c3 = upstream["c3"]

    champion = c1.get("model_id")
    run_id = context.get("run_id")
    if not isinstance(champion, str) or not champion.strip():
        raise PipelineInputError("C1 model_id is required.")
    if not isinstance(run_id, str) or not run_id.strip():
        raise PipelineInputError("Context run_id is required.")
    if context.get("model_id") != champion:
        raise PipelineInputError("Context and C1 model IDs differ.")

    package = c3.get("repair_package_status")
    privacy = c3.get("privacy_status")
    generator = c3.get("generator")
    rows = c3.get("n_synthetic_rows")
    stability = c2.get("diagnosis_stability")

    if package not in ("APPROVED", "REJECTED"):
        raise PipelineInputError("Invalid C3 repair_package_status.")
    if privacy not in ("PASS", "FAIL"):
        raise PipelineInputError("Invalid C3 privacy_status.")
    if generator not in ("CTGAN", "TVAE", "TabDDPM", "NONE"):
        raise PipelineInputError("Invalid C3 generator.")
    if type(rows) is not int or rows < 0:
        raise PipelineInputError("C3 n_synthetic_rows must be a nonnegative integer.")

    failures = []
    if package == "REJECTED":
        failures.append("C3 repair package REJECTED")
    if privacy == "FAIL":
        failures.append("C3 privacy check FAIL")

    # An independently verified rejection is sufficient to stop.
    # Missing diagnosis evidence is not silently converted into a numeric score.
    if stability is None:
        if not failures:
            raise PipelineInputError("Diagnosis stability is required to continue.")
    elif not _number(stability) or not 0 <= stability <= 1:
        raise PipelineInputError("Invalid diagnosis_stability.")
    elif stability < 0.5:
        failures.append("Diagnosis stability below 0.5")

    if not failures:
        raise PipelineUnavailable(
            "Upstream checks passed, but live candidate evaluation and runtime "
            "verification are not implemented yet. No release decision was produced."
        )

    audit_id = "c4-live-" + uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "decision": "ESCALATE",
        "evaluation_status": "UPSTREAM_GATE_FAILED",
        "champion_model_id": champion,
        "candidate_model_id": None,
        # Configured target only; no coverage was evaluated for this request.
        "joint_confidence_level": 0.95,
        "future_safety_margins": dict.fromkeys(MARGINS),
        "future_safety_intervals": dict.fromkeys(MARGINS),
        "pre_release_evidence": dict.fromkeys(EVIDENCE),
        "upstream_evidence_summary": {
            "c1": deepcopy(c1),
            "c2": deepcopy(c2),
            "c3": deepcopy(c3),
        },
        "hard_gate_status": {
            "repair_approved": package == "APPROVED",
            "privacy_check": privacy,
            "schema_compatible": None,
            "candidate_artifact_valid": None,
            "rollback_ready": None,
            "overall": "FAIL",
        },
        # Optional calibrated policy fields are omitted because no risk policy
        # was applied. The explicit status prevents interpreting this as calibration.
        "risk_control": {
            "policy_version": POLICY_VERSION,
            "status": "NOT_APPLIED_UPSTREAM_GATE_FAILED",
        },
        "reason": "; ".join(failures) + (
            ". Candidate evaluation and forecasting were not run."
        ),
        "required_action": (
            "Keep the champion. Review upstream rejection evidence and repair "
            "prerequisites before retrying. Do not deploy a candidate."
        ),
        "audit_id": audit_id,
        "policy_version": POLICY_VERSION,
        "forecast_model_version": None,
        "conformal_config_version": None,
        "mlflow_run_id": None,
        "decision_timestamp": timestamp,
    }

    data_root = os.environ.get("C4_DATA_ROOT")
    if not data_root:
        raise PipelineUnavailable("C4_DATA_ROOT is required for live audit storage.")
    root = Path(data_root).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise PipelineUnavailable("C4_DATA_ROOT must be an existing absolute directory.")

    directory = root / "c4_artifacts" / "live_pipeline_audit" / "v1"
    audit = {
        "schema_version": "1.0",
        "upstream": deepcopy(upstream),
        "response": {"c4": result},
        "candidate_training_executed": False,
        "N1_executed": False,
        "N2_executed": False,
        "production_release_authorized": False,
        "processing_before_audit_write_sec": perf_counter() - started,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{audit_id}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(audit, handle, allow_nan=False, indent=2)
    except OSError as exc:
        raise PipelineUnavailable(
            "Live audit storage failed. No decision was returned; inspect worker logs."
        ) from exc

    return {"c4": result}
