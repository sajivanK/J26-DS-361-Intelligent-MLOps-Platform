"""Live C4 upstream gates and authentic-candidate review.

No candidate training, future forecasting or production deployment.
"""
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
REVIEW_POLICY_VERSION = "c4-live-authentic-review-v1"

MARGINS = (
    "global_margin",
    "affected_slice_margin",
    "non_affected_margin",
)
EVIDENCE = (
    "validation_global_f1_gain",
    "validation_affected_slice_recovery",
    "validation_non_affected_margin",
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _directory_from_env(name):
    value = os.environ.get(name)
    if not value:
        raise PipelineUnavailable(f"{name} is required.")

    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise PipelineUnavailable(
            f"{name} must be an existing absolute directory."
        )
    return path.resolve()


def _save_audit(upstream, result, *, started, details=None):
    root = _directory_from_env("C4_DATA_ROOT")
    directory = root / "c4_artifacts/live_pipeline_audit/v1"

    audit = {
        "schema_version": "1.1",
        "upstream": deepcopy(upstream),
        "response": {"c4": result},
        "candidate_training_executed": False,
        "N1_executed": False,
        "N2_executed": False,
        "production_release_authorized": False,
        "processing_before_audit_write_sec": perf_counter() - started,
    }
    if details:
        audit["evaluation_details"] = details

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{result['audit_id']}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(audit, handle, allow_nan=False, indent=2)
    except OSError as exc:
        raise PipelineUnavailable(
            "Live audit storage failed. No decision was returned; "
            "inspect worker logs."
        ) from exc

    return {"c4": result}


def _base_result(upstream, *, version):
    return {
        "champion_model_id": upstream["c1"]["model_id"],
        "candidate_model_id": None,
        # Configured target only; not measured live coverage.
        "joint_confidence_level": 0.95,
        "future_safety_margins": dict.fromkeys(MARGINS),
        "future_safety_intervals": dict.fromkeys(MARGINS),
        "pre_release_evidence": dict.fromkeys(EVIDENCE),
        "upstream_evidence_summary": {
            "c1": deepcopy(upstream["c1"]),
            "c2": deepcopy(upstream["c2_to_c4"]),
            "c3": deepcopy(upstream["c3"]),
        },
        "audit_id": "c4-live-" + uuid4().hex,
        "policy_version": version,
        "forecast_model_version": None,
        "conformal_config_version": None,
        "mlflow_run_id": None,
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_pipeline(upstream):
    started = perf_counter()

    if not isinstance(upstream, dict):
        raise PipelineInputError("Expected an upstream JSON object.")

    try:
        json.dumps(upstream, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise PipelineInputError(
            "Upstream must contain finite JSON values."
        ) from exc

    for name in ("context", "c1", "c2_to_c4", "c3"):
        if not isinstance(upstream.get(name), dict):
            raise PipelineInputError(
                f"Missing upstream object: {name}."
            )

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
        raise PipelineInputError(
            "C3 n_synthetic_rows must be a nonnegative integer."
        )

    failures = []
    if package == "REJECTED":
        failures.append("C3 repair package REJECTED")
    if privacy == "FAIL":
        failures.append("C3 privacy check FAIL")

    # NONE / zero synthetic rows do not independently reject a repair.
    if stability is None:
        if not failures:
            raise PipelineInputError(
                "Diagnosis stability is required to continue."
            )
    elif not _number(stability) or not 0 <= stability <= 1:
        raise PipelineInputError("Invalid diagnosis_stability.")
    elif stability < 0.5:
        failures.append("Diagnosis stability below 0.5")

    if not failures:
        return _evaluate_approved(upstream, started=started)

    result = _base_result(upstream, version=POLICY_VERSION)
    result.update({
        "decision": "ESCALATE",
        "evaluation_status": "UPSTREAM_GATE_FAILED",
        "hard_gate_status": {
            "repair_approved": package == "APPROVED",
            "privacy_check": privacy,
            "schema_compatible": None,
            "candidate_artifact_valid": None,
            "rollback_ready": None,
            "overall": "FAIL",
        },
        "risk_control": {
            "policy_version": POLICY_VERSION,
            "status": "NOT_APPLIED_UPSTREAM_GATE_FAILED",
        },
        "reason": (
            "; ".join(failures)
            + ". Candidate evaluation and forecasting were not run."
        ),
        "required_action": (
            "Keep the champion. Review upstream rejection evidence "
            "and repair prerequisites before retrying. "
            "Do not deploy a candidate."
        ),
    })

    return _save_audit(upstream, result, started=started)


def _evaluate_approved(upstream, *, started):
    # Lazy import keeps upstream rejection independent of ML evaluation.
    from app.components.c4_live_candidate import (
        evaluate_upstream,
        verify_champion_recovery,
    )

    root = _directory_from_env("J26_DATA_DIR")

    # Validate audit configuration before doing candidate evaluation.
    _directory_from_env("C4_DATA_ROOT")

    try:
        evaluation = evaluate_upstream(
            data_root=root,
            upstream=upstream,
        )

        repair = Path(upstream["c3"]["repair_dataset_uri"])
        if not repair.is_absolute():
            repair = root / repair
        repair = repair.resolve(strict=True)

        manifests = list(repair.parent.glob("*_manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Expected one bound package manifest.")

        recovery = verify_champion_recovery(
            data_root=root,
            manifest_path=manifests[0],
        )

        if (
            recovery["manifest_sha256"]
            != evaluation["manifest_sha256"]
            or recovery["champion_model_sha256"]
            != evaluation["champion_model_sha256"]
        ):
            raise ValueError(
                "Evaluation and champion recovery bindings differ."
            )

    except (OSError, ValueError, KeyError) as exc:
        raise PipelineUnavailable(
            "Live candidate artifact, package binding or champion "
            "recovery verification failed. Inspect worker logs."
        ) from exc

    measurements = evaluation["measurements"]
    result = _base_result(
        upstream,
        version=REVIEW_POLICY_VERSION,
    )
    result.update({
        "decision": "HUMAN_REVIEW",
        "evaluation_status": "FORECAST_UNAVAILABLE",
        "candidate_model_id": (
            "c3-candidate-" + evaluation["candidate_model_sha256"]
        ),
        # All three values use positive-class F1 gain, in fraction units.
        # Accuracy measurements remain separately named in the audit.
        "pre_release_evidence": {
            "validation_global_f1_gain":
                measurements["global"]["positive_f1_gain"],
            "validation_affected_slice_recovery":
                measurements["affected"]["positive_f1_gain"],
            "validation_non_affected_margin":
                measurements["complement"]["positive_f1_gain"],
        },
        "hard_gate_status": {
            "repair_approved": True,
            "privacy_check": "PASS",
            "schema_compatible": True,
            "candidate_artifact_valid": True,
            # Contract definition: recoverable champion artifact available.
            # This does not claim a serving rollback was tested.
            "rollback_ready": True,
            "overall": "PASS",
        },
        "risk_control": {
            "policy_version": REVIEW_POLICY_VERSION,
            "status": "NOT_APPLIED_FORECAST_UNAVAILABLE",
        },
        "reason": (
            "The bound authentic candidate was loaded and evaluated. "
            "C3 current-holdout accuracy was reproduced, and champion "
            "artifact recovery was verified. Live forecasts and conformal "
            "intervals were not computed. Global positive-class F1 gain "
            f"is {measurements['global']['positive_f1_gain']:+.6f}. "
            "Evaluation reuses C3 approval evidence; it is not fresh "
            "confirmation. Serving rollback was not tested."
        ),
        "required_action": (
            "Keep the champion active. Review accuracy and positive-class "
            "F1 changes, obtain fresh confirmation evidence, and validate "
            "applicable forecasting and calibration before release. "
            "Do not deploy automatically."
        ),
    })

    details = {
        "candidate_evaluation": evaluation,
        "champion_recovery": recovery,
        "pre_release_metric": "positive_class_f1_gain",
        "measurement_units": "fraction",
        "serving_rollback_tested": False,
    }

    return _save_audit(
        upstream,
        result,
        started=started,
        details=details,
    )