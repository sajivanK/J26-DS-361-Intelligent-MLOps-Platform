"""Experimental transfer assessment using actual upstream packages."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def checked_json(path):
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    expected = path.with_suffix(".sha256").read_text().strip()
    if digest != expected:
        raise RuntimeError("Frozen artifact checksum mismatch: " + path.name)
    return json.loads(content), digest


def evaluate_sandbox(upstream):
    from .c4_pipeline_service import evaluate_pipeline
    from .c4_live_candidate import evaluate_upstream
    from .c4_live_features import build_feature_inventory

    repo = Path(__file__).resolve().parents[4]
    for folder in (
        repo / "components/c4/src",
        repo / "components/c4/scripts",
    ):
        text = str(folder)
        if text not in sys.path:
            sys.path.insert(0, text)

    from fit_n1_hybrid_development import HybridForecaster
    from c4.conformal.benchmark_n2 import (
        JointCalibration, apply_joint_calibration,
    )
    from c4.risk.benchmark_n3 import assess_safety
    from c4.risk.benchmark_policy import policy_decision

    # Existing service verifies upstream gates, package and recovery.
    baseline = evaluate_pipeline(upstream)["c4"]
    if baseline["evaluation_status"] == "UPSTREAM_GATE_FAILED":
        return {
            "scope": "experimental_transfer",
            "pipeline_assessment": baseline,
            "experimental_decision": "ESCALATE",
            "N1_executed": False,
            "N2_executed": False,
            "live_coverage_validated": False,
            "production_release_authorized": False,
        }

    if upstream["c3"].get("repair_strategy") != "AUTHENTIC_WEIGHTED_ADAPTATION":
        raise ValueError("Sandbox currently supports authentic adaptation only.")

    research = os.environ.get("C4_DATA_ROOT")
    if not research:
        raise RuntimeError("C4_DATA_ROOT is required.")
    live_root = Path(
        os.environ.get("J26_DATA_DIR", str(repo.parent / "data"))
    ).resolve()

    root = (
        Path(research)
        / "c4_artifacts/forecasting_experiments/adult"
    )
    model_dir = root / "adult-hybrid-development-v1/fit-v1"
    validation = root / "adult-hybrid-fresh-validation-v1"

    model = HybridForecaster.load(model_dir)
    evaluation = evaluate_upstream(
        data_root=live_root, upstream=upstream,
    )
    inventory = build_feature_inventory(
        upstream=upstream,
        evaluation=evaluation,
        columns=model.features,
    )
    values = dict(zip(inventory["columns"], inventory["values"]))
    missing = {name for name, value in values.items() if value is None}
    expected = {"c3_fidelity_score", "c3_target_coverage"}
    if missing != expected:
        raise ValueError("Unexpected unavailable sandbox features.")

    substitutions = {}
    for name in sorted(expected):
        index = model.ridge["columns"].index(name)
        replacement = float(model.ridge["mean"][index])
        values[name] = replacement
        substitutions[name] = {
            "original": None,
            "replacement": replacement,
            "source": "frozen Ridge training mean",
        }

    frame = pd.DataFrame(
        [[values[name] for name in model.features]],
        columns=model.features,
    )
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite sandbox features.")
    predictions = model.predict(frame)
    if predictions.shape != (1, 9):
        raise RuntimeError("Unexpected forecast dimensions.")

    calibration_document, calibration_hash = checked_json(
        validation / "n2-calibration-v1/completion-v1/calibration.json"
    )
    selection, _ = checked_json(
        validation / "policy-selection-v1/completion-v1/selection.json"
    )
    if selection["N2_calibration_sha256"] != calibration_hash:
        raise RuntimeError("Calibration/policy binding mismatch.")

    method = "hybrid_ridge_lightgbm_lightgbm"
    calibration = JointCalibration.from_dict(
        calibration_document["methods"][method]
    )
    adjusted = apply_joint_calibration(calibration, predictions)
    intervals = {
        target: [
            float(adjusted.iloc[0][target + "__lower"]),
            float(adjusted.iloc[0][target + "__upper"]),
        ]
        for target in calibration.target_columns
    }

    # Saved benchmark support is not validated live coverage.
    safety = assess_safety(
        intervals,
        auto_deploy_supported=calibration.auto_deploy_supported,
    )
    status = baseline["hard_gate_status"]
    gates = {
        "repair_approved": status.get("repair_approved") is True,
        "privacy_check": status.get("privacy_check") in (True, "PASS"),
        "diagnosis_stability": upstream["c2_to_c4"]["diagnosis_stability"] >= 0.5,
        "schema_compatible": status.get("schema_compatible") is True,
        "rollback_ready": status.get("rollback_ready") is True,
    }
    policy = selection["methods"][method]["adaptive"]["selected_policy"]
    governance = policy_decision(
        intervals,
        values,
        gates,
        n2_supported=calibration.auto_deploy_supported,
        policy=policy,
        unsupported_contexts=selection["context_support"]["unsupported_contexts"],
    )

    return {
        "scope": "experimental_transfer",
        "pipeline_run_id": upstream["context"]["run_id"],
        "package_id": evaluation["package_id"],
        "pipeline_assessment": baseline,
        "pre_release_report": evaluation["pre_release_report"],
        "substitutions": substitutions,
        "forecasts": predictions.iloc[0].to_dict(),
        "transferred_intervals": intervals,
        "transferred_adjustment": calibration.adjustment,
        "safety": safety,
        "governance": governance,
        "experimental_decision": governance["decision"],
        "selected_policy": policy,
        "N1_executed": True,
        "N2_executed": True,
        "live_coverage_validated": False,
        "fresh_confirmation": False,
        "production_release_authorized": False,
        "note": (
            "Actual package evaluation with experimental benchmark-model "
            "transfer. Approval holdout reused. Two inputs imputed. "
            "Intervals do not establish 95% live coverage."
        ),
    }
