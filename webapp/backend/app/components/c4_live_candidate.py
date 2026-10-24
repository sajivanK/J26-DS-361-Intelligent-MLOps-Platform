"""Evaluate an operator-controlled C3 authentic repair package.

No training, forecasting, calibration or deployment is performed.
The manifest and serialized models must come from trusted local storage.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _read(root, value, expected=None):
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Artifact is outside the configured live data root.")
    if not path.is_file():
        raise ValueError("Artifact must be a file.")

    contents = path.read_bytes()
    actual = hashlib.sha256(contents).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("Artifact checksum mismatch.")
    return contents, actual


def _slice_mask(frame, expression):
    """Restricted comparisons joined by AND/OR; no calls or attributes."""
    operators = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
    }

    def visit(node):
        if isinstance(node, ast.Name) and node.id in frame.columns:
            return frame[node.id]

        if isinstance(node, ast.Constant):
            if type(node.value) in (str, int, float):
                return node.value
            raise ValueError("Unsupported slice constant.")

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float)
        ):
            return -node.operand.value

        if isinstance(node, ast.Compare):
            result = pd.Series(True, index=frame.index)
            left = visit(node.left)
            for operation, comparator in zip(node.ops, node.comparators):
                function = operators.get(type(operation))
                if function is None:
                    raise ValueError("Unsupported slice comparison.")
                right = visit(comparator)
                result = result & function(left, right)
                left = right
            return result

        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                if isinstance(node.op, ast.And):
                    result = result & value
                elif isinstance(node.op, ast.Or):
                    result = result | value
                else:
                    raise ValueError("Unsupported slice operator.")
            return result

        raise ValueError("Unsupported slice expression.")

    if not isinstance(expression, str) or len(expression) > 2000:
        raise ValueError("Invalid slice expression.")

    result = visit(ast.parse(expression, mode="eval").body)
    if not isinstance(result, pd.Series):
        raise ValueError("Slice must produce a row-level mask.")
    if result.isna().any() or result.dtype != bool:
        raise ValueError("Invalid slice mask.")
    return result.to_numpy()


def evaluate_manifest(*, data_root, manifest_path):
    root = Path(data_root).resolve(strict=True)
    contents, manifest_hash = _read(root, manifest_path)
    package = json.loads(contents)
    metadata = package["downstream_handoff_metadata"]
    context = metadata["data_context"]

    if package["package_status"] != "APPROVED_FOR_REVIEW":
        raise ValueError("Package is not approved for review.")
    if package["chosen_strategy"] != "AUTHENTIC_WEIGHTED_ADAPTATION":
        raise ValueError("This evaluator supports authentic adaptation only.")
    if package["privacy_risk_indicators"]["status"] != "PASS":
        raise ValueError("Final package privacy status is not PASS.")
    if context["input_representation"] != "model_ready":
        raise ValueError("Model-ready input is required.")
    if context.get("preprocessing_uri") or context.get("target_mapping"):
        raise ValueError("External preprocessing/mapping is not supported here.")
    if metadata["metric"] != "accuracy":
        raise ValueError("This evaluator currently reproduces accuracy repairs.")
    if metadata["evaluation_provenance"]["target"] != "derived current holdout":
        raise ValueError("A supplied holdout needs a separate evaluator.")

    paths = metadata["artifact_paths"]
    hashes = metadata["artifact_sha256"]

    candidate_bytes, candidate_hash = _read(
        root,
        package["candidate_model_path"],
        package["candidate_model_sha256"],
    )
    champion_bytes, champion_hash = _read(
        root, paths["model"], hashes["model"]
    )
    current_bytes, current_hash = _read(
        root, paths["current_dataset"], hashes["current_dataset"]
    )
    adaptation_bytes, _ = _read(
        root,
        package["candidate_dataset_path"],
        package["candidate_dataset_sha256"],
    )

    features = context["feature_columns"]
    target = context["target_column"]
    if (
        not features
        or len(features) != len(set(features))
        or target in features
    ):
        raise ValueError("Invalid ordered feature schema.")

    def frame(contents):
        value = pd.read_csv(io.BytesIO(contents))[features + [target]]
        if value.empty or value.isna().any().any():
            raise ValueError("Empty data or missing values.")
        numeric = value.select_dtypes(include=np.number)
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Non-finite data.")
        return value

    current = frame(current_bytes)
    adaptation = frame(adaptation_bytes)

    def keys(value):
        return pd.util.hash_pandas_object(value[features], index=False)

    current_keys = keys(current)
    representatives = current.loc[~current_keys.duplicated()]
    _, heldout = train_test_split(
        representatives,
        test_size=0.25,
        stratify=representatives[target],
        random_state=metadata["partition_seeds"]["current"],
    )
    evaluation = current.loc[
        current_keys.isin(set(keys(heldout)))
    ].reset_index(drop=True)

    if len(evaluation) != metadata["partition_rows"]["evaluation"]:
        raise ValueError("Evaluation row count differs from provenance.")
    if len(adaptation) != metadata["partition_rows"]["adaptation"]:
        raise ValueError("Adaptation row count differs from provenance.")
    if set(keys(evaluation)) & set(keys(adaptation)):
        raise ValueError("Adaptation/evaluation overlap.")

    rule = package["diagnostic_summary"]["affected_slice_rule"]
    affected = _slice_mask(evaluation[features], rule)
    if not affected.any() or not (~affected).any():
        raise ValueError("Affected slice and complement must be nonempty.")

    # Deserialization is restricted to operator-controlled local artifacts.
    champion = joblib.load(io.BytesIO(champion_bytes))
    candidate = joblib.load(io.BytesIO(candidate_bytes))
    X = evaluation[features]
    y = evaluation[target].to_numpy()
    positive = context["positive_label"]

    predictions = []
    for model in (champion, candidate):
        names = getattr(model, "feature_names_in_", None)
        if names is not None and list(names) != features:
            raise ValueError("Model feature order mismatch.")
        classes = list(getattr(model, "classes_", []))
        if len(classes) != 2 or positive not in classes:
            raise ValueError("Invalid binary model classes.")
        if not set(np.unique(y)).issubset(set(classes)):
            raise ValueError("Evaluation labels differ from model classes.")
        predicted = np.asarray(model.predict(X)).reshape(-1)
        if len(predicted) != len(y):
            raise ValueError("Prediction row count mismatch.")
        if not set(np.unique(predicted)).issubset(set(classes)):
            raise ValueError("Invalid predicted labels.")
        predictions.append(predicted)

    before, after = predictions

    def positive_probabilities(model):
        if not hasattr(model, "predict_proba"):
            raise ValueError("Probability output is required for AUC.")
        classes = list(model.classes_)
        probability = np.asarray(model.predict_proba(X), dtype=float)
        if probability.shape != (len(y), len(classes)):
            raise ValueError("Probability output shape mismatch.")
        if (
            not np.isfinite(probability).all()
            or (probability < 0).any()
            or (probability > 1).any()
            or not np.allclose(probability.sum(axis=1), 1.0)
        ):
            raise ValueError("Invalid model probabilities.")
        return probability[:, classes.index(positive)]

    before_probability = positive_probabilities(champion)
    after_probability = positive_probabilities(candidate)

    def population_auc(mask, probability):
        truth = y[mask] == positive
        if len(np.unique(truth)) < 2:
            return None
        return float(roc_auc_score(truth, probability[mask]))

    measurements = {}
    for name, mask in (
        ("global", np.ones(len(y), dtype=bool)),
        ("affected", affected),
        ("complement", ~affected),
    ):
        old = float(accuracy_score(y[mask], before[mask]))
        new = float(accuracy_score(y[mask], after[mask]))
        f1_before = float(f1_score(
            y[mask], before[mask], pos_label=positive, zero_division=0
        ))
        f1_after = float(f1_score(
            y[mask], after[mask], pos_label=positive, zero_division=0
        ))
        measurements[name] = {
            "rows": int(mask.sum()),
            "champion_accuracy": old,
            "candidate_accuracy": new,
            "accuracy_gain": new - old,
            "champion_positive_f1": f1_before,
            "candidate_positive_f1": f1_after,
            "positive_f1_gain": f1_after - f1_before,
            "champion_macro_f1": float(f1_score(
                y[mask], before[mask], average="macro",
                labels=list(champion.classes_), zero_division=0,
            )),
            "candidate_macro_f1": float(f1_score(
                y[mask], after[mask], average="macro",
                labels=list(champion.classes_), zero_division=0,
            )),
            "champion_auc": population_auc(mask, before_probability),
            "candidate_auc": population_auc(mask, after_probability),
        }

        item = measurements[name]
        item["macro_f1_gain"] = (
            item["candidate_macro_f1"] - item["champion_macro_f1"]
        )
        item["auc_gain"] = (
            None
            if item["champion_auc"] is None or item["candidate_auc"] is None
            else item["candidate_auc"] - item["champion_auc"]
        )

    comparisons = {}
    for population, name in (
        ("global", "global"),
        ("affected_slice", "affected"),
        ("non_affected", "complement"),
    ):
        item = measurements[name]
        comparisons[population] = {
            "rows": item["rows"],
            "champion": {
                "accuracy": item["champion_accuracy"],
                "macro_f1": item["champion_macro_f1"],
                "auc": item["champion_auc"],
            },
            "candidate": {
                "accuracy": item["candidate_accuracy"],
                "macro_f1": item["candidate_macro_f1"],
                "auc": item["candidate_auc"],
            },
            "candidate_minus_champion": {
                "accuracy": item["accuracy_gain"],
                "macro_f1": item["macro_f1_gain"],
                "auc": item["auc_gain"],
            },
        }

    pre_release_report = {
        "comparisons": comparisons,
        "slice_basis": "c2_diagnosed_slice",
        "evaluation_scope": "C3 approval holdout reproduction",
        "fresh_confirmation": False,
    }

    saved = package["utility_metrics"]
    checks = (
        (measurements["global"]["champion_accuracy"], saved["overall_before"]),
        (measurements["global"]["candidate_accuracy"], saved["overall_after"]),
        (measurements["affected"]["champion_accuracy"], saved["baseline_score"]),
        (measurements["affected"]["candidate_accuracy"], saved["candidate_score"]),
    )
    if any(
        not np.isclose(actual, expected, rtol=0, atol=1e-12)
        for actual, expected in checks
    ):
        raise ValueError("Saved C3 accuracy does not reproduce.")

    return {
        "status": "CANDIDATE_EVALUATION_VERIFIED",
        "pipeline_run_id": metadata["pipeline_run_id"],
        "package_id": package["package_id"],
        "champion_model_id": package["model_id"],
        "manifest_sha256": manifest_hash,
        "candidate_model_sha256": candidate_hash,
        "champion_model_sha256": champion_hash,
        "current_dataset_sha256": current_hash,
        "measurements": measurements,
        "pre_release_report": pre_release_report,
        "evaluation_scope": "C3 approval holdout reproduction",
        "fresh_confirmation": False,
        "rollback_ready": None,
        "N1_executed": False,
        "N2_executed": False,
        "production_release_authorized": False,
    }


def evaluate_upstream(*, data_root, upstream):
    """Resolve an approved package from trusted local C3 run storage."""
    root = Path(data_root).resolve(strict=True)
    context = upstream["context"]
    c1 = upstream["c1"]
    c2 = upstream["c2_to_c4"]
    c3 = upstream["c3"]

    if c3.get("repair_package_status") != "APPROVED":
        raise ValueError("Candidate evaluation requires an approved package.")
    if c3.get("privacy_status") != "PASS":
        raise ValueError("Final package privacy check did not pass.")
    if c3.get("repair_strategy") != "AUTHENTIC_WEIGHTED_ADAPTATION":
        raise ValueError("This evaluator supports authentic adaptation only.")

    stability = c2.get("diagnosis_stability")
    if (
        type(stability) not in (int, float)
        or not np.isfinite(stability)
        or not 0.5 <= stability <= 1.0
    ):
        raise ValueError("Diagnosis stability does not pass.")

    run_id = context.get("run_id")
    champion = context.get("model_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Missing pipeline run ID.")
    if not champion or champion != c1.get("model_id"):
        raise ValueError("Champion model identity mismatch.")

    repair_value = c3.get("repair_dataset_uri")
    if not isinstance(repair_value, str) or not repair_value.strip():
        raise ValueError("Missing repair dataset reference.")

    repair_path = Path(repair_value)
    if not repair_path.is_absolute():
        repair_path = root / repair_path
    repair_path = repair_path.resolve(strict=True)

    # A request may select an existing C3 run, never an arbitrary model file.
    runs_root = (root / "platform_runs/c3_repairs").resolve(strict=True)
    if not repair_path.is_relative_to(runs_root):
        raise ValueError("Repair dataset is outside trusted C3 run storage.")

    manifests = list(repair_path.parent.glob("*_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one C3 package manifest.")

    manifest_bytes, _ = _read(root, manifests[0])
    package = json.loads(manifest_bytes)
    metadata = package["downstream_handoff_metadata"]

    if metadata.get("pipeline_run_id") != run_id:
        raise ValueError("Package belongs to a different pipeline run.")
    if package.get("model_id") != champion:
        raise ValueError("Package belongs to a different champion.")
    if package.get("package_status") != "APPROVED_FOR_REVIEW":
        raise ValueError("Saved package is not approved for review.")
    if package.get("chosen_strategy") != c3["repair_strategy"]:
        raise ValueError("Repair strategy differs from saved package.")

    saved_repair = Path(package["candidate_dataset_path"]).resolve(strict=True)
    if repair_path != saved_repair:
        raise ValueError("Repair dataset differs from saved package.")

    saved_current = Path(
        metadata["artifact_paths"]["current_dataset"]
    ).resolve(strict=True)
    current_value = context.get("current_dataset_uri")
    if not isinstance(current_value, str) or not current_value.strip():
        raise ValueError("Missing current dataset reference.")
    current_path = Path(current_value)
    if not current_path.is_absolute():
        current_path = root / current_path
    if current_path.resolve(strict=True) != saved_current:
        raise ValueError("Current batch differs from saved package.")

    original_c2 = metadata["original_c2"]
    if c2.get("affected_slice_rule") != original_c2.get("affected_slice_rule"):
        raise ValueError("Incoming slice rule differs from saved diagnosis.")
    if stability != original_c2.get("diagnosis_stability"):
        raise ValueError("Incoming diagnosis stability differs from package.")

    # Bind all fields of the flat C3 contract to its saved export.
    saved_bytes, _ = _read(root, repair_path.parent / "result.json")
    saved_result = json.loads(saved_bytes)
    fields = (
        "repair_package_status", "repair_strategy", "generator",
        "n_synthetic_rows", "augmentation_ratio", "fidelity_score",
        "utility_score", "privacy_status", "target_coverage",
        "prescription_confidence", "repair_version", "repair_timestamp",
    )
    for field in fields:
        if field not in c3 or c3[field] != saved_result.get(field):
            raise ValueError(f"C3 field differs from saved export: {field}")

    result = evaluate_manifest(
        data_root=root,
        manifest_path=manifests[0],
    )
    result["upstream_package_binding_verified"] = True
    return result

def verify_champion_recovery(*, data_root, manifest_path):
    """Preserve champion bytes and verify recovery predictions.

    This checks artifact recovery only. It does not switch a serving endpoint.
    """
    import os
    import tempfile

    root = Path(data_root).resolve(strict=True)
    contents, manifest_hash = _read(root, manifest_path)
    package = json.loads(contents)
    metadata = package["downstream_handoff_metadata"]
    context = metadata["data_context"]

    champion_bytes, champion_hash = _read(
        root,
        metadata["artifact_paths"]["model"],
        metadata["artifact_sha256"]["model"],
    )
    current_bytes, _ = _read(
        root,
        metadata["artifact_paths"]["current_dataset"],
        metadata["artifact_sha256"]["current_dataset"],
    )

    directory = root / "c4_artifacts/live_champion_recovery/v1"
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    if not directory.is_relative_to(root):
        raise ValueError("Recovery directory is outside the data root.")

    snapshot = directory / f"{champion_hash}.joblib"

    if not snapshot.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".pending-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(champion_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, snapshot)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    restored_bytes, _ = _read(root, snapshot, champion_hash)
    original = joblib.load(io.BytesIO(champion_bytes))
    restored = joblib.load(io.BytesIO(restored_bytes))

    features = context["feature_columns"]
    probe = pd.read_csv(io.BytesIO(current_bytes))[features]
    if probe.empty or probe.isna().any().any():
        raise ValueError("Invalid recovery probe data.")

    if not np.array_equal(
        original.predict(probe), restored.predict(probe)
    ):
        raise ValueError("Recovered champion predictions differ.")

    if hasattr(original, "predict_proba"):
        if not np.array_equal(
            original.predict_proba(probe),
            restored.predict_proba(probe),
        ):
            raise ValueError("Recovered champion probabilities differ.")

    return {
        "status": "CHAMPION_ARTIFACT_RECOVERY_VERIFIED",
        "champion_model_id": package["model_id"],
        "champion_model_sha256": champion_hash,
        "manifest_sha256": manifest_hash,
        "snapshot_path": str(snapshot),
        "probe_rows": len(probe),
        "serving_rollback_tested": False,
        "production_release_authorized": False,
    }

def _evaluate_approved(upstream):
    """Evaluate a bound authentic candidate; never authorize deployment."""
    from app.components.c4_live_candidate import (
        evaluate_upstream,
        verify_champion_recovery,
    )

    configured = os.environ.get("J26_DATA_DIR")
    if not configured:
        raise PipelineUnavailable(
            "J26_DATA_DIR is required for trusted live C3 artifacts."
        )

    root = Path(configured).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise PipelineUnavailable(
            "J26_DATA_DIR must be an existing absolute directory."
        )
    root = root.resolve()

    try:
        evaluation = evaluate_upstream(
            data_root=root, upstream=upstream
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
            recovery["manifest_sha256"] != evaluation["manifest_sha256"]
            or recovery["champion_model_sha256"]
            != evaluation["champion_model_sha256"]
        ):
            raise ValueError("Evaluation and recovery bindings differ.")
    except (OSError, ValueError, KeyError) as exc:
        raise PipelineUnavailable(
            "Live candidate artifact, binding or recovery verification "
            "failed. Inspect worker logs for the underlying cause."
        ) from exc

    measurements = evaluation["measurements"]
    audit_id = "c4-live-" + uuid4().hex
    version = "c4-live-authentic-review-v1"

    result = {
        "decision": "HUMAN_REVIEW",
        "evaluation_status": "FORECAST_UNAVAILABLE",
        "champion_model_id": evaluation["champion_model_id"],
        "candidate_model_id": (
            "c3-candidate-" + evaluation["candidate_model_sha256"]
        ),
        # Configured target, not measured live coverage.
        "joint_confidence_level": 0.95,
        "future_safety_margins": dict.fromkeys(MARGINS),
        "future_safety_intervals": dict.fromkeys(MARGINS),
        "pre_release_evidence": {
            "validation_global_f1_gain":
                measurements["global"]["positive_f1_gain"],
            "validation_affected_slice_recovery":
                measurements["affected"]["positive_f1_gain"],
            "validation_non_affected_margin":
                measurements["complement"]["positive_f1_gain"],
        },
        "upstream_evidence_summary": {
            "c1": deepcopy(upstream["c1"]),
            "c2": deepcopy(upstream["c2_to_c4"]),
            "c3": deepcopy(upstream["c3"]),
        },
        "hard_gate_status": {
            "repair_approved": True,
            "privacy_check": "PASS",
            "schema_compatible": True,
            "candidate_artifact_valid": True,
            # Artifact availability/recovery, as defined by the contract.
            "rollback_ready": True,
            "overall": "PASS",
        },
        "risk_control": {
            "policy_version": version,
            "status": "NOT_APPLIED_FORECAST_UNAVAILABLE",
        },
        "reason": (
            "The bound C3 authentic candidate was loaded and evaluated; "
            "saved current-holdout accuracy was reproduced. Champion "
            "recovery artifact availability and prediction equivalence "
            "were verified. Live forecasting and conformal calibration "
            "for this repair strategy were not executed. Global "
            "positive-class F1 gain is "
            f"{measurements['global']['positive_f1_gain']:+.6f}. "
            "This is reuse of C3 approval evidence, not fresh confirmation."
        ),
        "required_action": (
            "Keep the champion active. Review measured accuracy and "
            "positive-class F1 changes, obtain fresh confirmation evidence, "
            "and validate applicable forecasting/calibration before release. "
            "Do not deploy automatically."
        ),
        "audit_id": audit_id,
        "policy_version": version,
        "forecast_model_version": None,
        "conformal_config_version": None,
        "mlflow_run_id": None,
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    audit_root_value = os.environ.get("C4_DATA_ROOT")
    if not audit_root_value:
        raise PipelineUnavailable("C4_DATA_ROOT is required for audit storage.")
    audit_root = Path(audit_root_value).expanduser()
    if not audit_root.is_absolute() or not audit_root.is_dir():
        raise PipelineUnavailable("Invalid C4_DATA_ROOT.")

    audit = {
        "schema_version": "1.1",
        "upstream": deepcopy(upstream),
        "response": {"c4": result},
        "candidate_evaluation": evaluation,
        "champion_recovery": recovery,
        "pre_release_metric": "positive_class_f1_gain",
        "measurement_units": "fraction",
        "candidate_training_executed": False,
        "N1_executed": False,
        "N2_executed": False,
        "serving_rollback_tested": False,
        "production_release_authorized": False,
    }

    directory = audit_root / "c4_artifacts/live_pipeline_audit/v1"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{audit_id}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(audit, handle, allow_nan=False, indent=2)
    except OSError as exc:
        raise PipelineUnavailable(
            "Live audit persistence failed; no decision was returned."
        ) from exc

    return {"c4": result}