"""Artifact-bound C1/C2 integration for tabular binary classifiers.

This service has no scenario, dataset-name, or cached-candidate fallbacks. A
request carries its own schema and trusted artifact references. Evaluation rows
are never used to fit a generator or candidate model.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from component3.src.contracts.external_contracts import (
    ContractValidationError, DependencyResolutionError, validate_three_section_contract,
    resolve_data_context_artifacts, verify_path_authorization,
)
from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus, compute_file_sha256
from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule, UnsafeSliceRuleError
from component3.src.core.model_loader import load_trusted_pickle
from component3.src.generators.generator_selector import GeneratorSelector, GeneratorExecutionError
from component3.src.models.model_registry import get_model_registry
from component3.src.orchestration.orchestrator_models import ExecutionState, OrchestratorExecutionLog
from component3.src.validation.fidelity_validator import FidelityValidator
from component3.src.validation.privacy_validator import PrivacyValidator


def artifact_path(value, category):
    ok, error, path = verify_path_authorization(value, category)
    if not ok or path is None or not path.is_file():
        raise DependencyResolutionError(error or f"Missing {category} artifact: {value}")
    return path


def _number(options, name, default, low, high, integer=False):
    value = options.get(name, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high
            or (integer and not isinstance(value, int))):
        raise ContractValidationError(f"{name} must be {'an integer' if integer else 'a number'} in [{low}, {high}].")
    return value


def _frame(path, features, target, name):
    frame = pd.read_csv(path)
    required = features + [target]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ContractValidationError(f"{name} missing columns: {missing}")
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any():
        raise ContractValidationError(f"{name} must be non-empty with no missing feature/target values.")
    numeric = frame.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ContractValidationError(f"{name} contains non-finite numbers.")
    return frame


def _row_keys(frame, features):
    # Features alone define leakage: changing a duplicate row's label does not
    # make that row an independent evaluation observation.
    return pd.util.hash_pandas_object(frame[features], index=False)


def _disjoint(left, right, features, description):
    if set(_row_keys(left, features)) & set(_row_keys(right, features)):
        raise ContractValidationError(f"Overlapping feature rows in {description}; provide independent partitions.")


def _split(frame, target, seed):
    features = [c for c in frame if c != target]
    keys = _row_keys(frame, features)
    representatives = frame.loc[~keys.duplicated()]
    try:
        train, test = train_test_split(representatives, test_size=0.25,
                                       stratify=representatives[target], random_state=seed)
    except ValueError as exc:
        raise ContractValidationError(f"Insufficient class support for an independent split: {exc}") from exc
    train_keys, test_keys = set(_row_keys(train, features)), set(_row_keys(test, features))
    return (frame.loc[keys.isin(train_keys)].reset_index(drop=True),
            frame.loc[keys.isin(test_keys)].reset_index(drop=True))


class ModelInputs:
    """Use the supplied fitted transformer; never refit it on production rows."""
    def __init__(self, model, preprocessor, features, target, mapping, positive_label=None):
        self.model, self.preprocessor = model, preprocessor
        self.features, self.target = features, target
        self.mapping = mapping or getattr(preprocessor, "target_mapping", None) or {}
        self.classes = list(getattr(model, "classes_", []))
        if len(self.classes) != 2:
            raise ContractValidationError("Supported task: a fitted binary classifier exposing classes_.")
        self.positive = self.classes[1] if positive_label is None else positive_label
        if self.positive not in self.classes:
            raise ContractValidationError("positive_label must be a label in model.classes_.")
        try:
            clone(model)
        except Exception as exc:
            raise ContractValidationError(f"Model must support sklearn clone/get_params and fit: {exc}") from exc

    def X(self, frame):
        features = frame[self.features].copy()
        if self.preprocessor is None:
            return features
        # Native C3 preprocessors may need the target as well as features.
        value = self.preprocessor.transform(features)
        if hasattr(value, "X"):
            value = value.X
        elif isinstance(value, tuple):
            value = value[0]
        if getattr(value, "shape", (0,))[0] != len(frame):
            raise ContractValidationError("Preprocessing changed the row count.")
        return value

    def y(self, frame):
        raw = frame[self.target]
        if self.mapping:
            # JSON object keys are strings; native mappings may use numbers.
            values = raw.map(lambda v: self.mapping.get(v, self.mapping.get(str(v))))
        else:
            values = raw
        if values.isna().any() or not set(values.unique()).issubset(set(self.classes)):
            raise ContractValidationError("Target labels do not match model.classes_; supply an exact target_mapping.")
        return values.to_numpy()

    def score(self, model, frame, metric):
        X, y = self.X(frame), self.y(frame)
        if len(set(y)) != 2:
            raise ContractValidationError("Each evaluation slice must contain both target classes.")
        if metric == "roc_auc":
            if not hasattr(model, "predict_proba"):
                raise ContractValidationError("roc_auc requires predict_proba.")
            index = list(model.classes_).index(self.positive)
            return float(roc_auc_score(y == self.positive, model.predict_proba(X)[:, index]))
        predictions = model.predict(X)
        if metric == "accuracy":
            return float(accuracy_score(y, predictions))
        return float(f1_score(y, predictions, pos_label=self.positive, zero_division=0))

    def fit(self, frame):
        candidate = clone(self.model)
        candidate.fit(self.X(frame), self.y(frame))
        return candidate

    def improvement_interval(self, baseline, candidate, frame, metric):
        """Paired, stratified bootstrap on untouched evaluation observations."""
        X, y = self.X(frame), self.y(frame)
        positive = self.positive
        if metric == "roc_auc":
            base = baseline.predict_proba(X)[:, list(baseline.classes_).index(positive)]
            after = candidate.predict_proba(X)[:, list(candidate.classes_).index(positive)]
            truth = y == positive
            measure = roc_auc_score
        else:
            base, after, truth = baseline.predict(X), candidate.predict(X), y
            measure = accuracy_score if metric == "accuracy" else lambda a, b: f1_score(a, b, pos_label=positive, zero_division=0)
        rng = np.random.default_rng(42)
        groups = [np.flatnonzero(y == value) for value in self.classes]
        gains = []
        for _ in range(300):
            indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            gains.append(measure(truth[indices], after[indices]) - measure(truth[indices], base[indices]))
        return [float(v) for v in np.percentile(gains, [2.5, 97.5])]


def execute_live_repair(payload, output_directory=None, *, generator_selector=None):
    """Return (RepairPackage, execution log) for an explicit three-section input.

    New datasets need feature_columns and categorical_features. Models may be
    sklearn-compatible classifiers or pipelines; numeric/model-ready inputs must
    explicitly declare input_representation='model_ready' if no preprocessor is
    supplied. Errors name the missing dependency instead of substituting data.
    """
    ok, errors, container = validate_three_section_contract(payload)
    if not ok:
        raise ContractValidationError("; ".join(errors))
    dc, c1, c2 = container.data_context, container.c1_evidence, container.c2_diagnosis
    options = dc.raw_metadata
    c1_raw = c1.raw_metadata
    if c1_raw.get("model_id", dc.model_id) != dc.model_id:
        raise ContractValidationError("C1 model_id does not match data_context.model_id.")
    for name in ("reference_dataset_uri", "current_dataset_uri"):
        if c1_raw.get(name) and Path(c1_raw[name]).resolve() != Path(getattr(dc, name)).resolve():
            raise ContractValidationError(f"C1 {name} does not match data_context.")
    if c2.raw_metadata.get("model_id", dc.model_id) != dc.model_id:
        raise ContractValidationError("C2 model_id does not match data_context.model_id.")
    ok, errors, paths = resolve_data_context_artifacts(dc)
    if not ok:
        raise DependencyResolutionError("; ".join(errors))

    registry = get_model_registry()
    entry = registry.get(dc.model_id) if registry.has(dc.model_id) else None
    model_uri = options.get("model_uri") or options.get("model_path") or (entry.model_path if entry else dc.model_id)
    model_path = artifact_path(model_uri, "models")
    explicit_model = options.get("model_uri") or options.get("model_path")
    prep_uri = dc.preprocessing_uri or (entry.preprocessor_path if entry and not explicit_model and options.get("input_representation") != "model_ready" else None)
    prep_path = artifact_path(prep_uri, "models") if prep_uri else None
    paths["model"] = model_path
    if prep_path:
        paths["preprocessor"] = prep_path
    expected_hashes = options.get("artifact_sha256", {})
    if not isinstance(expected_hashes, dict):
        raise ContractValidationError("artifact_sha256 must be an object.")
    hashes = {key: compute_file_sha256(path) for key, path in paths.items()}
    if c1_raw.get("model_sha256") and c1_raw["model_sha256"] != hashes["model"]:
        raise DependencyResolutionError("C1 monitored-model hash does not match the supplied model.")
    for key, expected in expected_hashes.items():
        if key in ("source_evaluation_uri", "evaluation_dataset_uri"):
            continue
        if key not in hashes or hashes[key] != expected:
            raise DependencyResolutionError(f"Artifact hash mismatch: {key}")

    run_id = f"c3-{uuid.uuid4().hex}"
    request_id = f"REQ-{uuid.uuid4().hex}"
    log = OrchestratorExecutionLog(run_id, request_id, ExecutionState.VALIDATING)
    from component3.src.contracts.external_contracts import REPO_ROOT
    out_dir = Path(output_directory) if output_directory else REPO_ROOT / "component3/results/live_runs"
    authorized, error, resolved = verify_path_authorization(out_dir, "outputs")
    if not authorized:
        raise DependencyResolutionError(error)
    out_dir = resolved / run_id
    started = time.perf_counter()
    original = dict(container.raw_payload)
    original = {**original.get("upstream", {}), **original}
    provenance = {
        "mode": "integrated", "artifact_sha256": hashes,
        "artifact_paths": {key: str(path) for key, path in paths.items()},
        "original_c1": original.get("c1_evidence", original.get("c1", {})),
        "original_c2": original.get("c2_diagnosis", original.get("c2_to_c3", original.get("c2", {}))),
        "decision_mode": "RULE_BASED", "seed": 42,
        "data_context": {**options, "model_id": dc.model_id, "dataset_id": dc.dataset_id,
                         "target_column": dc.target_column, "label_status": dc.label_status,
                         "preprocessing_uri": str(prep_path) if prep_path else None},
    }
    package = RepairPackage(request_id=request_id, run_id=run_id, dataset_id=dc.dataset_id,
        model_id=dc.model_id, diagnostic_summary={"affected_slice_rule": c2.affected_slice_rule},
        drift_summary={"predicted_harm": c1.predicted_harm_proportion,
                       "harm_severity": c1.drift_severity},
        downstream_handoff_metadata=provenance, evidence_limitations=(
            "Synthetic labels are generated candidates; acceptance requires independent labeled evaluation.",
            "Empirical privacy screening is not differential privacy.",))

    def finish(reasons=(), **changes):
        result = replace(package, rejection_reasons=tuple(reasons), **changes)
        state = {RepairPackageStatus.APPROVED_FOR_REVIEW: ExecutionState.ACCEPTED,
                 RepairPackageStatus.ABSTAINED_NO_REPAIR: ExecutionState.ACCEPTED,
                 RepairPackageStatus.BLOCKED: ExecutionState.BLOCKED,
                 RepairPackageStatus.FAILED: ExecutionState.FAILED}.get(result.package_status, ExecutionState.REJECTED)
        log.transition_to(state, "; ".join(reasons))
        log.stage_durations["total_pipeline_seconds"] = round(time.perf_counter() - started, 3)
        log.package_path = str(result.save_package(out_dir))
        return result, log

    if c1.drift_severity.lower() == "none" or c1.predicted_harm_proportion < 0.01:
        return finish(["No meaningful predicted harm; no repair attempted."],
                      package_status=RepairPackageStatus.ABSTAINED_NO_REPAIR)
    if not dc.has_authentic_labels:
        return finish(["Authentic current labels unavailable; repair utility cannot be verified."],
                      package_status=RepairPackageStatus.BLOCKED)
    if c2.diagnosis_stability < 0.5:
        return finish(["Diagnosis stability below 0.50."], package_status=RepairPackageStatus.BLOCKED)

    features = options.get("feature_columns")
    if not isinstance(features, list) or not features or not all(isinstance(f, str) and f for f in features):
        raise ContractValidationError("data_context.feature_columns must list the model's ordered input features.")
    if len(set(features)) != len(features) or dc.target_column in features:
        raise ContractValidationError("feature_columns must be unique and exclude the target.")
    categories = options.get("categorical_features", [])
    if (not isinstance(categories, list) or not all(isinstance(f, str) for f in categories)
            or len(set(categories)) != len(categories) or not set(categories).issubset(features)):
        raise ContractValidationError("categorical_features must be a subset of feature_columns.")
    numeric = [f for f in features if f not in categories]
    if not set(c1.affected_features).issubset(features) or not set(c2.affected_features_ranked).issubset(features) or c2.root_cause_feature not in features:
        raise ContractValidationError("C1/C2 affected features must exist in feature_columns.")
    reference = _frame(paths["reference_dataset"], features, dc.target_column, "reference dataset")
    current = _frame(paths["current_dataset"], features, dc.target_column, "current dataset")
    if "batch_size" in c1_raw and (isinstance(c1_raw["batch_size"], bool) or c1_raw["batch_size"] != len(current)):
        raise ContractValidationError("C1 batch_size does not match the supplied current dataset.")
    for name, frame in (("reference", reference), ("current", current)):
        if any(not pd.api.types.is_numeric_dtype(frame[f]) for f in numeric):
            raise ContractValidationError(f"{name}: non-numeric feature missing from categorical_features.")
    if len(current) < 100 or len(reference) < 100:
        raise ContractValidationError("Reference and current datasets each require at least 100 labeled rows.")
    try:
        current_mask = evaluate_slice_rule(current[features], c2.affected_slice_rule)
    except UnsafeSliceRuleError as exc:
        raise ContractValidationError(f"Invalid C2 slice: {exc}") from exc
    observed_support = float(current_mask.mean())
    if abs(observed_support - c2.slice_support) > 0.1:
        raise ContractValidationError(f"C2 slice_support {c2.slice_support:.3f} disagrees with actual current support {observed_support:.3f}.")
    if int(current_mask.sum()) < 50 or observed_support < 0.05:
        return finish(["Insufficient actual diagnosed slice support (need 50 rows and 5%)."],
                      package_status=RepairPackageStatus.BLOCKED)

    target = dc.target_column
    # Shared records cannot leak from reference training into current evaluation.
    # Group duplicate features together when deriving internal holdouts.
    current_keys = set(_row_keys(current, features))
    independent_reference = reference.loc[~_row_keys(reference, features).isin(current_keys)]
    if len(independent_reference) < 100:
        raise ContractValidationError("Need at least 100 reference rows independent of the current batch.")
    source_train, source_eval = _split(independent_reference, target, 42)
    adaptation, evaluation = _split(current, target, 43)
    for field, destination in (("source_evaluation_uri", "source"), ("evaluation_dataset_uri", "target")):
        if options.get(field):
            path = artifact_path(options[field], "datasets")
            supplied = _frame(path, features, target, field)
            provenance["artifact_paths"][field] = str(path)
            provenance["artifact_sha256"][field] = compute_file_sha256(path)
            if field in expected_hashes and expected_hashes[field] != provenance["artifact_sha256"][field]:
                raise DependencyResolutionError(f"Artifact hash mismatch: {field}")
            if destination == "source":
                source_train, source_eval = independent_reference, supplied
            else:
                adaptation, evaluation = current, supplied
    _disjoint(source_train, source_eval, features, "source training/evaluation")
    _disjoint(adaptation, evaluation, features, "current adaptation/evaluation")
    _disjoint(source_train, evaluation, features, "source training/target evaluation")
    _disjoint(adaptation, source_eval, features, "current adaptation/source evaluation")
    if len(adaptation) > dc.adaptation_budget_size:
        try:
            adaptation, _ = train_test_split(adaptation, train_size=dc.adaptation_budget_size,
                stratify=adaptation[target], random_state=44)
            adaptation = adaptation.reset_index(drop=True)
        except ValueError as exc:
            raise ContractValidationError(f"Adaptation budget cannot support both classes: {exc}") from exc
    try:
        training_slice = adaptation.loc[evaluate_slice_rule(adaptation[features], c2.affected_slice_rule)].copy()
        evaluation_slice = evaluation.loc[evaluate_slice_rule(evaluation[features], c2.affected_slice_rule)].copy()
    except UnsafeSliceRuleError as exc:
        raise ContractValidationError(str(exc)) from exc
    if len(training_slice) < 30 or len(evaluation_slice) < 20:
        return finish(["Need at least 30 adaptation and 20 independent evaluation rows in the diagnosed slice."],
                      package_status=RepairPackageStatus.BLOCKED)

    model = load_trusted_pickle(model_path, trusted=True)
    provenance["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
    preprocessor = load_trusted_pickle(prep_path, trusted=True) if prep_path else None
    if preprocessor is None and not isinstance(model, Pipeline) and options.get("input_representation") != "model_ready":
        raise ContractValidationError("Supply preprocessing_uri, a fitted Pipeline, or input_representation='model_ready'.")
    if options.get("task_type", "binary_classification") != "binary_classification":
        raise ContractValidationError("Only tabular binary_classification is currently supported.")
    mapping = options.get("target_mapping", {})
    if not isinstance(mapping, dict):
        raise ContractValidationError("target_mapping must be an object.")
    inputs = ModelInputs(model, preprocessor, features, target, mapping, options.get("positive_label"))
    provenance["data_context"]["positive_label"] = inputs.positive.item() if hasattr(inputs.positive, "item") else inputs.positive
    metric = options.get("repair_metric", "roc_auc" if c1_raw.get("harm_metric") == "auc_drop" else "f1")
    if metric not in ("f1", "accuracy", "roc_auc"):
        raise ContractValidationError("repair_metric must be f1, accuracy, or roc_auc.")
    for frame in (source_train, source_eval, adaptation, evaluation_slice):
        inputs.y(frame)
    try:
        baseline = inputs.score(model, evaluation_slice, metric)
        source_baseline = inputs.score(model, source_eval, metric)
    except (ValueError, TypeError) as exc:
        raise ContractValidationError(f"Model/preprocessor/schema compatibility failed: {exc}") from exc

    ratio = _number(options, "augmentation_ratio", 0.3, 0.05, 1.0)
    epochs = _number(options, "generator_epochs", 15, 1, 1000, integer=True)
    max_rows = _number(options, "max_synthetic_rows", 1000, 1, 50000, integer=True)
    min_gain = _number(options, "minimum_utility_gain", 0.01, 0.001, 1.0)
    generator = options.get("generator")
    if generator is None:
        interaction = float(c1_raw.get("interaction_suspected", 0.0))
        generator = "TabDDPM" if c2.root_cause_category == "combined" else ("CTGAN" if c2.root_cause_feature in categories and interaction < 0.4 else "TVAE")
    names = {"tvae": "TVAE", "ctgan": "CTGAN", "tabddpm": "TabDDPM"}
    if not isinstance(generator, str) or generator.lower() not in names:
        raise ContractValidationError("generator must be TVAE, CTGAN, or TabDDPM.")
    generator = names[generator.lower()]
    count = min(max_rows, max(1, math.ceil(ratio * len(training_slice))))
    provenance.update({"observed_slice_support": observed_support, "metric": metric,
        "prescription_confidence": c2.diagnosis_stability,
        "prescription_policy": "Rule based: categorical root cause selects CTGAN; combined diagnosis selects TabDDPM; otherwise TVAE. An explicit generator may be requested.",
        "requested_augmentation_ratio": ratio,
        "feature_columns": features, "categorical_features": categories,
        "partition_rows": {"source_train": len(source_train), "source_eval": len(source_eval),
                           "adaptation": len(adaptation), "evaluation": len(evaluation),
                           "generator_training_slice": len(training_slice),
                           "reference_current_overlap_excluded": len(reference) - len(independent_reference)},
        "partition_seeds": {"source": 42, "current": 43, "adaptation_budget": 44},
        "evaluation_provenance": {"target": "supplied holdout" if options.get("evaluation_dataset_uri") else "derived current holdout",
                                  "source": "supplied holdout" if options.get("source_evaluation_uri") else "derived reference holdout",
                                  "original_model_training_overlap": "Must be checked by the model/data owner; C3 checks candidate-training overlap."},
        "evaluation_policy": "single candidate; independent rows; no evaluation-driven generator search"})
    log.transition_to(ExecutionState.GENERATING, f"Training {generator} on actual current diagnosed slice")
    try:
        synthetic, generation = (generator_selector or GeneratorSelector()).fit_and_sample(
            generator_name=generator, training_df=training_slice, num_samples=count,
            categorical_columns=categories + [target], numerical_columns=numeric,
            enable_fallback=False, max_retries=1, generator_kwargs={"epochs": epochs})
    except GeneratorExecutionError as exc:
        return finish([str(exc)], package_status=RepairPackageStatus.FAILED,
                      generator_info={"generator": "NONE", "requested_generator": generator})
    actual = names.get(str(generation.get("executed_generator", "")).lower())
    if actual is None:
        raise ContractValidationError("Generator did not report a supported executed_generator.")
    package = replace(package, chosen_strategy="TARGETED_SYNTHETIC_REPAIR",
                      generator_info={**generation, "generator": actual})
    reasons = []
    if set(synthetic.columns) != set(features + [target]) or len(synthetic) != count or synthetic.isna().any().any():
        return finish(["Generated data failed structural schema/missing-value checks."], package_status=RepairPackageStatus.REJECTED)
    synthetic = synthetic[features + [target]]
    try:
        for col in numeric:
            if not pd.api.types.is_numeric_dtype(synthetic[col]) or not np.isfinite(synthetic[col]).all():
                raise ValueError(f"Invalid numeric values in {col}")
        for col in categories + [target]:
            if not set(synthetic[col]).issubset(set(training_slice[col])):
                raise ValueError(f"Unknown discrete values in {col}")
        coverage = float(evaluate_slice_rule(synthetic[features], c2.affected_slice_rule).mean())
        if coverage < 0.8:
            reasons.append("Synthetic target coverage below 0.80.")
        fidelity = FidelityValidator().validate(synthetic, training_slice, numeric, categories + [target])
        # Feature-level privacy also catches rows copied with a different label.
        privacy = PrivacyValidator(min_dcr_p05=0.05).validate(synthetic[features],
            pd.concat([source_train, adaptation], ignore_index=True)[features],
            continuous_columns=numeric, discrete_columns=categories)
    except Exception as exc:
        return finish([f"Candidate validation could not complete: {exc}"], package_status=RepairPackageStatus.REJECTED)
    if fidelity.status.value != "PASS":
        reasons.extend(fidelity.reasons or ["Fidelity validation did not pass."])
    if privacy.status.value != "PASS":
        reasons.extend(privacy.reasons or ["Privacy validation did not pass."])
    report = {"structure": {"passed": True}, "target_coverage": coverage,
              "fidelity_report": fidelity.to_dict(), "privacy_report": privacy.to_dict()}
    package = replace(package, validation_report=report, fidelity_indicators=fidelity.metrics,
                      privacy_risk_indicators={"status": privacy.status.value, **privacy.metrics})
    if reasons:
        return finish(reasons, package_status=RepairPackageStatus.REJECTED)

    log.transition_to(ExecutionState.EVALUATING, "Comparing same-family candidate with baseline and real-only control")
    real_training = pd.concat([source_train, adaptation], ignore_index=True)
    augmented = pd.concat([real_training, synthetic], ignore_index=True)
    try:
        control = inputs.fit(real_training)
        candidate = inputs.fit(augmented)
        control_score = inputs.score(control, evaluation_slice, metric)
        candidate_score = inputs.score(candidate, evaluation_slice, metric)
        source_after = inputs.score(candidate, source_eval, metric)
        overall_before = inputs.score(model, evaluation, metric)
        overall_after = inputs.score(candidate, evaluation, metric)
        confidence_interval = inputs.improvement_interval(model, candidate, evaluation_slice, metric)
    except Exception as exc:
        return finish([f"Candidate retraining/evaluation failed: {exc}"], package_status=RepairPackageStatus.REJECTED)
    gain = candidate_score - baseline
    if gain < min_gain:
        reasons.append(f"Affected-slice {metric} gain {gain:+.4f} below {min_gain:.4f}.")
    if confidence_interval[0] <= 0.0:
        reasons.append("Paired bootstrap 95% interval does not establish a positive affected-slice gain.")
    if candidate_score < control_score:
        reasons.append("Synthetic augmentation underperforms the matched real-only retraining control.")
    if source_after - source_baseline < -dc.source_retention_tolerance:
        reasons.append("Source performance retention exceeded its permitted loss.")
    if overall_after - overall_before < -dc.source_retention_tolerance:
        reasons.append("Overall current performance exceeded its permitted loss.")
    utility = {"metric": metric, "baseline_score": baseline, "candidate_score": candidate_score,
               "authentic_control_score": control_score, "delta_target_f1_vs_baseline": gain,
               "utility_gain": gain, "training_rows": len(augmented),
               "bootstrap_ci_95": confidence_interval,
               "overall_before": overall_before, "overall_after": overall_after}
    report["utility"] = {"passed": not reasons, **utility}
    package = replace(package, utility_metrics=utility, validation_report=report,
        source_retention_metrics={"before": source_baseline, "after": source_after,
                                  "delta": source_after - source_baseline})
    if reasons:
        return finish(reasons, package_status=RepairPackageStatus.REJECTED)
    # Persist approved artifacts only; no stale or rejected CSV is downloadable.
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, model_path_out = out_dir / "repair.csv", out_dir / "candidate_model.joblib"
    synthetic.to_csv(csv_path, index=False)
    import joblib
    joblib.dump(candidate, model_path_out)
    return finish(package_status=RepairPackageStatus.APPROVED_FOR_REVIEW,
        candidate_dataset_path=str(csv_path), candidate_dataset_sha256=compute_file_sha256(csv_path),
        candidate_model_path=str(model_path_out), candidate_model_sha256=compute_file_sha256(model_path_out),
        quantity_selected=len(synthetic))


def response_for_package(package, log):
    """UI enrichment stays outside the exact C3-to-C4 contract."""
    from component3.src.adapters.c4_retraining_adapter import C4RetrainingAdapter
    contract = C4RetrainingAdapter().to_c4_flat_contract(package)
    metadata = package.downstream_handoff_metadata
    approved = package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    preview = pd.read_csv(package.candidate_dataset_path).head(8) if approved else pd.DataFrame()
    return {
        **contract, "run_id": package.run_id, "package_id": package.package_id,
        "reason": "; ".join(package.rejection_reasons), "execution_status": package.package_status.value,
        "validation_report": package.validation_report,
        "readiness_passed": package.package_status not in (RepairPackageStatus.BLOCKED, RepairPackageStatus.FAILED),
        "readiness_reason": "; ".join(package.rejection_reasons),
        "preview_rows": preview.to_dict(orient="records"), "preview_columns": preview.columns.tolist(),
        "download_url": f"/api/c3/packages/{package.run_id}/dataset" if approved else None,
        "upstream_evidence": {"c1": metadata.get("original_c1"), "c2": metadata.get("original_c2")},
        "provenance": metadata, "utility_metrics": package.utility_metrics,
        "source_retention_metrics": package.source_retention_metrics, "execution_log": log.to_dict(),
    }
