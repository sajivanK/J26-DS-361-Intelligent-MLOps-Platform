"""Explicit operational candidate training/evaluation; no forecasting or release.

This path consumes processed labelled inputs with reviewed operator bindings.
It is separate from frozen benchmark execution and never opens future data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json

import numpy as np
import pandas as pd

from c4.contracts.platform_adapter import AdaptedPlatformInput
from c4.contracts.runtime_compatibility import RuntimeBindings, verify_runtime_compatibility
from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.models.champion import ChampionConfig, train_champion
from c4.models.evaluation import evaluate_pre_release
from c4.scenarios.diagnosis import diagnosed_slice_mask


def _load_model(contents):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(bytearray(contents))
    return model


def _model_bytes(model):
    return bytes(model.get_booster().save_raw(raw_format="ubj"))


def _valid_model(model):
    if model.get_booster().feature_names != list(FEATURE_COLUMNS):
        raise ValueError("Model runtime feature schema differs from Adult configuration.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Model runtime classes must be [0, 1].")


def _frame(contents):
    return pd.read_csv(io.BytesIO(contents)).loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()


@dataclass(frozen=True)
class RuntimeCandidateResult:
    candidate_model_bytes: bytes
    report: dict


def evaluate_runtime_candidate(
    adapted: AdaptedPlatformInput,
    bindings: RuntimeBindings,
    *,
    execute_training: bool = False,
) -> RuntimeCandidateResult:
    """Fit a fresh candidate after explicit opt-in and verified binding checks.

    No artifact directories are written. The caller must separately persist
    returned candidate bytes and report under a new version before further use.
    'model_runtime_validated' refers to schema, class/probability checks and
    serialization replay; it is not a safety or production certification.
    """
    if execute_training is not True:
        raise ValueError("Candidate training requires execute_training=True.")
    if (bindings.feature_columns != tuple(FEATURE_COLUMNS)
            or bindings.target_column != TARGET_COLUMN):
        raise ValueError("Configured schema must match the existing Adult trainer.")
    checked = verify_runtime_compatibility(adapted, bindings)
    snapshots = checked.verified_bytes
    metadata = json.loads(snapshots["champion_metadata"].decode("utf-8-sig"))
    config = ChampionConfig(**metadata["training_config"])
    champion = _load_model(snapshots["champion_model"])
    _valid_model(champion)
    reference = _frame(snapshots["reference"])
    repair = _frame(snapshots["repair"])
    release = _frame(snapshots["release_validation"])
    lineage = json.loads(snapshots["lineage"].decode("utf-8-sig"))
    # Preserve source-qualified identity for the current labelled validation set.
    release.index = lineage["release_validation_row_ids"]
    mask = diagnosed_slice_mask(release, adapted.inputs.c2_diagnosis,
                                rule_feature_space=bindings.rule_feature_space)
    if (not isinstance(mask, pd.Series) or not mask.index.equals(release.index)
            or not pd.api.types.is_bool_dtype(mask.dtype) or mask.isna().any()
            or not mask.any() or mask.all()):
        raise ValueError("Diagnosed slice and complement require valid aligned nonempty populations.")
    # Validate champion outputs before spending time on training.
    evaluate_pre_release(champion, champion, release, mask)
    training = pd.concat([reference, repair], ignore_index=True)
    candidate = train_champion(training, config)
    _valid_model(candidate)
    report = evaluate_pre_release(champion, candidate, release, mask)
    candidate_bytes = _model_bytes(candidate)
    restored = _load_model(candidate_bytes)
    _valid_model(restored)
    restored_report = evaluate_pre_release(champion, restored, release, mask)
    sample = release.loc[:, list(FEATURE_COLUMNS)]
    if (not np.array_equal(candidate.predict(sample), restored.predict(sample))
            or not np.array_equal(candidate.predict_proba(sample), restored.predict_proba(sample))
            or report != restored_report):
        raise ValueError("Candidate serialization replay differs.")
    return RuntimeCandidateResult(candidate_bytes, {
        "schema_version": "1.0",
        "status": "CANDIDATE_EVALUATED_RELEASE_PENDING",
        "run_id": adapted.context.run_id,
        "dataset": adapted.context.dataset,
        "champion_model_id": adapted.context.model_id,
        "candidate_model_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "input_sha256": {name: hashlib.sha256(value).hexdigest()
                         for name, value in snapshots.items()},
        "upstream_evidence": adapted.audit_envelope,
        "training_config": asdict(config),
        "training_rows": len(training), "reference_rows": len(reference),
        "repair_rows": len(repair), "validation_rows": len(release),
        "candidate_training_executed": True,
        "candidate_reload_verified": True,
        "model_runtime_validated": True,
        "declared_lineage_checked": True,
        "pre_release_population_basis": "C2_diagnosed_slice_on_current_labelled_validation",
        "pre_release_evaluation": report,
        "forecast_applicable": False,
        "forecast_reason": checked.forecast_reason,
        "N1_executed": False, "N2_executed": False,
        "future_outcomes_read": False,
        "validation_used_for_training": False,
        "artifact_persistence_completed": False,
        "production_release_authorized": False,
    })
