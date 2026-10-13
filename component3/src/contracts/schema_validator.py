"""Standalone schema validation for Component 1 and Component 2 contracts.

Validates payload structure, types, constraints, and slice rules without
depending on experiment simulation scripts or unsafe code execution.
"""

import math
from typing import Any
import pandas as pd

from component3.src.contracts.safe_slice_evaluator import (
    evaluate_slice_rule,
    UnsafeSliceRuleError,
)

DETECTOR_NAMES = ("ks", "chi2", "psi", "mmd", "lsdd", "classifier_auc")
CONFIDENCE_NAMES = ("mean_confidence", "confidence_std", "pct_low_confidence")


class SchemaValidationError(ValueError):
    """Raised when an incoming component payload violates its contract schema."""


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    return True


def _validate_fields(payload: dict, contract: dict) -> None:
    """Validate presence and basic types of all declared fields."""
    if not isinstance(payload, dict):
        _fail("Payload must be a dictionary.")

    fields = contract.get("fields", {})
    required = {
        name for name, spec in fields.items()
        if spec.get("required", False)
    }

    missing = required - payload.keys()
    if missing:
        _fail(f"Missing required fields: {sorted(missing)}")

    extra = payload.keys() - fields.keys()
    if extra:
        _fail(f"Unexpected fields: {sorted(extra)}")

    for name, spec in fields.items():
        if name not in payload:
            continue
        value = payload[name]
        declared_type = spec.get("type")

        if declared_type == "string":
            if not isinstance(value, str):
                _fail(f"{name} must be a string.")
            if "allowed_values" in spec and value not in spec["allowed_values"]:
                _fail(f"{name} must be one of {spec['allowed_values']}.")
        elif declared_type == "float":
            if not _number(value):
                _fail(f"{name} must be numeric.")
            if spec.get("range") == "0.0 to 1.0" and not 0.0 <= float(value) <= 1.0:
                _fail(f"{name} must be in range 0.0 to 1.0.")
        elif declared_type == "integer":
            if type(value) is not int:
                _fail(f"{name} must be an integer.")
            if name == "temporal_window" and value < 1:
                _fail(f"{name} must be >= 1.")
        elif declared_type == "array[string]":
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                _fail(f"{name} must be a list of strings.")
        elif declared_type == "array[float, float]":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(_number(x) for x in value)
            ):
                _fail(f"{name} must be a list of two numeric floats.")
        elif declared_type == "object":
            if not isinstance(value, dict):
                _fail(f"{name} must be an object/dict.")


def validate_c1_schema(payload: dict, contract: dict) -> None:
    """Validate Component 1 evidence against its official contract schema."""
    _validate_fields(payload, contract)

    harm = payload.get("predicted_harm")
    if not _number(harm) or not 0.0 <= float(harm) <= 1.0:
        _fail("predicted_harm must be a number between 0 and 1.")

    interval = payload.get("harm_interval")
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or not all(_number(v) and 0.0 <= float(v) <= 1.0 for v in interval)
        or interval[0] > interval[1]
    ):
        _fail("harm_interval must contain two ordered values in [0, 1].")

    scores = payload.get("detector_scores", {})
    if not isinstance(scores, dict):
        _fail("detector_scores must be a dictionary.")
    for name in DETECTOR_NAMES:
        if name not in scores or not _number(scores[name]):
            _fail(f"Invalid or missing detector score: {name}")

    confidence = payload.get("confidence_stats", {})
    if not isinstance(confidence, dict):
        _fail("confidence_stats must be a dictionary.")
    for name in CONFIDENCE_NAMES:
        if name not in confidence or not _number(confidence[name]):
            _fail(f"Invalid or missing confidence statistic: {name}")

    fired = payload.get("detectors_fired")
    if type(fired) is not int or not 0 <= fired <= len(DETECTOR_NAMES):
        _fail("detectors_fired must be an integer between 0 and 6.")

    if not isinstance(payload.get("affected_features"), list):
        _fail("affected_features must be a list.")


def validate_c2_schema(
    payload: dict,
    contract: dict,
    reference_df: pd.DataFrame,
) -> None:
    """Validate C2 diagnosis without executing pandas query strings."""
    _validate_fields(payload, contract)

    for name in (
        "slice_support",
        "diagnosis_stability",
        "drift_impact_score",
    ):
        value = payload[name]
        if not _number(value) or not 0.0 <= float(value) <= 1.0:
            _fail(f"{name} must be between 0 and 1.")

    try:
        evaluate_slice_rule(
            reference_df,
            payload["affected_slice_rule"],
        )
    except UnsafeSliceRuleError as exc:
        _fail(f"Invalid affected_slice_rule: {exc}")
