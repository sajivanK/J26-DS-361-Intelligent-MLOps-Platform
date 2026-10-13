"""Component 3 Unified Input Contract Validator.

Validates Component 1 drift evidence and Component 2
diagnosis specifications before Component 3 processing.

Contract ownership:
- C1: Component 1 team
- C2: Component 2 team

This module does not modify shared contracts.
"""

import json
from pathlib import Path
import pandas as pd

from component3.src.contracts.safe_slice_evaluator import (
    evaluate_slice_rule,
    UnsafeSliceRuleError,
)
from component3.src.contracts.schema_validator import (
    validate_c1_schema,
    validate_c2_schema,
    SchemaValidationError,
)

ROOT = Path(__file__).resolve().parents[3]
C1_CONTRACT_PATH = ROOT / "shared/contracts/c1_output_schema.json"
C2_CONTRACT_PATH = ROOT / "shared/contracts/c2_output_to_c3_schema.json"


class InputContractError(ValueError):
    """Raised when an incoming component payload is invalid."""


def load_json(path: Path) -> dict:
    """Load and parse JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_contracts() -> tuple[dict, dict]:
    """Load both official C1 and C2 contract schemas."""
    c1 = load_json(C1_CONTRACT_PATH)
    c2 = load_json(C2_CONTRACT_PATH)
    return c1, c2


def validate_c1_input(payload: dict, contract: dict) -> dict:
    """Validate C1 evidence using the standalone schema validator."""
    try:
        validate_c1_schema(payload, contract)
    except SchemaValidationError as exc:
        raise InputContractError(
            f"Invalid Component 1 evidence: {exc}"
        ) from exc
    return {
        "component": "C1",
        "valid": True,
        "field_count": len(payload),
        "message": "C1 evidence passed contract validation.",
    }


def validate_c2_input(
    payload: dict,
    contract: dict,
    reference_df: pd.DataFrame,
) -> dict:
    """Validate C2 diagnosis using the standalone schema validator."""
    try:
        validate_c2_schema(
            payload=payload,
            contract=contract,
            reference_df=reference_df,
        )
    except SchemaValidationError as exc:
        raise InputContractError(
            f"Invalid Component 2 diagnosis: {exc}"
        ) from exc
    return {
        "component": "C2",
        "valid": True,
        "field_count": len(payload),
        "message": "C2 diagnosis passed contract validation.",
    }


def validate_cross_component_consistency(
    c1_payload: dict,
    c2_payload: dict,
    current_df: pd.DataFrame,
) -> dict:
    """Check consistency between valid C1 and C2 payloads.

    Hard errors indicate an unusable input combination.
    Warnings indicate differences requiring further review.
    Run individual C1/C2 contract validation first.
    """
    errors = []
    warnings = []

    drift_type = c1_payload.get("drift_type", "unknown")
    category = c2_payload.get("root_cause_category", "unknown")

    expected_categories = {
        "none": "unknown",
        "marginal_single": "marginal",
        "marginal_multi": "marginal",
        "interaction": "interaction",
        "combined": "combined",
    }

    expected_category = expected_categories.get(drift_type, "unknown")
    if category != expected_category:
        warnings.append(
            f"C1 drift type '{drift_type}' maps to '{expected_category}', "
            f"but C2 reports '{category}'."
        )

    # Feature consistency check
    c1_features = set(c1_payload.get("affected_features", []))
    c2_root_cause = c2_payload.get("root_cause_feature")
    c2_ranked = set(c2_payload.get("affected_features_ranked", []))

    if drift_type != "none":
        if c2_root_cause and c2_root_cause not in c1_features:
            warnings.append(
                f"C2 root_cause_feature '{c2_root_cause}' not listed in "
                f"C1 affected_features: {sorted(c1_features)}."
            )
        extra_c2 = c2_ranked - c1_features
        if extra_c2:
            warnings.append(
                f"C2 ranked features contain features not in C1: {sorted(extra_c2)}."
            )
    else:
        # For drift_type == none, C2 should report slice_support == 0.0
        if c2_payload.get("slice_support", 0.0) > 0.0:
            warnings.append(
                "C1 detected no drift, but C2 reports non-zero slice support."
            )

    # Slice support validation using the restricted AST evaluator.
    slice_rule = c2_payload.get("affected_slice_rule", "")
    try:
        mask = evaluate_slice_rule(current_df, slice_rule)
        actual_support = float(mask.mean())
        reported_support = c2_payload.get("slice_support")
        if (
            not isinstance(reported_support, (int, float))
            or isinstance(reported_support, bool)
            or not 0.0 <= reported_support <= 1.0
        ):
            errors.append(
                "C2 slice_support must be a number between 0 and 1."
            )
        elif abs(reported_support - actual_support) > 1e-5:
            errors.append(
                f"C2 slice_support ({reported_support:.6f}) "
                f"does not match actual current data support "
                f"({actual_support:.6f})."
            )
        if drift_type != "none" and not mask.any():
            warnings.append(
                "C2 affected_slice_rule matched 0 records "
                "in current drifted batch."
            )
    except UnsafeSliceRuleError as exc:
        errors.append(
            f"Invalid or unsafe affected_slice_rule: {exc}"
        )
    except Exception as exc:
        errors.append(
            f"affected_slice_rule failed execution on current dataset: {exc}"
        )

    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "warning_count": len(warnings),
    }



