"""Repair Readiness Engine for Component 3.

Evaluates incoming C1 drift evidence, C2 root-cause diagnosis, dataset
sufficiency, diagnosis stability, and label provenance to determine if
synthetic repair generation and supervised retraining are ready.
"""

import math
from typing import Any, Optional
import pandas as pd

from component3.src.contracts.input_validator import (
    InputContractError,
    load_contracts,
    validate_c1_input,
    validate_c2_input,
    validate_cross_component_consistency,
)
from component3.src.contracts.safe_slice_evaluator import (
    UnsafeSliceRuleError,
    extract_affected_slice,
)
from component3.src.readiness.readiness_checks import (
    check_data_sufficiency,
    check_diagnosis_stability,
    check_label_trust,
    check_slice_sufficiency,
)
from component3.src.readiness.readiness_models import (
    ReadinessDecision,
    ReadinessReason,
    ReadinessStatus,
)


def assess_repair_readiness(
    c1_payload: dict[str, Any],
    c2_payload: dict[str, Any],
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: dict[str, Any],
    label_provenance: Optional[dict[str, Any]] = None,
    c1_contract: Optional[dict[str, Any]] = None,
    c2_contract: Optional[dict[str, Any]] = None,
) -> ReadinessDecision:
    """Assess overall repair readiness across 10 sequential governance gates."""
    reasons: list[ReadinessReason] = []
    warnings: list[str] = []

    reference_rows = len(reference_df) if isinstance(reference_df, pd.DataFrame) else 0
    current_rows = len(current_df) if isinstance(current_df, pd.DataFrame) else 0
    affected_slice_rows = 0

    stability = (
        float(c2_payload["diagnosis_stability"])
        if isinstance(c2_payload, dict)
        and "diagnosis_stability" in c2_payload
        and isinstance(c2_payload["diagnosis_stability"], (int, float))
        and not isinstance(c2_payload["diagnosis_stability"], bool)
        else None
    )

    def decision(
        status: ReadinessStatus,
        generation_ready: bool = False,
        supervised_ready: bool = False,
    ) -> ReadinessDecision:
        return ReadinessDecision(
            status=status,
            reason_codes=tuple(reasons),
            reference_rows=reference_rows,
            current_rows=current_rows,
            affected_slice_rows=affected_slice_rows,
            diagnosis_stability=stability,
            generation_ready=generation_ready,
            supervised_retraining_ready=supervised_ready,
            warnings=tuple(warnings),
        )

    # Load contracts if not provided
    if c1_contract is None or c2_contract is None:
        try:
            loaded_c1, loaded_c2 = load_contracts()
            c1_contract = c1_contract or loaded_c1
            c2_contract = c2_contract or loaded_c2
        except Exception as exc:
            reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
            warnings.append(f"Failed to load integration contracts: {exc}")
            return decision(ReadinessStatus.BLOCKED)

    # Gate 0: Validate readiness configuration before applying thresholds.
    if not isinstance(config, dict):
        reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
        warnings.append("Configuration must be a dictionary.")
        return decision(ReadinessStatus.BLOCKED)

    settings = config.get("readiness")
    if not isinstance(settings, dict):
        reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
        warnings.append(
            "Missing or invalid 'readiness' configuration section."
        )
        return decision(ReadinessStatus.BLOCKED)

    required_settings = (
        "min_reference_rows",
        "min_current_rows",
        "min_affected_slice_rows",
        "min_diagnosis_stability",
        "require_trusted_labels",
        "allow_unlabeled_current_data",
    )
    missing_settings = [
        key for key in required_settings if key not in settings
    ]
    if missing_settings:
        reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
        warnings.append(
            f"Missing readiness settings: {missing_settings}"
        )
        return decision(ReadinessStatus.BLOCKED)

    # Gate 0 continued: Validate configuration value types and ranges.
    integer_settings = (
        "min_reference_rows",
        "min_current_rows",
        "min_affected_slice_rows",
    )
    for key in integer_settings:
        value = settings[key]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
            warnings.append(
                f"Invalid readiness setting '{key}': "
                "expected a positive integer."
            )
            return decision(ReadinessStatus.BLOCKED)

    stability_threshold = settings["min_diagnosis_stability"]
    if (
        not isinstance(stability_threshold, (int, float))
        or isinstance(stability_threshold, bool)
        or not math.isfinite(stability_threshold)
        or not 0.0 <= stability_threshold <= 1.0
    ):
        reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
        warnings.append(
            "Invalid readiness setting 'min_diagnosis_stability': "
            "expected a finite number between 0 and 1."
        )
        return decision(ReadinessStatus.BLOCKED)

    boolean_settings = (
        "require_trusted_labels",
        "allow_unlabeled_current_data",
    )
    for key in boolean_settings:
        if not isinstance(settings[key], bool):
            reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
            warnings.append(
                f"Invalid readiness setting '{key}': "
                "expected a boolean."
            )
            return decision(ReadinessStatus.BLOCKED)

    min_reference = settings["min_reference_rows"]
    min_current = settings["min_current_rows"]
    min_slice = settings["min_affected_slice_rows"]
    min_stability = settings["min_diagnosis_stability"]
    require_trust = settings["require_trusted_labels"]
    allow_unlabeled = settings["allow_unlabeled_current_data"]

    # Gate 1: Check schema validity of C1 and C2 inputs.
    try:
        validate_c1_input(c1_payload, c1_contract)
        validate_c2_input(c2_payload, c2_contract, reference_df)
    except (InputContractError, Exception) as exc:
        reasons.append(ReadinessReason.INVALID_INPUT_CONTRACT)
        warnings.append(str(exc))
        return decision(ReadinessStatus.BLOCKED)

    # Gate 2: Check cross-component consistency.
    # Fail safely if validation raises an unexpected exception.
    try:
        consistency = validate_cross_component_consistency(
            c1_payload, c2_payload, current_df
        )
        if not isinstance(consistency, dict):
            raise ValueError(
                "Cross-component validator returned an invalid result."
            )
        if not isinstance(consistency.get("valid"), bool):
            raise ValueError(
                "Cross-component validation result must contain "
                "a boolean 'valid' field."
            )
        for field_name in ("errors", "warnings"):
            messages = consistency.get(field_name, [])
            if (
                not isinstance(messages, list)
                or not all(isinstance(item, str) for item in messages)
            ):
                raise ValueError(
                    f"Cross-component '{field_name}' must be a list of strings."
                )
    except Exception as exc:
        reasons.append(ReadinessReason.CROSS_COMPONENT_INCONSISTENCY)
        warnings.append(
            f"Cross-component validation failed: {type(exc).__name__}: {exc}"
        )
        return decision(ReadinessStatus.BLOCKED)

    if not consistency["valid"]:
        reasons.append(ReadinessReason.CROSS_COMPONENT_INCONSISTENCY)
        warnings.extend(consistency.get("errors", []))
        return decision(ReadinessStatus.BLOCKED)

    warnings.extend(consistency.get("warnings", []))

    # Gate 3: Check if drift was detected.
    drift_type = c1_payload.get("drift_type", "none")
    if drift_type == "none":
        reasons.append(ReadinessReason.NO_DRIFT_DETECTED)
        return decision(ReadinessStatus.BLOCKED)

    # Gate 4: Check dataset sample sufficiency.
    data_result = check_data_sufficiency(
        reference_rows=reference_rows,
        current_rows=current_rows,
        min_reference_rows=min_reference,
        min_current_rows=min_current,
    )
    if not data_result["passed"]:
        if not data_result["reference_ok"]:
            reasons.append(ReadinessReason.INSUFFICIENT_REFERENCE_ROWS)
        if not data_result["current_ok"]:
            reasons.append(ReadinessReason.INSUFFICIENT_CURRENT_ROWS)
        reasons.append(ReadinessReason.INSUFFICIENT_DATA_SUPPORT)
        return decision(ReadinessStatus.BLOCKED)

    # Gate 5: Safely extract the affected slice.
    slice_rule = c2_payload.get("affected_slice_rule", "")
    try:
        affected_df = extract_affected_slice(current_df, slice_rule)
        affected_slice_rows = len(affected_df)
    except (UnsafeSliceRuleError, Exception) as exc:
        reasons.append(ReadinessReason.SLICE_EXTRACTION_FAILED)
        warnings.append(f"Slice extraction error: {exc}")
        return decision(ReadinessStatus.BLOCKED)

    # Gate 6: Check affected slice sufficiency.
    slice_result = check_slice_sufficiency(
        affected_slice_rows=affected_slice_rows,
        min_affected_slice_rows=min_slice,
    )
    if not slice_result["passed"]:
        reasons.append(ReadinessReason.INSUFFICIENT_AFFECTED_SLICE)
        return decision(ReadinessStatus.BLOCKED)

    # Gate 7: Check diagnosis reliability.
    stability_result = check_diagnosis_stability(
        stability,
        min_stability,
    )
    if not stability_result["passed"]:
        reasons.append(ReadinessReason.LOW_DIAGNOSIS_STABILITY)
        return decision(ReadinessStatus.REVIEW)

    # Gate 8: Check label provenance.
    label_result = check_label_trust(
        label_provenance,
        require_trusted_labels=require_trust,
        allow_unlabeled_current_data=allow_unlabeled,
    )
    if not label_result["generation_allowed"]:
        reasons.append(ReadinessReason.UNTRUSTED_LABELS)
        return decision(ReadinessStatus.BLOCKED)

    # Gate 9: Warnings require human review.
    if consistency["warnings"]:
        reasons.append(ReadinessReason.CROSS_COMPONENT_INCONSISTENCY)
        return decision(ReadinessStatus.REVIEW)

    # Gate 10: Separate generation from supervised retraining.
    if not label_result["supervised_retraining_ready"]:
        if not label_result["provenance_available"]:
            reasons.append(ReadinessReason.LABEL_PROVENANCE_UNKNOWN)
        else:
            reasons.append(ReadinessReason.UNTRUSTED_LABELS)
        return decision(
            ReadinessStatus.REVIEW,
            generation_ready=True,
            supervised_ready=False,
        )

    # All readiness gates passed.
    return decision(
        ReadinessStatus.READY,
        generation_ready=True,
        supervised_ready=True,
    )
