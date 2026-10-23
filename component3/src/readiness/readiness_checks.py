"""Independent readiness checks for Component 3."""

import math
from typing import Any, Optional


def _is_valid_row_count(value: Any) -> bool:
    """Accept only non-negative Python integer row counts."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_valid_minimum_rows(value: Any) -> bool:
    """Accept only positive Python integer thresholds."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
    )


def check_data_sufficiency(
    reference_rows: int,
    current_rows: int,
    min_reference_rows: int,
    min_current_rows: int,
) -> dict[str, Any]:
    """Safely check reference and current dataset row sufficiency."""
    reference_ok = (
        _is_valid_row_count(reference_rows)
        and _is_valid_minimum_rows(min_reference_rows)
        and reference_rows >= min_reference_rows
    )
    current_ok = (
        _is_valid_row_count(current_rows)
        and _is_valid_minimum_rows(min_current_rows)
        and current_rows >= min_current_rows
    )
    return {
        "passed": reference_ok and current_ok,
        "reference_ok": reference_ok,
        "current_ok": current_ok,
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "min_reference_rows": min_reference_rows,
        "min_current_rows": min_current_rows,
    }


def check_slice_sufficiency(
    affected_slice_rows: int,
    min_affected_slice_rows: int,
) -> dict[str, Any]:
    """Safely check whether the diagnosed affected slice is large enough."""
    passed = (
        _is_valid_row_count(affected_slice_rows)
        and _is_valid_minimum_rows(min_affected_slice_rows)
        and affected_slice_rows >= min_affected_slice_rows
    )
    return {
        "passed": passed,
        "affected_slice_rows": affected_slice_rows,
        "min_affected_slice_rows": min_affected_slice_rows,
    }



def check_diagnosis_stability(
    diagnosis_stability: Optional[float],
    min_diagnosis_stability: float,
) -> dict[str, Any]:
    """Validate diagnosis stability and its configured threshold."""
    def valid_stability(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )

    score_valid = valid_stability(diagnosis_stability)
    threshold_valid = valid_stability(min_diagnosis_stability)

    passed = (
        score_valid
        and threshold_valid
        and diagnosis_stability >= min_diagnosis_stability
    )

    return {
        "passed": bool(passed),
        "diagnosis_stability": (
            float(diagnosis_stability)
            if score_valid
            else diagnosis_stability
        ),
        "min_diagnosis_stability": min_diagnosis_stability,
    }



def check_label_trust(
    label_provenance: Optional[dict[str, Any]],
    require_trusted_labels: bool,
    allow_unlabeled_current_data: bool,
) -> dict[str, Any]:
    """Assess label eligibility without treating configuration as proof.
    Supervised retraining requires verified, trusted ground-truth
    labels regardless of configuration.
    """
    provenance_available = isinstance(label_provenance, dict)
    labels_available = (
        provenance_available
        and label_provenance.get("labels_available") is True
    )
    trusted = (
        labels_available
        and label_provenance.get("trusted") is True
        and label_provenance.get("source_verified") is True
    )

    # A configuration switch cannot establish ground-truth trust.
    supervised_ready = bool(trusted)

    # Unverified labels may still support exploratory preparation,
    # but they are not authorized for supervised retraining.
    generation_allowed = bool(
        labels_available or allow_unlabeled_current_data
    )

    return {
        "passed": generation_allowed,
        "provenance_available": provenance_available,
        "labels_available": labels_available,
        "trusted": trusted,
        "generation_allowed": generation_allowed,
        "supervised_retraining_ready": supervised_ready,
    }

