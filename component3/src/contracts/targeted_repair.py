"""Safe targeted repair contract and diagnostic payload validation.

Enforces constrained AST predicate evaluation, non-oracle diagnostic integrity,
and automated fallback to TARGETED_REPAIR_BLOCKED when legitimate diagnosis is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import pandas as pd

from component3.src.contracts.safe_slice_evaluator import (
    UnsafeSliceRuleError,
    evaluate_slice_rule,
)


class DiagnosticProvenance(str, Enum):
    """Provenance origin of diagnostic slice definitions."""
    COMPONENT_2_DIAGNOSIS = "COMPONENT_2_DIAGNOSIS"
    MONITORING_DETECTOR = "MONITORING_DETECTOR"
    SIMULATION_ORACLE = "SIMULATION_ORACLE"
    USER_MANUAL = "USER_MANUAL"
    UNSPECIFIED = "UNSPECIFIED"


class TargetedRepairStatus(str, Enum):
    """Execution status for targeted synthetic repair."""
    ELIGIBLE = "ELIGIBLE"
    TARGETED_REPAIR_BLOCKED = "TARGETED_REPAIR_BLOCKED"
    INVALID_PREDICATE = "INVALID_PREDICATE"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"


@dataclass(frozen=True)
class DiagnosticSliceDefinition:
    """Constrained, auditable specification of a drifted subpopulation slice."""
    slice_id: str
    slice_predicate: str
    affected_features: tuple[str, ...]
    reported_slice_support: float
    diagnostic_confidence: float = 1.0
    provenance: DiagnosticProvenance = DiagnosticProvenance.COMPONENT_2_DIAGNOSIS
    is_non_oracle: bool = True

    def __post_init__(self) -> None:
        if not self.slice_id or not self.slice_id.strip():
            raise ValueError("Diagnostic slice_id must be a non-empty string.")
        if not self.slice_predicate or not self.slice_predicate.strip():
            raise ValueError("Diagnostic slice_predicate must be a non-empty string.")
        if not (0.0 <= self.reported_slice_support <= 1.0):
            raise ValueError(f"Slice support must be in [0.0, 1.0], got {self.reported_slice_support}")
        if not (0.0 <= self.diagnostic_confidence <= 1.0):
            raise ValueError(f"Diagnostic confidence must be in [0.0, 1.0], got {self.diagnostic_confidence}")

    def evaluate_mask(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate slice predicate safely using AST evaluator without eval() or exec()."""
        try:
            return evaluate_slice_rule(df, self.slice_predicate)
        except UnsafeSliceRuleError as exc:
            raise ValueError(f"Unsafe or invalid slice predicate '{self.slice_predicate}': {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "slice_predicate": self.slice_predicate,
            "affected_features": list(self.affected_features),
            "reported_slice_support": self.reported_slice_support,
            "diagnostic_confidence": self.diagnostic_confidence,
            "provenance": self.provenance.value,
            "is_non_oracle": self.is_non_oracle,
        }


def validate_diagnostic_slice(
    diag_slice: DiagnosticSliceDefinition | None,
    current_df: pd.DataFrame | None = None,
    min_slice_rows: int = 10,
) -> tuple[TargetedRepairStatus, str, pd.Series | None]:
    """Validate diagnostic slice eligibility under non-oracle contracts."""
    if diag_slice is None:
        return (
            TargetedRepairStatus.TARGETED_REPAIR_BLOCKED,
            "No diagnostic slice payload provided by Components 1 and 2.",
            None,
        )

    # Check non-oracle contract
    if not diag_slice.is_non_oracle or diag_slice.provenance == DiagnosticProvenance.SIMULATION_ORACLE:
        return (
            TargetedRepairStatus.TARGETED_REPAIR_BLOCKED,
            f"Diagnostic slice '{diag_slice.slice_id}' is marked as an oracle simulation rule. "
            "Targeted synthetic repair cannot use oracle simulation rules.",
            None,
        )

    if diag_slice.diagnostic_confidence < 0.5:
        return (
            TargetedRepairStatus.TARGETED_REPAIR_BLOCKED,
            f"Diagnostic confidence ({diag_slice.diagnostic_confidence:.2f}) is below acceptable threshold (0.50).",
            None,
        )

    if current_df is not None:
        try:
            mask = diag_slice.evaluate_mask(current_df)
            matched_rows = int(mask.sum())
            if matched_rows < min_slice_rows:
                return (
                    TargetedRepairStatus.INSUFFICIENT_SUPPORT,
                    f"Slice predicate matched only {matched_rows} rows in available data (minimum {min_slice_rows}).",
                    mask,
                )
            return TargetedRepairStatus.ELIGIBLE, "Diagnostic slice is eligible for targeted repair.", mask
        except Exception as exc:
            return (
                TargetedRepairStatus.INVALID_PREDICATE,
                f"Failed to evaluate slice predicate on dataset: {exc}",
                None,
            )

    return TargetedRepairStatus.ELIGIBLE, "Diagnostic slice definition is syntactically valid.", None
