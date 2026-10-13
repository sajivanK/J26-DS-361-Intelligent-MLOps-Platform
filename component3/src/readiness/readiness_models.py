"""Structured decision models for Component 3 repair readiness."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class ReadinessStatus(str, Enum):
    """Overall repair readiness outcome."""
    READY = "READY"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"


class ReadinessReason(str, Enum):
    """Machine-readable reasons for a readiness decision."""
    INVALID_INPUT_CONTRACT = "INVALID_INPUT_CONTRACT"
    INSUFFICIENT_DATA_SUPPORT = "INSUFFICIENT_DATA_SUPPORT"
    INSUFFICIENT_REFERENCE_ROWS = "INSUFFICIENT_REFERENCE_ROWS"
    INSUFFICIENT_CURRENT_ROWS = "INSUFFICIENT_CURRENT_ROWS"
    INSUFFICIENT_AFFECTED_SLICE = "INSUFFICIENT_AFFECTED_SLICE"
    LOW_DIAGNOSIS_STABILITY = "LOW_DIAGNOSIS_STABILITY"
    UNTRUSTED_LABELS = "UNTRUSTED_LABELS"
    LABEL_PROVENANCE_UNKNOWN = "LABEL_PROVENANCE_UNKNOWN"
    NO_DRIFT_DETECTED = "NO_DRIFT_DETECTED"
    SLICE_EXTRACTION_FAILED = "SLICE_EXTRACTION_FAILED"
    CROSS_COMPONENT_INCONSISTENCY = "CROSS_COMPONENT_INCONSISTENCY"


@dataclass(frozen=True)
class ReadinessDecision:
    """Immutable, structured output from the readiness engine."""
    status: ReadinessStatus
    reason_codes: tuple[ReadinessReason, ...] = field(default_factory=tuple)
    reference_rows: int = 0
    current_rows: int = 0
    affected_slice_rows: int = 0
    diagnosis_stability: Optional[float] = None
    generation_ready: bool = False
    supervised_retraining_ready: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation of the decision."""
        result = asdict(self)
        result["status"] = self.status.value
        result["reason_codes"] = [
            reason.value for reason in self.reason_codes
        ]
        result["warnings"] = list(self.warnings)
        return result
