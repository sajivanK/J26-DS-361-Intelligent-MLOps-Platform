"""Validation models and status enumerations for Component 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class ValidationStageResult:
    stage: str
    status: ValidationStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "reasons": list(self.reasons),
        }
