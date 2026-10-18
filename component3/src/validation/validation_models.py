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


class AcceptanceDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


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


@dataclass(frozen=True)
class RepairPackageDecision:
    decision: AcceptanceDecision
    stage_results: dict[str, ValidationStageResult]
    reasons: tuple[str, ...] = ()
    supervised_retraining_approved: bool = False
    package_status: str = "REJECTED"  # "APPROVED" or "REJECTED" per C3/C4 contract

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "package_status": self.package_status,
            "supervised_retraining_approved": self.supervised_retraining_approved,
            "reasons": list(self.reasons),
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
        }
