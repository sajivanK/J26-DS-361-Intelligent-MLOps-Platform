"""Orchestrator state models and execution telemetry for Component 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionState(str, Enum):
    """Execution state machine for Component 3 repair orchestrator."""
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PRESCRIBING = "PRESCRIBING"
    GENERATING = "GENERATING"
    EVALUATING = "EVALUATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass
class StateTransition:
    """Record of a state machine transition."""
    from_state: str
    to_state: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp_utc": self.timestamp_utc,
            "detail": self.detail,
        }


@dataclass
class OrchestratorExecutionLog:
    """Audit log capturing the complete execution trajectory of a repair run."""
    run_id: str
    request_id: str
    current_state: ExecutionState
    transitions: list[StateTransition] = field(default_factory=list)
    stage_durations: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    package_path: str | None = None

    def transition_to(self, new_state: ExecutionState, detail: str = "") -> None:
        """Record state machine transition."""
        trans = StateTransition(
            from_state=self.current_state.value,
            to_state=new_state.value,
            detail=detail,
        )
        self.transitions.append(trans)
        self.current_state = new_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "current_state": self.current_state.value,
            "transitions": [t.to_dict() for t in self.transitions],
            "stage_durations": dict(self.stage_durations),
            "error_message": self.error_message,
            "package_path": self.package_path,
        }
