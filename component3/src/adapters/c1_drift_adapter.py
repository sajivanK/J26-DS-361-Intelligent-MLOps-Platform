"""Component 1 Drift Detection Integration Adapter for Component 3.

Ingests drift monitoring payloads from Component 1 and maps them into validated
RepairRequest contracts with schema validation and provisional contract tracking.

PROVISIONAL ADAPTER NOTICE:
This adapter establishes a typed integration boundary between Component 1 (Drift Detection)
and Component 3 (Repair Engine). Field mappings reflect agreed architectural contracts
and will be finalized upon Component 1 interface freeze.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any
import uuid

from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    InvalidRepairRequestError,
    RepairObjective,
    RepairRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class C1DriftPayload:
    """Incoming contract schema from Component 1 (Drift Detection)."""
    drift_detected: bool
    drift_type: str  # e.g. "CONCEPT_DRIFT", "COVARIATE_SHIFT", "SUBPOPULATION_SHIFT"
    severity_score: float  # [0.0, 1.0]
    affected_features: tuple[str, ...] = ()
    monitoring_window_metadata: dict[str, Any] = field(default_factory=dict)
    evidence_provenance: dict[str, Any] = field(default_factory=dict)
    payload_id: str = field(default_factory=lambda: f"C1-EV-{uuid.uuid4().hex[:8].upper()}")
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_features"] = list(self.affected_features)
        return d


def normalize_c1_drift_type(raw_type: str) -> DriftType:
    """Map arbitrary string representations of drift to DriftType enum."""
    clean = str(raw_type).strip().upper()
    mapping = {
        "CONCEPT": DriftType.CONCEPT_DRIFT,
        "CONCEPT_DRIFT": DriftType.CONCEPT_DRIFT,
        "COVARIATE": DriftType.COVARIATE_SHIFT,
        "COVARIATE_SHIFT": DriftType.COVARIATE_SHIFT,
        "DATA_DRIFT": DriftType.COVARIATE_SHIFT,
        "SUBPOPULATION": DriftType.SUBPOPULATION_SHIFT,
        "SUBPOPULATION_SHIFT": DriftType.SUBPOPULATION_SHIFT,
        "NONE": DriftType.NONE,
        "NO_DRIFT": DriftType.NONE,
    }
    return mapping.get(clean, DriftType.UNKNOWN)


def normalize_c1_severity(score: float) -> DriftSeverity:
    """Classify numerical drift severity score into DriftSeverity category."""
    if score < 0.10:
        return DriftSeverity.NONE
    if score < 0.35:
        return DriftSeverity.LOW
    if score < 0.65:
        return DriftSeverity.MEDIUM
    if score < 0.85:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


class C1DriftAdapter:
    """Adapter transforming C1 drift payloads into validated Component 3 RepairRequests."""

    def __init__(self, default_dataset_id: str = "telco_churn") -> None:
        self.default_dataset_id = default_dataset_id

    def adapt(
        self,
        payload: dict[str, Any] | C1DriftPayload,
        dataset_id: str | None = None,
        adaptation_budget_size: int = 200,
        has_authentic_adaptation_labels: bool = True,
        repair_objective: RepairObjective = RepairObjective.BALANCED_RETENTION,
    ) -> RepairRequest:
        """Validate and transform C1 payload into a RepairRequest."""
        if isinstance(payload, dict):
            # Parse dict into C1DriftPayload
            try:
                features = tuple(payload.get("affected_features", []))
                c1_obj = C1DriftPayload(
                    drift_detected=bool(payload.get("drift_detected", False)),
                    drift_type=str(payload.get("drift_type", "NONE")),
                    severity_score=float(payload.get("severity_score", 0.0)),
                    affected_features=features,
                    monitoring_window_metadata=dict(payload.get("monitoring_window_metadata", {})),
                    evidence_provenance=dict(payload.get("evidence_provenance", {})),
                    payload_id=str(payload.get("payload_id", f"C1-EV-{uuid.uuid4().hex[:8].upper()}")),
                    timestamp_utc=str(payload.get("timestamp_utc", datetime.now(timezone.utc).isoformat())),
                )
            except Exception as exc:
                raise InvalidRepairRequestError(f"Malformed C1 drift payload: {exc}") from exc
        else:
            c1_obj = payload

        # Score sanity check
        score = max(0.0, min(1.0, float(c1_obj.severity_score)))
        drift_type = normalize_c1_drift_type(c1_obj.drift_type)
        severity = normalize_c1_severity(score) if c1_obj.drift_detected else DriftSeverity.NONE

        active_dataset_id = dataset_id or self.default_dataset_id

        request = RepairRequest(
            request_id=f"REQ-FROM-{c1_obj.payload_id}",
            dataset_id=active_dataset_id,
            drift_detected=c1_obj.drift_detected,
            drift_type=drift_type,
            drift_severity=severity,
            drift_severity_score=score,
            affected_features=c1_obj.affected_features,
            adaptation_budget_size=adaptation_budget_size,
            has_authentic_adaptation_labels=has_authentic_adaptation_labels,
            repair_objective=repair_objective,
        )

        logger.info(
            "Adapted C1 payload '%s' -> RepairRequest '%s' (type=%s, severity=%s, score=%.3f)",
            c1_obj.payload_id,
            request.request_id,
            drift_type.value,
            severity.value,
            score,
        )
        return request
