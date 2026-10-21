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
from pathlib import Path
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
        "CONCEPT_SHIFT": DriftType.CONCEPT_DRIFT,
        "COVARIATE": DriftType.COVARIATE_SHIFT,
        "COVARIATE_SHIFT": DriftType.COVARIATE_SHIFT,
        "DATA_DRIFT": DriftType.COVARIATE_SHIFT,
        "MARGINAL": DriftType.COVARIATE_SHIFT,
        "MARGINAL_SINGLE": DriftType.COVARIATE_SHIFT,
        "MARGINAL_MULTI": DriftType.COVARIATE_SHIFT,
        "COMBINED": DriftType.COVARIATE_SHIFT,
        "SUBPOPULATION": DriftType.SUBPOPULATION_SHIFT,
        "SUBPOPULATION_SHIFT": DriftType.SUBPOPULATION_SHIFT,
        "INTERACTION": DriftType.INTERACTION,
        "NONE": DriftType.NONE,
        "NO_DRIFT": DriftType.NONE,
    }
    return mapping.get(clean, DriftType.UNKNOWN)


def normalize_c1_severity(score: float, severity_label: str | None = None) -> DriftSeverity:
    """Classify numerical drift severity score or categorical label into DriftSeverity."""
    if severity_label:
        clean_lbl = str(severity_label).strip().lower()
        lbl_mapping = {
            "none": DriftSeverity.NONE,
            "low": DriftSeverity.LOW,
            "moderate": DriftSeverity.MEDIUM,
            "medium": DriftSeverity.MEDIUM,
            "high": DriftSeverity.HIGH,
            "critical": DriftSeverity.CRITICAL,
        }
        if clean_lbl in lbl_mapping:
            return lbl_mapping[clean_lbl]

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
        source_model_path: str | Path | None = None,
        source_preprocessor_path: str | Path | None = None,
    ) -> RepairRequest:
        """Validate and transform C1 payload into a RepairRequest."""
        if isinstance(payload, dict):
            try:
                # 1. Resolve accuracy drop / severity score
                raw_score = (
                    payload.get("predicted_accuracy_drop")
                    if "predicted_accuracy_drop" in payload
                    else payload.get("predicted_harm", payload.get("severity_score", 0.0))
                )
                score_flt = float(raw_score)
                # If expressed as percentage points (e.g. 14.2), normalize to proportion 0.142
                normalized_score = score_flt / 100.0 if score_flt > 1.0 else score_flt
                normalized_score = max(0.0, min(1.0, normalized_score))

                # 2. Resolve drift type & severity
                raw_type = str(payload.get("drift_type", "NONE"))
                raw_sev = str(payload.get("drift_severity", "none"))

                # 3. Derive drift_detected flag if not explicitly boolean
                if "drift_detected" in payload:
                    detected = bool(payload["drift_detected"])
                else:
                    detected = (
                        raw_type.lower() not in ("none", "no_drift")
                        and raw_sev.lower() != "none"
                        and normalized_score > 0.005
                    )

                # 4. Features
                raw_feats = payload.get("affected_features", payload.get("features_drifted", ()))
                features = tuple(str(f) for f in raw_feats)

                # 5. Provenance tracking
                prov = dict(payload.get("evidence_provenance", {}))
                prov["original_c1_payload"] = dict(payload)
                if "confidence_interval" in payload:
                    prov["confidence_interval"] = payload["confidence_interval"]
                elif "harm_interval" in payload:
                    prov["harm_interval"] = payload["harm_interval"]

                c1_obj = C1DriftPayload(
                    drift_detected=detected,
                    drift_type=raw_type,
                    severity_score=normalized_score,
                    affected_features=features,
                    monitoring_window_metadata=dict(payload.get("monitoring_window_metadata", payload.get("window_metadata", {}))),
                    evidence_provenance=prov,
                    payload_id=str(payload.get("detection_id", payload.get("payload_id", f"C1-EV-{uuid.uuid4().hex[:8].upper()}"))),
                    timestamp_utc=str(payload.get("timestamp", payload.get("timestamp_utc", datetime.now(timezone.utc).isoformat()))),
                )
            except Exception as exc:
                raise InvalidRepairRequestError(f"Malformed C1 drift payload: {exc}") from exc
        else:
            c1_obj = payload

        # Score sanity check
        score = max(0.0, min(1.0, float(c1_obj.severity_score)))
        drift_type = normalize_c1_drift_type(c1_obj.drift_type)
        raw_sev_str = payload.get("drift_severity") if isinstance(payload, dict) else None
        severity = normalize_c1_severity(score, severity_label=raw_sev_str) if c1_obj.drift_detected else DriftSeverity.NONE

        active_dataset_id = dataset_id or self.default_dataset_id

        # Path overrides if provided
        kwargs: dict[str, Any] = {}
        if source_model_path:
            kwargs["source_model_path"] = source_model_path
        if source_preprocessor_path:
            kwargs["source_preprocessor_path"] = source_preprocessor_path

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
            **kwargs,
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
