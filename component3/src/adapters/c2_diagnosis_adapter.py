"""Component 2 Drift Diagnosis Integration Adapter for Component 3.

Ingests root cause diagnosis payloads from Component 2 and maps them into validated
DiagnosticSliceDefinition contracts, strictly enforcing non-oracle diagnostic integrity.

PROVISIONAL ADAPTER NOTICE:
This adapter establishes a typed integration boundary between Component 2 (Root Cause Diagnosis)
and Component 3 (Repair Engine). Field mappings reflect agreed architectural contracts
and will be finalized upon Component 2 interface freeze.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any
import uuid

from component3.src.contracts.targeted_repair import (
    DiagnosticProvenance,
    DiagnosticSliceDefinition,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class C2DiagnosisPayload:
    """Incoming contract schema from Component 2 (Drift Diagnosis)."""
    diagnosed_problematic_slice: str  # e.g. "tenure > 29"
    slice_predicate: str  # Safe machine-readable predicate e.g. "tenure > 29"
    affected_feature_details: tuple[dict[str, Any], ...] = ()
    slice_support: int | float = 0  # Row count or support fraction
    diagnostic_confidence: float = 1.0  # [0.0, 1.0]
    provenance: str = "COMPONENT_2_DIAGNOSIS"  # e.g. "SUBGROUP_DISCOVERY", "SIMULATION_ORACLE"
    diagnosis_id: str = field(default_factory=lambda: f"C2-DIAG-{uuid.uuid4().hex[:8].upper()}")
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_feature_details"] = list(self.affected_feature_details)
        return d


def normalize_c2_provenance(raw_provenance: str) -> DiagnosticProvenance:
    """Classify provenance string and guard against oracle leakage."""
    clean = str(raw_provenance).strip().upper()
    oracle_terms = ("ORACLE", "SIMULATION", "SIMULATED", "GROUND_TRUTH", "SYNTHETIC_FLIP", "INTERNAL_TEST")
    if any(term in clean for term in oracle_terms):
        return DiagnosticProvenance.SIMULATION_ORACLE
    if "USER" in clean:
        return DiagnosticProvenance.USER_SPECIFIED
    if "HEURISTIC" in clean:
        return DiagnosticProvenance.UNTRUSTED_HEURISTIC
    return DiagnosticProvenance.COMPONENT_2_DIAGNOSIS


class C2DiagnosisAdapter:
    """Adapter transforming C2 diagnosis payloads into validated DiagnosticSliceDefinitions."""

    def adapt(
        self,
        payload: dict[str, Any] | C2DiagnosisPayload,
        slice_id: str | None = None,
    ) -> DiagnosticSliceDefinition:
        """Validate and map incoming C2 diagnosis payload."""
        if isinstance(payload, dict):
            try:
                details = tuple(payload.get("affected_feature_details", []))
                c2_obj = C2DiagnosisPayload(
                    diagnosed_problematic_slice=str(
                        payload.get("diagnosed_problematic_slice", payload.get("slice_predicate", ""))
                    ),
                    slice_predicate=str(
                        payload.get("slice_predicate", payload.get("diagnosed_problematic_slice", ""))
                    ),
                    affected_feature_details=details,
                    slice_support=payload.get("slice_support", 0),
                    diagnostic_confidence=float(payload.get("diagnostic_confidence", 1.0)),
                    provenance=str(payload.get("provenance", "COMPONENT_2_DIAGNOSIS")),
                    diagnosis_id=str(payload.get("diagnosis_id", f"C2-DIAG-{uuid.uuid4().hex[:8].upper()}")),
                    timestamp_utc=str(payload.get("timestamp_utc", datetime.now(timezone.utc).isoformat())),
                )
            except Exception as exc:
                raise ValueError(f"Malformed C2 diagnosis payload: {exc}") from exc
        else:
            c2_obj = payload

        # Determine affected feature names from details or predicate
        feature_names: list[str] = []
        for feat in c2_obj.affected_feature_details:
            if isinstance(feat, dict) and "feature_name" in feat:
                feature_names.append(feat["feature_name"])
            elif isinstance(feat, str):
                feature_names.append(feat)

        predicate = c2_obj.slice_predicate.strip()
        if not feature_names and predicate:
            # Extract basic identifier token from predicate
            import re
            tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", predicate)
            reserved = {"and", "or", "not", "in", "is", "True", "False"}
            feature_names = [t for t in tokens if t not in reserved]

        # Determine provenance
        prov_enum = normalize_c2_provenance(c2_obj.provenance)
        is_non_oracle = (prov_enum != DiagnosticProvenance.SIMULATION_ORACLE)

        # Support calculation
        support_val = float(c2_obj.slice_support)
        support_fraction = support_val if 0.0 <= support_val <= 1.0 else 0.25

        active_slice_id = slice_id or f"SLICE-FROM-{c2_obj.diagnosis_id}"

        slice_def = DiagnosticSliceDefinition(
            slice_id=active_slice_id,
            slice_predicate=predicate,
            affected_features=tuple(feature_names),
            reported_slice_support=support_fraction,
            diagnostic_confidence=c2_obj.diagnostic_confidence,
            provenance=prov_enum,
            is_non_oracle=is_non_oracle,
        )

        logger.info(
            "Adapted C2 payload '%s' -> DiagnosticSliceDefinition '%s' (provenance=%s, non_oracle=%s)",
            c2_obj.diagnosis_id,
            active_slice_id,
            prov_enum.value,
            is_non_oracle,
        )
        return slice_def
