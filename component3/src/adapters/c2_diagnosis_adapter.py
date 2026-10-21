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
    diagnosed_problematic_slice: str = ""
    slice_predicate: str = ""
    affected_slice_rule: str = ""
    root_cause_feature: str = ""
    root_cause_category: str = "marginal"
    affected_feature_details: tuple[dict[str, Any], ...] = ()
    affected_features_ranked: tuple[str, ...] = ()
    affected_value_ranges: dict[str, Any] = field(default_factory=dict)
    slice_support: int | float = 0
    diagnosis_confidence: float = 1.0
    diagnosis_stability: float = 1.0
    drift_impact_score: float = 0.5
    temporal_window: int = 1
    provenance: str = "COMPONENT_2_DIAGNOSIS"
    diagnosis_id: str = field(default_factory=lambda: f"C2-DIAG-{uuid.uuid4().hex[:8].upper()}")
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_feature_details"] = list(self.affected_feature_details)
        d["affected_features_ranked"] = list(self.affected_features_ranked)
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
        import re

        if isinstance(payload, dict):
            try:
                # 1. Resolve slice rule with full key aliases
                raw_rule = (
                    payload.get("affected_slice_rule")
                    or payload.get("slice_predicate")
                    or payload.get("diagnosed_problematic_slice")
                    or ""
                )
                # Normalize boolean operators (e.g. AND -> and, OR -> or) for AST compatibility
                normalized_rule = str(raw_rule).strip()
                normalized_rule = re.sub(r"\bAND\b", "and", normalized_rule)
                normalized_rule = re.sub(r"\bOR\b", "or", normalized_rule)
                normalized_rule = re.sub(r"\bNOT\b", "not", normalized_rule)

                # 2. Ranked features
                ranked = payload.get("affected_features_ranked", payload.get("affected_features", ()))
                ranked_tuple = tuple(str(f).strip() for f in ranked)

                # 3. Confidence / stability
                conf = float(
                    payload.get("diagnosis_stability")
                    if "diagnosis_stability" in payload
                    else payload.get("diagnostic_confidence", payload.get("drift_impact_score", 1.0))
                )

                details = tuple(payload.get("affected_feature_details", []))
                c2_obj = C2DiagnosisPayload(
                    diagnosed_problematic_slice=normalized_rule,
                    slice_predicate=normalized_rule,
                    affected_slice_rule=normalized_rule,
                    root_cause_feature=str(payload.get("root_cause_feature", "")).strip(),
                    root_cause_category=str(payload.get("root_cause_category", "marginal")).strip(),
                    affected_feature_details=details,
                    affected_features_ranked=ranked_tuple,
                    affected_value_ranges=dict(payload.get("affected_value_ranges", {})),
                    slice_support=payload.get("slice_support", 0),
                    diagnosis_confidence=conf,
                    diagnosis_stability=float(payload.get("diagnosis_stability", conf)),
                    drift_impact_score=float(payload.get("drift_impact_score", 0.5)),
                    temporal_window=int(payload.get("temporal_window", 1)),
                    provenance=str(payload.get("provenance", "COMPONENT_2_DIAGNOSIS")),
                    diagnosis_id=str(payload.get("diagnosis_id", f"C2-DIAG-{uuid.uuid4().hex[:8].upper()}")),
                    timestamp_utc=str(payload.get("timestamp_utc", datetime.now(timezone.utc).isoformat())),
                )
            except Exception as exc:
                raise ValueError(f"Malformed C2 diagnosis payload: {exc}") from exc
        else:
            c2_obj = payload

        # Determine affected feature names from ranked features, root cause, or details
        feature_names: list[str] = []
        if c2_obj.root_cause_feature:
            feature_names.append(c2_obj.root_cause_feature)

        for feat in c2_obj.affected_features_ranked:
            if feat not in feature_names:
                feature_names.append(feat)

        for feat_detail in c2_obj.affected_feature_details:
            if isinstance(feat_detail, dict) and "feature_name" in feat_detail:
                fn = feat_detail["feature_name"]
                if fn not in feature_names:
                    feature_names.append(fn)
            elif isinstance(feat_detail, str) and feat_detail not in feature_names:
                feature_names.append(feat_detail)

        predicate = c2_obj.slice_predicate.strip()
        if not feature_names and predicate:
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
            diagnostic_confidence=c2_obj.diagnosis_confidence,
            provenance=prov_enum,
            is_non_oracle=is_non_oracle,
        )

        logger.info(
            "Adapted C2 payload '%s' -> DiagnosticSliceDefinition '%s' (provenance=%s, non_oracle=%s, features=%s)",
            c2_obj.diagnosis_id,
            active_slice_id,
            prov_enum.value,
            is_non_oracle,
            feature_names,
        )
        return slice_def
