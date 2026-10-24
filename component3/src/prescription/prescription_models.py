"""Prescription decision models and data structures for Component 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RepairStrategy(str, Enum):
    """Supported repair strategies in the Component 3 taxonomy."""
    NO_REPAIR = "NO_REPAIR"
    AUTHENTIC_ONLY_ADAPTATION = "AUTHENTIC_ONLY_ADAPTATION"
    AUTHENTIC_OVERSAMPLING = "AUTHENTIC_OVERSAMPLING"
    AUTHENTIC_WEIGHTED_ADAPTATION = "AUTHENTIC_WEIGHTED_ADAPTATION"
    SYNTHETIC_UNWEIGHTED = "SYNTHETIC_UNWEIGHTED"
    SYNTHETIC_WEIGHTED_HYBRID = "SYNTHETIC_WEIGHTED_HYBRID"
    TARGETED_SYNTHETIC_REPAIR = "TARGETED_SYNTHETIC_REPAIR"


class DecisionMode(str, Enum):
    """Supported decision modes in the prescription engine."""
    RULE_BASED = "RULE_BASED"
    LEARNED = "LEARNED"
    LEARNED_ADVISORY = "LEARNED_ADVISORY"
    SHADOW_MODE = "SHADOW_MODE"


class EvidenceStrength(str, Enum):
    """Categorical evidence strength classification based on empirical registry."""
    CONFIRMED_BENCHMARK = "CONFIRMED_BENCHMARK"
    EXPLORATORY_PILOT = "EXPLORATORY_PILOT"
    THEORETICAL_HEURISTIC = "THEORETICAL_HEURISTIC"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class PrescriptionDecision:
    """Versioned prescription decision emitted by the prescription engine."""
    strategy: RepairStrategy
    recommended_generator: str | None
    sample_weight_policy: dict[str, float]
    initial_candidate_quantity: int
    evidence_strength: EvidenceStrength
    decision_reasons: tuple[str, ...]
    governance_checks_passed: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    pre_decision_features: dict[str, Any] = field(default_factory=dict)
    targeted_slice_id: str | None = None
    decision_mode: DecisionMode = DecisionMode.RULE_BASED
    predicted_delta_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert prescription to serializable dictionary."""
        d = asdict(self)
        d["strategy"] = self.strategy.value
        d["evidence_strength"] = self.evidence_strength.value
        d["decision_mode"] = self.decision_mode.value
        d["decision_reasons"] = list(self.decision_reasons)
        d["governance_checks_passed"] = list(self.governance_checks_passed)
        d["warnings"] = list(self.warnings)
        return d
