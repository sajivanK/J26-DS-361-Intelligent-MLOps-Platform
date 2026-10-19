"""Drift Experiment Configuration Framework for Component 3.

Defines standardized configuration for 8 drift categories, strictly differentiating:
1. Injected Drift Strength (control parameter lambda in [0.0, 1.0])
2. Measured Drift Severity (detector statistics: KS, TVD, PSI, MMD)
3. Observed Model Performance Degradation (delta F1, delta Accuracy)

Guarantees that pure feature distribution shift is never mislabeled as concept drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any


class DriftCategory(str, Enum):
    """Eight canonical tabular data drift families."""
    NUMERICAL_DISTRIBUTION_SHIFT = "numerical_distribution_shift"
    CATEGORICAL_DISTRIBUTION_SHIFT = "categorical_distribution_shift"
    MIXED_FEATURE_SHIFT = "mixed_feature_shift"
    MISSINGNESS_SHIFT = "missingness_shift"
    FEATURE_CORRELATION_SHIFT = "feature_correlation_shift"
    CLASS_PRIOR_SHIFT = "class_prior_shift"
    COVARIATE_SHIFT = "covariate_shift"
    GENUINE_CONCEPT_DRIFT = "genuine_concept_drift"


class DriftSeverityTier(str, Enum):
    """Categorical severity tier assigned from measured drift statistics."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftScenarioConfig:
    """Configuration specification for a controlled drift injection scenario."""
    scenario_id: str
    dataset_id: str
    category: DriftCategory
    target_features: list[str]
    injected_strength: float
    slice_rule: str
    is_concept_drift: bool = False
    measured_severity_metric: str | None = None
    measured_severity_value: float | None = None
    severity_tier: DriftSeverityTier = DriftSeverityTier.MODERATE
    model_degradation_metric: str | None = None
    model_degradation_value: float | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate scenario definitions and scientific boundaries."""
        if not (0.0 <= self.injected_strength <= 1.0):
            raise ValueError(f"Injected drift strength must be in [0.0, 1.0], got {self.injected_strength}")

        # Scientific constraint: Feature distribution shifts cannot be claimed as concept drift
        # without explicit conditional probability shifts P(Y|X) and verified ground-truth labels
        if self.category != DriftCategory.GENUINE_CONCEPT_DRIFT and self.is_concept_drift:
            raise ValueError(
                f"Scenario '{self.scenario_id}' has category '{self.category.value}', "
                f"which represents input distribution shift P(X). It cannot be flagged as "
                f"is_concept_drift=True unless category is GENUINE_CONCEPT_DRIFT with ground-truth verification."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to dictionary."""
        d = asdict(self)
        d["category"] = self.category.value
        d["severity_tier"] = self.severity_tier.value
        return d

    def save(self, filepath: Path | str) -> None:
        """Serialize configuration to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> DriftScenarioConfig:
        """Load scenario configuration from JSON file."""
        p = Path(filepath)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["category"] = DriftCategory(data["category"])
        data["severity_tier"] = DriftSeverityTier(data.get("severity_tier", "moderate"))
        return cls(**data)


class DriftScenarioRegistry:
    """Catalog managing standardized drift scenarios across datasets."""

    def __init__(self) -> None:
        self._scenarios: dict[str, DriftScenarioConfig] = {}
        self._register_default_scenarios()

    def register(self, config: DriftScenarioConfig) -> None:
        """Register or update a drift scenario."""
        self._scenarios[config.scenario_id] = config

    def get(self, scenario_id: str) -> DriftScenarioConfig:
        """Retrieve scenario by unique ID."""
        if scenario_id not in self._scenarios:
            raise KeyError(f"Drift scenario '{scenario_id}' not found. Available: {sorted(self._scenarios.keys())}")
        return self._scenarios[scenario_id]

    def list_scenarios(self, dataset_id: str | None = None, category: DriftCategory | None = None) -> list[str]:
        """Filter scenarios by dataset or category."""
        results: list[str] = []
        for s_id, s in self._scenarios.items():
            if dataset_id and s.dataset_id != dataset_id:
                continue
            if category and s.category != category:
                continue
            results.append(s_id)
        return sorted(results)

    def _register_default_scenarios(self) -> None:
        """Register standard baseline scenarios S01-S07 for UCI Adult plus reference templates."""
        # S01: Age Continuous Tail Shift
        self.register(
            DriftScenarioConfig(
                scenario_id="S01",
                dataset_id="uci_adult",
                category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
                target_features=["age"],
                injected_strength=0.5,
                slice_rule="age > 0.69",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.HIGH,
                measured_severity_metric="ks_statistic",
                measured_severity_value=0.42,
                model_degradation_metric="delta_f1",
                model_degradation_value=-0.045,
                description="Right-skewed continuous shift on age feature isolating elder population slice.",
            )
        )

        # S02: Capital Loss Numerical Shift
        self.register(
            DriftScenarioConfig(
                scenario_id="S02",
                dataset_id="uci_adult",
                category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
                target_features=["capital_loss"],
                injected_strength=0.4,
                slice_rule="capital_loss > -0.20",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.MODERATE,
                measured_severity_metric="ks_statistic",
                measured_severity_value=0.38,
                description="Perturbation in capital loss distribution affecting investment loss slice.",
            )
        )

        # S03: Marital Status Categorical Shift
        self.register(
            DriftScenarioConfig(
                scenario_id="S03",
                dataset_id="uci_adult",
                category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
                target_features=["marital_status"],
                injected_strength=0.6,
                slice_rule="marital_status == 2",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.HIGH,
                measured_severity_metric="total_variation_distance",
                measured_severity_value=0.46,
                model_degradation_metric="delta_f1",
                model_degradation_value=-0.038,
                description="Frequency proportion shift on marital_status (married civilian spouse).",
            )
        )

        # S04: Race Categorical Shift (Primary Sprint 05 scenario)
        self.register(
            DriftScenarioConfig(
                scenario_id="S04",
                dataset_id="uci_adult",
                category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
                target_features=["race"],
                injected_strength=0.5,
                slice_rule="race == 2",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.MODERATE,
                measured_severity_metric="total_variation_distance",
                measured_severity_value=0.39,
                model_degradation_metric="delta_f1",
                model_degradation_value=-0.031,
                description="Categorical frequency redistribution on demographic race attribute.",
            )
        )

        # S05: Mixed Feature Shift (Age + Education Num)
        self.register(
            DriftScenarioConfig(
                scenario_id="S05",
                dataset_id="uci_adult",
                category=DriftCategory.MIXED_FEATURE_SHIFT,
                target_features=["age", "education_num"],
                injected_strength=0.5,
                slice_rule="age > 0.69 and education_num < -0.03",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.HIGH,
                measured_severity_metric="multivariate_energy_distance",
                measured_severity_value=0.44,
                description="Simultaneous continuous shift on age and education level.",
            )
        )

        # S06: Interaction Shift (Marital Status x Relationship)
        self.register(
            DriftScenarioConfig(
                scenario_id="S06",
                dataset_id="uci_adult",
                category=DriftCategory.FEATURE_CORRELATION_SHIFT,
                target_features=["marital_status", "relationship"],
                injected_strength=0.7,
                slice_rule="marital_status == 2 and relationship == 0",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.HIGH,
                measured_severity_metric="frobenius_correlation_norm",
                measured_severity_value=0.52,
                description="Correlation break decoupling marital status and household relationship.",
            )
        )

        # S07: Combined Shift (Age x Marital Status)
        self.register(
            DriftScenarioConfig(
                scenario_id="S07",
                dataset_id="uci_adult",
                category=DriftCategory.COVARIATE_SHIFT,
                target_features=["age", "marital_status"],
                injected_strength=0.6,
                slice_rule="age > 0.69 and marital_status == 2",
                is_concept_drift=False,
                severity_tier=DriftSeverityTier.HIGH,
                measured_severity_metric="population_stability_index",
                measured_severity_value=0.48,
                description="Joint covariate shift across continuous age and categorical marital status.",
            )
        )

        # S08: Concept Drift Template (Feature-target conditional distribution shift)
        self.register(
            DriftScenarioConfig(
                scenario_id="S08_CONCEPT_ADULT",
                dataset_id="uci_adult",
                category=DriftCategory.GENUINE_CONCEPT_DRIFT,
                target_features=["education_num", "hours_per_week"],
                injected_strength=0.5,
                slice_rule="education_num > 1.0 and hours_per_week > 0.5",
                is_concept_drift=True,
                severity_tier=DriftSeverityTier.CRITICAL,
                measured_severity_metric="conditional_label_shift",
                measured_severity_value=0.35,
                model_degradation_metric="delta_f1",
                model_degradation_value=-0.082,
                description="Simulated genuine concept drift with altered P(Income | Education, Hours) under verified ground truth.",
            )
        )


_GLOBAL_DRIFT_REGISTRY: DriftScenarioRegistry | None = None


def get_drift_registry() -> DriftScenarioRegistry:
    """Singleton accessor for global drift scenario registry."""
    global _GLOBAL_DRIFT_REGISTRY
    if _GLOBAL_DRIFT_REGISTRY is None:
        _GLOBAL_DRIFT_REGISTRY = DriftScenarioRegistry()
    return _GLOBAL_DRIFT_REGISTRY
