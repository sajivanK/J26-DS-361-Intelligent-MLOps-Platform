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


class LabelValidityStatus(str, Enum):
    """Provenance and scientific validity status of target labels under drift."""
    VERIFIED_OBSERVATIONAL_GROUND_TRUTH = "verified_observational_ground_truth"
    VERIFIED_RESAMPLED_GROUND_TRUTH = "verified_resampled_ground_truth"
    VERIFIED_OBSERVATIONAL_MISSINGNESS = "verified_observational_missingness"
    SYNTHETIC_CONCEPT_TRUTH = "synthetic_concept_truth"
    STRESS_TEST_UNCERTAIN = "stress_test_uncertain"


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
    label_validity: LabelValidityStatus = LabelValidityStatus.STRESS_TEST_UNCERTAIN
    concept_policy_description: str = ""
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
        d["label_validity"] = self.label_validity.value
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
        if "label_validity" in data:
            data["label_validity"] = LabelValidityStatus(data["label_validity"])
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

        # ----------------------------------------------------------------------
        # Calibrated Multi-Dataset Benchmark Scenarios (8 Types x 3 Datasets)
        # ----------------------------------------------------------------------
        # UCI Adult Full
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_NUM",
                dataset_id="uci_adult_full",
                category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
                target_features=["age"],
                injected_strength=0.5,
                slice_rule="age > 45",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Right-skewed continuous shift on age feature isolating elder population slice.",
                metadata={
                    "frozen_feature_std": {"age": 13.7000},
                    "std_source": "train_partition_std",
                    "scenario_version": "1.0.0",
                },
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_CAT",
                dataset_id="uci_adult_full",
                category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
                target_features=["marital_status"],
                injected_strength=0.5,
                slice_rule="marital_status == 'Married-civ-spouse'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Frequency redistribution on marital status attribute.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_MIXED",
                dataset_id="uci_adult_full",
                category=DriftCategory.MIXED_FEATURE_SHIFT,
                target_features=["age", "education_num"],
                injected_strength=0.5,
                slice_rule="age > 40 and education_num < 10",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Simultaneous continuous shift on age and education level.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_MISS",
                dataset_id="uci_adult_full",
                category=DriftCategory.MISSINGNESS_SHIFT,
                target_features=["occupation", "workclass"],
                injected_strength=0.35,
                slice_rule="age > 30",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_OBSERVATIONAL_MISSINGNESS,
                description="Targeted missingness injection into occupation and employment fields.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_CORR",
                dataset_id="uci_adult_full",
                category=DriftCategory.FEATURE_CORRELATION_SHIFT,
                target_features=["marital_status", "relationship"],
                injected_strength=0.6,
                slice_rule="marital_status == 'Married-civ-spouse'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Correlation break decoupling marital status and household relationship.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_PRIOR",
                dataset_id="uci_adult_full",
                category=DriftCategory.CLASS_PRIOR_SHIFT,
                target_features=["target"],
                injected_strength=0.6,
                slice_rule="target == '>50K'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_RESAMPLED_GROUND_TRUTH,
                description="Prior probability shift P(Y) shifting high-earner ratio via authentic stratified resampling.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_COV",
                dataset_id="uci_adult_full",
                category=DriftCategory.COVARIATE_SHIFT,
                target_features=["education_num"],
                injected_strength=0.5,
                slice_rule="education_num > 10",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_OBSERVATIONAL_GROUND_TRUTH,
                description="Subpopulation covariate shift focused on higher education cohort.",
                metadata={
                    "frozen_feature_std": {"education_num": 2.5645},
                    "std_source": "train_partition_std",
                    "scenario_version": "1.0.0",
                },
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="ADULT_CONCEPT",
                dataset_id="uci_adult_full",
                category=DriftCategory.GENUINE_CONCEPT_DRIFT,
                target_features=["hours_per_week"],
                injected_strength=0.5,
                slice_rule="hours_per_week > 40.0",
                is_concept_drift=True,
                label_validity=LabelValidityStatus.SYNTHETIC_CONCEPT_TRUTH,
                concept_policy_description="Controlled synthetic label-inversion simulation altering P(Y|X) for high-hours subpopulation under frozen development threshold (hours_per_week > 40.0).",
                description="Controlled synthetic label-inversion simulation altering conditional income probability P(Y|X) for hours_per_week > 40.0.",
                metadata={
                    "frozen_threshold": 40.0,
                    "threshold_source": "train_partition_median",
                    "threshold_feature": "hours_per_week",
                    "scenario_version": "1.0.0",
                },
            )
        )

        # Telco Customer Churn
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_NUM",
                dataset_id="telco_churn",
                category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
                target_features=["MonthlyCharges"],
                injected_strength=0.5,
                slice_rule="MonthlyCharges > 70",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Continuous inflation shift on monthly charges feature.",
                metadata={
                    "frozen_feature_std": {"MonthlyCharges": 30.0196},
                    "std_source": "train_partition_std",
                    "scenario_version": "1.0.0",
                },
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_CAT",
                dataset_id="telco_churn",
                category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
                target_features=["Contract"],
                injected_strength=0.5,
                slice_rule="Contract == 'Month-to-month'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Contract plan proportion shift towards month-to-month subscriptions.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_MIXED",
                dataset_id="telco_churn",
                category=DriftCategory.MIXED_FEATURE_SHIFT,
                target_features=["MonthlyCharges", "Contract"],
                injected_strength=0.5,
                slice_rule="MonthlyCharges > 65",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Simultaneous shift on monthly charges and contract tenure types.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_MISS",
                dataset_id="telco_churn",
                category=DriftCategory.MISSINGNESS_SHIFT,
                target_features=["TotalCharges", "TechSupport"],
                injected_strength=0.35,
                slice_rule="tenure > 12",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_OBSERVATIONAL_MISSINGNESS,
                description="Targeted missingness in billing history and technical support records.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_CORR",
                dataset_id="telco_churn",
                category=DriftCategory.FEATURE_CORRELATION_SHIFT,
                target_features=["tenure", "TotalCharges"],
                injected_strength=0.6,
                slice_rule="tenure > 24",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Correlation break decoupling tenure longevity from accumulated total charges.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_PRIOR",
                dataset_id="telco_churn",
                category=DriftCategory.CLASS_PRIOR_SHIFT,
                target_features=["Churn"],
                injected_strength=0.65,
                slice_rule="Churn == 'Yes'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_RESAMPLED_GROUND_TRUTH,
                description="Macro churn surge shifting churn prevalence from 26.5% to ~45% via authentic resampling.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_COV",
                dataset_id="telco_churn",
                category=DriftCategory.COVARIATE_SHIFT,
                target_features=["tenure"],
                injected_strength=0.5,
                slice_rule="Contract == 'Month-to-month'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_OBSERVATIONAL_GROUND_TRUTH,
                description="Subpopulation covariate shift evaluating month-to-month subscriber segment.",
                metadata={
                    "frozen_feature_std": {"tenure": 24.6110},
                    "std_source": "train_partition_std",
                    "scenario_version": "1.0.0",
                },
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="TELCO_CONCEPT",
                dataset_id="telco_churn",
                category=DriftCategory.GENUINE_CONCEPT_DRIFT,
                target_features=["tenure"],
                injected_strength=0.5,
                slice_rule="tenure > 29.0",
                is_concept_drift=True,
                label_validity=LabelValidityStatus.SYNTHETIC_CONCEPT_TRUTH,
                concept_policy_description="Controlled synthetic label-inversion simulation altering P(Y|X) for established customers under frozen development threshold (tenure > 29.0).",
                description="Controlled synthetic label-inversion simulation altering churn relationship for established tenure accounts (tenure > 29.0).",
                metadata={
                    "frozen_threshold": 29.0,
                    "threshold_source": "train_partition_median",
                    "threshold_feature": "tenure",
                    "scenario_version": "1.0.0",
                },
            )
        )

        # Bank Marketing
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_NUM",
                dataset_id="bank_marketing",
                category=DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT,
                target_features=["euribor3m"],
                injected_strength=0.5,
                slice_rule="euribor3m > 3.0",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Macroeconomic interest rate shift on euribor 3-month rate indicator.",
                metadata={
                    "frozen_feature_std": {"euribor3m": 1.7340},
                    "std_source": "train_partition_std",
                    "scenario_version": "1.0.0",
                },
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_CAT",
                dataset_id="bank_marketing",
                category=DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT,
                target_features=["contact"],
                injected_strength=0.5,
                slice_rule="contact == 'cellular'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Communication channel shift altering contact method distribution.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_MIXED",
                dataset_id="bank_marketing",
                category=DriftCategory.MIXED_FEATURE_SHIFT,
                target_features=["euribor3m", "job"],
                injected_strength=0.5,
                slice_rule="euribor3m > 2.0",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Joint macroeconomic and professional employment category shift.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_MISS",
                dataset_id="bank_marketing",
                category=DriftCategory.MISSINGNESS_SHIFT,
                target_features=["poutcome", "education"],
                injected_strength=0.35,
                slice_rule="age > 35",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_OBSERVATIONAL_MISSINGNESS,
                description="Targeted missingness injection into previous outcome and education records.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_CORR",
                dataset_id="bank_marketing",
                category=DriftCategory.FEATURE_CORRELATION_SHIFT,
                target_features=["emp_var_rate", "euribor3m"],
                injected_strength=0.6,
                slice_rule="euribor3m > 1.5",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.STRESS_TEST_UNCERTAIN,
                description="Correlation break decoupling employment variation rate from euribor interest index.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_PRIOR",
                dataset_id="bank_marketing",
                category=DriftCategory.CLASS_PRIOR_SHIFT,
                target_features=["y"],
                injected_strength=0.6,
                slice_rule="y == 'yes'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_RESAMPLED_GROUND_TRUTH,
                description="Term deposit conversion surge from 11.3% to ~25% via authentic stratified resampling.",
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_COV",
                dataset_id="bank_marketing",
                category=DriftCategory.COVARIATE_SHIFT,
                target_features=["euribor3m"],
                injected_strength=0.5,
                slice_rule="poutcome == 'success'",
                is_concept_drift=False,
                label_validity=LabelValidityStatus.VERIFIED_OBSERVATIONAL_GROUND_TRUTH,
                description="Subpopulation covariate shift focused on historically successful campaign contacts.",
                metadata={
                    "frozen_feature_std": {"euribor3m": 1.7340},
                    "std_source": "train_partition_std",
                    "scenario_version": "1.0.0",
                },
            )
        )
        self.register(
            DriftScenarioConfig(
                scenario_id="BANK_CONCEPT",
                dataset_id="bank_marketing",
                category=DriftCategory.GENUINE_CONCEPT_DRIFT,
                target_features=["euribor3m"],
                injected_strength=0.5,
                slice_rule="euribor3m > 4.857",
                is_concept_drift=True,
                label_validity=LabelValidityStatus.SYNTHETIC_CONCEPT_TRUTH,
                concept_policy_description="Controlled synthetic label-inversion simulation altering P(Y|X) for high euribor3m subpopulation under frozen development threshold (euribor3m > 4.857).",
                description="Controlled synthetic label-inversion simulation altering deposit propensity under frozen economic indicator threshold (euribor3m > 4.857).",
                metadata={
                    "frozen_threshold": 4.857,
                    "threshold_source": "train_partition_median",
                    "threshold_feature": "euribor3m",
                    "scenario_version": "1.0.0",
                },
            )
        )


_GLOBAL_DRIFT_REGISTRY: DriftScenarioRegistry | None = None


def get_drift_registry() -> DriftScenarioRegistry:
    """Singleton accessor for global drift scenario registry."""
    global _GLOBAL_DRIFT_REGISTRY
    if _GLOBAL_DRIFT_REGISTRY is None:
        _GLOBAL_DRIFT_REGISTRY = DriftScenarioRegistry()
    return _GLOBAL_DRIFT_REGISTRY
