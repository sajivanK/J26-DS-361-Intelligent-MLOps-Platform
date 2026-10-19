"""Prescription Experiment Data Contract for Component 3.

Defines the formal schema for multi-dataset, multi-model repair experiment records
and enforces a strict PRE_DECISION_FEATURE_WHITELIST to mathematically prevent
post-generation data leakage in prescription meta-learning models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import pandas as pd


# Strict whitelist of features available BEFORE a repair prescription decision is made.
# Under NO circumstances may any post-generation outcome (coverage, fidelity, privacy,
# utility, runtime, package status) or untouched test-set metric (baseline_test_f1)
# enter this whitelist. The test set is strictly final evaluation evidence.
PRE_DECISION_FEATURE_WHITELIST: tuple[str, ...] = (
    # Section A: Dataset & Model Metadata
    "dataset_id",
    "dataset_version",
    "baseline_model_family",
    "baseline_model_version",
    "total_features_count",
    "numerical_features_count",
    "categorical_features_count",
    "numerical_feature_ratio",
    "dataset_missingness_rate",
    "class_imbalance_ratio",
    "reference_training_rows",
    # Section B: Pre-Decision Drift Context
    "drift_scenario",
    "drift_type",
    "drift_severity",
    "measured_drift_severity_score",
    "affected_features",
    "affected_feature_count",
    "slice_support",
    "diagnosis_stability",
    "slice_rule",
    "pre_repair_overall_f1",
    "pre_repair_slice_f1",
    # Section C: Candidate Prescription Levers (Decision Variables)
    "generator",
    "repair_quantity",
    "epochs",
    "generator_learning_rate",
    "seed",
)

# Post-generation outcome fields (strictly excluded from prescription model training features)
POST_GENERATION_OUTCOME_FIELDS: tuple[str, ...] = (
    "target_coverage",
    "structural_status",
    "fidelity_score",
    "mean_ks_statistic",
    "mean_total_variation_distance",
    "correlation_frobenius_norm",
    "duplicate_record_ratio",
    "privacy_status",
    "dcr_p05",
    "dcr_mean",
    "exact_match_count",
    "near_duplicate_count",
    "utility_score",
    "delta_overall_f1",
    "delta_affected_slice_f1",
    "delta_non_affected_f1",
    "delta_baseline_slice_f1",
    "training_seconds",
    "sampling_seconds",
    "validation_seconds",
    "total_seconds",
    "decision",
    "package_status",
)

TARGET_AND_GOVERNANCE_FIELDS: tuple[str, ...] = (
    "prescriptive_viability_label",
    "repair_improvement_label",
    "label_source",
    "label_authorized",
    "simulation_only",
    "evaluation_methodology",
    "experiment_provenance",
)

# Explicitly prohibited fields: post-generation outcomes and holdout test metrics
PROHIBITED_PRESCRIPTION_INPUTS: frozenset[str] = frozenset(
    set(POST_GENERATION_OUTCOME_FIELDS)
    | set(TARGET_AND_GOVERNANCE_FIELDS)
    | {"baseline_test_f1", "test_f1", "test_accuracy", "test_metrics", "test_loss"}
)

# Mathematical assertion: PRE_DECISION_FEATURE_WHITELIST must never intersect with prohibited inputs
assert not (set(PRE_DECISION_FEATURE_WHITELIST) & PROHIBITED_PRESCRIPTION_INPUTS), (
    f"Leakage defect: Prohibited fields found in PRE_DECISION_FEATURE_WHITELIST: "
    f"{set(PRE_DECISION_FEATURE_WHITELIST) & PROHIBITED_PRESCRIPTION_INPUTS}"
)


@dataclass
class PrescriptionExperimentRecord:
    """Formal structured record conforming to the Sprint 06 Data Contract."""
    # Provenance
    experiment_id: str

    # Section A: Dataset & Model Metadata
    dataset_id: str
    dataset_version: str
    baseline_model_family: str
    baseline_model_version: str
    total_features_count: int
    numerical_features_count: int
    categorical_features_count: int
    numerical_feature_ratio: float
    dataset_missingness_rate: float
    class_imbalance_ratio: float
    reference_training_rows: int

    # Section B: Pre-Decision Drift Context
    drift_scenario: str
    drift_type: str
    drift_severity: str
    measured_drift_severity_score: float
    affected_features: str
    affected_feature_count: int
    slice_support: float
    diagnosis_stability: float
    slice_rule: str
    pre_repair_overall_f1: float
    pre_repair_slice_f1: float

    # Section C: Candidate Repair Configuration
    generator: str
    repair_quantity: int
    epochs: int
    generator_learning_rate: float
    seed: int

    # Section D: Post-Generation Outcomes
    target_coverage: float
    structural_status: str
    fidelity_score: float
    mean_ks_statistic: float
    mean_total_variation_distance: float
    correlation_frobenius_norm: float
    duplicate_record_ratio: float
    privacy_status: str
    dcr_p05: float
    dcr_mean: float
    exact_match_count: int
    near_duplicate_count: int
    utility_score: float
    delta_overall_f1: float
    delta_affected_slice_f1: float
    delta_non_affected_f1: float
    delta_baseline_slice_f1: float
    training_seconds: float
    sampling_seconds: float
    validation_seconds: float
    total_seconds: float
    decision: str
    package_status: str

    # Section E: Research Targets & Governance
    prescriptive_viability_label: int
    repair_improvement_label: int
    label_source: str
    label_authorized: bool
    simulation_only: bool
    evaluation_methodology: str
    experiment_provenance: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    def extract_pre_decision_features(self) -> dict[str, Any]:
        """Extract only the leak-free pre-decision context and candidate action features."""
        d = self.to_dict()
        return {k: d[k] for k in PRE_DECISION_FEATURE_WHITELIST if k in d}


def validate_prescription_record(record: dict[str, Any]) -> bool:
    """Validate that a record conforms to all requirements of the prescription contract."""
    # Check experiment ID
    if not record.get("experiment_id"):
        raise ValueError("Record missing required 'experiment_id'")

    # Verify Pre-Decision Whitelist items are present
    missing_pre = [k for k in PRE_DECISION_FEATURE_WHITELIST if k not in record]
    if missing_pre:
        raise ValueError(f"Record missing pre-decision features: {missing_pre}")

    # Verify Post-Generation items are present
    missing_post = [k for k in POST_GENERATION_OUTCOME_FIELDS if k not in record]
    if missing_post:
        raise ValueError(f"Record missing post-generation outcome fields: {missing_post}")

    # Check bounds
    if not (0.0 <= float(record["target_coverage"]) <= 1.0):
        raise ValueError(f"target_coverage must be in [0, 1], got {record['target_coverage']}")
    if not (0.0 <= float(record["fidelity_score"]) <= 1.0):
        raise ValueError(f"fidelity_score must be in [0, 1], got {record['fidelity_score']}")
    if int(record["prescriptive_viability_label"]) not in (0, 1):
        raise ValueError(f"prescriptive_viability_label must be 0 or 1, got {record['prescriptive_viability_label']}")

    # Label source sanity check
    if record["label_authorized"] and record["label_source"] not in ("teacher_pseudo_label", "human_verified", "delayed_ground_truth"):
        raise ValueError(f"Unrecognized authorized label_source: '{record['label_source']}'")

    return True


def filter_leak_free_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to strictly include ONLY pre-decision features and the target label."""
    allowed_cols = [c for c in PRE_DECISION_FEATURE_WHITELIST if c in df.columns]
    if "prescriptive_viability_label" in df.columns:
        allowed_cols.append("prescriptive_viability_label")
    return df[allowed_cols].copy()
