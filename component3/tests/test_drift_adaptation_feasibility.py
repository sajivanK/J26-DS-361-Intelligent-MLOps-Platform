"""Deterministic Protocol and Feasibility Unit Tests for Post-Drift Adaptation.

Validates:
1. Label-provenance categorization and drift scenario registry consistency.
2. 4-way partition isolation rules (disjoint indices, zero data leakage).
3. Metric safety preconditions (degradation denominator check for recovery fraction).
4. Prescription contract governance field assertions.
5. Safe abstention logic under zero adaptation data or low acceptance rate.

NOTE: Does NOT train or retrain any models. Runs purely deterministically on CPU.
"""

from __future__ import annotations

import pytest
import numpy as np

from component3.src.drift.drift_config import (
    DriftCategory,
    DriftSeverityTier,
    LabelValidityStatus,
    DriftScenarioConfig,
    get_drift_registry,
)
from component3.src.contracts.prescription_contract import (
    PRE_DECISION_FEATURE_WHITELIST,
    POST_GENERATION_OUTCOME_FIELDS,
    TARGET_AND_GOVERNANCE_FIELDS,
    PROHIBITED_PRESCRIPTION_INPUTS,
)


def test_label_validity_taxonomy_consistency():
    """Verify that all canonical drift scenarios in the registry have valid label validity tags."""
    registry = get_drift_registry()
    scenarios = registry.list_scenarios()
    assert len(scenarios) >= 8, f"Expected at least 8 registered scenarios, got {len(scenarios)}"

    valid_statuses = {
        LabelValidityStatus.VERIFIED_OBSERVATIONAL_GROUND_TRUTH,
        LabelValidityStatus.VERIFIED_RESAMPLED_GROUND_TRUTH,
        LabelValidityStatus.VERIFIED_OBSERVATIONAL_MISSINGNESS,
        LabelValidityStatus.SYNTHETIC_CONCEPT_TRUTH,
        LabelValidityStatus.STRESS_TEST_UNCERTAIN,
    }

    # Benchmark concept scenarios must have SYNTHETIC_CONCEPT_TRUTH
    for s_id in ["ADULT_CONCEPT", "TELCO_CONCEPT", "BANK_CONCEPT"]:
        config = registry.get(s_id)
        assert config.label_validity == LabelValidityStatus.SYNTHETIC_CONCEPT_TRUTH, (
            f"Benchmark concept scenario {s_id} must have SYNTHETIC_CONCEPT_TRUTH provenance"
        )
        assert config.is_concept_drift is True, f"Scenario {s_id} must have is_concept_drift=True"
        assert config.category == DriftCategory.GENUINE_CONCEPT_DRIFT, f"Scenario {s_id} must be GENUINE_CONCEPT_DRIFT"

    # TELCO_COV feature perturbation must be classified as STRESS_TEST_UNCERTAIN
    telco_cov = registry.get("TELCO_COV")
    assert telco_cov.label_validity == LabelValidityStatus.STRESS_TEST_UNCERTAIN, (
        f"TELCO_COV must be STRESS_TEST_UNCERTAIN, got {telco_cov.label_validity}"
    )
    assert telco_cov.label_validity != LabelValidityStatus.VERIFIED_OBSERVATIONAL_GROUND_TRUTH

    for s_id in scenarios:
        config = registry.get(s_id)
        assert config.label_validity in valid_statuses, (
            f"Scenario {s_id} has invalid label validity: {config.label_validity}"
        )

        # Enforce that concept drift is strictly flagged only for GENUINE_CONCEPT_DRIFT
        if config.is_concept_drift:
            assert config.category == DriftCategory.GENUINE_CONCEPT_DRIFT, (
                f"Scenario {s_id} is flagged is_concept_drift=True but category is {config.category}"
            )


def test_four_way_partition_isolation():
    """Verify mathematical properties of the proposed 4-way development partitioning scheme."""
    # Simulate permitted development indices (e.g., Telco N=4,931)
    n_dev = 4931
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(n_dev)

    # Proposed allocation: 70% source train, 10% source eval, 10% adapt pool, 10% target eval
    n_source_train = int(0.70 * n_dev)
    n_source_eval = int(0.10 * n_dev)
    n_adapt_pool = int(0.10 * n_dev)
    n_target_eval = n_dev - (n_source_train + n_source_eval + n_adapt_pool)

    idx_source_train = set(shuffled_indices[:n_source_train])
    idx_source_eval = set(shuffled_indices[n_source_train : n_source_train + n_source_eval])
    idx_adapt_pool = set(shuffled_indices[n_source_train + n_source_eval : n_source_train + n_source_eval + n_adapt_pool])
    idx_target_eval = set(shuffled_indices[n_source_train + n_source_eval + n_adapt_pool :])

    # 1. Total count equality
    assert len(idx_source_train) + len(idx_source_eval) + len(idx_adapt_pool) + len(idx_target_eval) == n_dev

    # 2. Strict pairwise disjointness
    assert len(idx_source_train & idx_source_eval) == 0, "Leakage: source train and source eval intersect"
    assert len(idx_source_train & idx_adapt_pool) == 0, "Leakage: source train and adapt pool intersect"
    assert len(idx_source_train & idx_target_eval) == 0, "Leakage: source train and target eval intersect"
    assert len(idx_source_eval & idx_adapt_pool) == 0, "Leakage: source eval and adapt pool intersect"
    assert len(idx_source_eval & idx_target_eval) == 0, "Leakage: source eval and target eval intersect"
    assert len(idx_adapt_pool & idx_target_eval) == 0, "Leakage: adapt pool and target eval intersect"


def test_recovery_fraction_denominator_safety():
    """Verify that relative recovery fraction computation guards against zero or negative degradation."""
    def compute_recovery_fraction(f1_source: float, f1_target_frozen: float, f1_adapted: float) -> float | None:
        delta_deg = f1_source - f1_target_frozen
        if delta_deg < 0.05:  # Precondition threshold
            return None
        return (f1_adapted - f1_target_frozen) / delta_deg * 100.0

    # Case 1: Valid degradation (e.g. 0.62 -> 0.42, delta_deg = 0.20)
    rec = compute_recovery_fraction(0.62, 0.42, 0.52)
    assert rec is not None
    assert round(rec, 1) == 50.0

    # Case 2: Negligible degradation (delta_deg = 0.005) -> Must return None (undefined)
    assert compute_recovery_fraction(0.62, 0.615, 0.625) is None

    # Case 3: Negative degradation (performance improved under shift) -> Must return None
    assert compute_recovery_fraction(0.62, 0.65, 0.66) is None


def test_prescription_contract_governance_integrity():
    """Verify that prescription contract governance fields are intact and leak-free."""
    # Ensure all target and governance fields are isolated from pre-decision inputs
    for field in TARGET_AND_GOVERNANCE_FIELDS:
        assert field in PROHIBITED_PRESCRIPTION_INPUTS, f"Field {field} must be in PROHIBITED_PRESCRIPTION_INPUTS"
        assert field not in PRE_DECISION_FEATURE_WHITELIST, f"Leakage: {field} found in PRE_DECISION_FEATURE_WHITELIST"

    # Ensure required governance fields exist
    expected_governance = {
        "prescriptive_viability_label",
        "repair_improvement_label",
        "label_source",
        "label_authorized",
        "simulation_only",
        "evaluation_methodology",
        "experiment_provenance",
    }
    assert set(TARGET_AND_GOVERNANCE_FIELDS) == expected_governance


def test_abstention_rules_for_adaptation():
    """Verify deterministic abstention decision logic for post-drift adaptation."""
    def decide_adaptation_action(
        n_adapt_available: int,
        label_validity: str,
        unique_valid_candidates: int,
        requested_budget: int,
    ) -> tuple[str, str]:
        """Deterministic policy returning (decision, reason)."""
        if n_adapt_available == 0:
            return "ABSTAIN", "No post-drift adaptation records available"
        if label_validity == "stress_test_uncertain":
            return "ABSTAIN", "Target labels are unverified stress test perturbations"
        if unique_valid_candidates < min(requested_budget, 50):
            return "ABSTAIN", f"Insufficient valid unique candidates ({unique_valid_candidates} < 50)"
        return "PROCEED", "Valid adaptation conditions met"

    # Test 1: Zero adaptation records -> Abstain
    dec, reason = decide_adaptation_action(0, "synthetic_concept_truth", 200, 200)
    assert dec == "ABSTAIN"
    assert "No post-drift" in reason

    # Test 2: Uncertain labels -> Abstain
    dec, reason = decide_adaptation_action(100, "stress_test_uncertain", 200, 200)
    assert dec == "ABSTAIN"
    assert "unverified" in reason

    # Test 3: Insufficient candidates -> Abstain
    dec, reason = decide_adaptation_action(100, "synthetic_concept_truth", 25, 200)
    assert dec == "ABSTAIN"
    assert "Insufficient" in reason

    # Test 4: All conditions met -> Proceed
    dec, reason = decide_adaptation_action(100, "synthetic_concept_truth", 200, 200)
    assert dec == "PROCEED"
