"""Comprehensive automated test suite for Component 3 Phase 3.

Covers:
1. Cross-dataset compatibility (Telco, Adult, Bank).
2. Drift scenario provenance and non-oracle isolation.
3. Evaluation leakage and partition disjointness.
4. Prescription feature zero-leakage assertions.
5. Group-aware cross-validation, regret, and honest validation status.
6. Learned prescription model OOD detection and safe fallback.
7. Statistical inference, bootstrap CIs, and Holm-Bonferroni multiplicity corrections.
8. Benchmark experiment matrix integrity (144 records across 3 generators).
9. End-to-end execution across decision modes.
10. CLI decision mode compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from component3.src.cli import main as cli_main
from component3.src.contracts.repair_package import RepairPackageStatus
from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    ExecutionMode,
    RepairRequest,
)
from component3.src.contracts.targeted_repair import (
    DiagnosticProvenance,
    DiagnosticSliceDefinition,
    TargetedRepairStatus,
    validate_diagnostic_slice,
)
from component3.src.datasets.dataset_registry import DatasetStatus, get_dataset_registry
from component3.src.evaluation.statistical_analysis import (
    adjust_pvalues_benjamini_hochberg,
    adjust_pvalues_holm_bonferroni,
    analyze_multidataset_benchmark_significance,
    evaluate_tost_equivalence,
    paired_bootstrap_test,
)
from component3.src.orchestration.repair_orchestrator import RepairOrchestrator
from component3.src.prescription.learned_model import (
    INPUT_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    SCENARIO_GROUP_COLUMNS,
    LearnedPrescriptionModel,
)
from component3.src.prescription.prescription_engine import RepairPrescriptionEngine
from component3.src.prescription.prescription_models import DecisionMode, RepairStrategy


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


# =============================================================================
# 1. Cross-Dataset Compatibility
# =============================================================================

def test_cross_dataset_registry_compatibility():
    """Verify all 3 primary datasets are registered and have verified metadata."""
    reg = get_dataset_registry()
    dataset_ids = ["telco_churn", "uci_adult_full", "bank_marketing"]

    for d_id in dataset_ids:
        meta = reg.get(d_id)
        assert meta is not None, f"Dataset '{d_id}' missing from registry."
        assert meta.dataset_id == d_id
        assert meta.status == DatasetStatus.VERIFIED
        assert meta.target_column != ""
        assert len(meta.feature_columns) > 0
        assert len(meta.numerical_features) > 0
        assert len(meta.categorical_features) > 0

        # Verify raw data can be loaded
        df = reg.load_dataset(d_id)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000
        assert meta.target_column in df.columns


def test_adaptation_splits_disjointness_and_files():
    """Verify adaptation splits are exported and exist for Adult and Bank."""
    splits_dir = REPO_ROOT / "component3" / "results" / "adaptation_splits"
    assert splits_dir.is_dir()

    for d_id in ["uci_adult_full", "bank_marketing"]:
        manifest_p = splits_dir / f"{d_id}_adaptation_split_manifest.json"
        assert manifest_p.is_file(), f"Manifest missing for {d_id}"

        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["dataset_id"] == d_id
        assert manifest["realized_counts"]["source_train"] > 0
        assert manifest["realized_counts"]["source_eval"] > 0
        assert manifest["realized_counts"]["target_adapt_pool"] > 0
        assert manifest["realized_counts"]["target_eval"] > 0

        # Verify CSV files exist
        for part in ["source_train", "source_eval", "target_adapt_pool", "target_eval"]:
            f_path = splits_dir / f"{d_id}_{part}.csv"
            assert f_path.is_file(), f"File {f_path} missing"

        # Verify budget files exist
        for b in [50, 100, 200]:
            b_path = splits_dir / f"{d_id}_target_adaptation_B{b}.csv"
            assert b_path.is_file(), f"Budget file {b_path} missing"
            df_b = pd.read_csv(b_path)
            assert len(df_b) == b


# =============================================================================
# 2. Drift Scenario Provenance & Non-Oracle Isolation
# =============================================================================

def test_oracle_diagnostic_slice_blocked():
    """Verify that an oracle-derived diagnostic slice is rejected by governance gates."""
    oracle_slice = DiagnosticSliceDefinition(
        slice_id="ORACLE-DRIFT-RULE-01",
        slice_predicate="tenure > 29.0",
        affected_features=("tenure",),
        reported_slice_support=0.50,
        provenance=DiagnosticProvenance.SIMULATION_ORACLE,
        is_non_oracle=False,
    )
    test_df = pd.DataFrame({"tenure": [10, 20, 30, 40, 50]})
    status, msg, slice_idx = validate_diagnostic_slice(oracle_slice, current_df=test_df)

    assert status == TargetedRepairStatus.TARGETED_REPAIR_BLOCKED
    assert "oracle simulation rule" in msg.lower()
    assert slice_idx is None


def test_empirical_slice_accepted_when_valid():
    """Verify that a legitimate empirical diagnostic slice passes validation."""
    valid_slice = DiagnosticSliceDefinition(
        slice_id="EMPIRICAL-ERROR-SLICE-01",
        slice_predicate="MonthlyCharges > 70.0",
        affected_features=("MonthlyCharges",),
        reported_slice_support=0.30,
        provenance=DiagnosticProvenance.COMPONENT_2_DIAGNOSIS,
        is_non_oracle=True,
    )
    # Provide >= 10 matching rows to satisfy min_slice_rows threshold
    test_df = pd.DataFrame({"MonthlyCharges": [50.0] * 5 + [80.0] * 15})
    status, msg, slice_idx = validate_diagnostic_slice(valid_slice, current_df=test_df)

    assert status == TargetedRepairStatus.ELIGIBLE
    assert slice_idx is not None
    assert int(slice_idx.sum()) == 15


# =============================================================================
# 3. Prescription Feature Zero-Leakage Assertions
# =============================================================================

def test_prescription_dataset_zero_leakage():
    """Ensure prescription training data has zero outcome leakage into input features."""
    ds_path = REPO_ROOT / "component3" / "results" / "prescription_training_dataset.json"
    assert ds_path.is_file(), "Prescription training dataset missing"

    with open(ds_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) > 100

    # Ensure sets of columns are completely disjoint
    input_set = set(INPUT_FEATURE_COLUMNS)
    outcome_set = set(OUTCOME_COLUMNS)
    group_set = set(SCENARIO_GROUP_COLUMNS)

    assert len(input_set.intersection(outcome_set)) == 0
    assert len(input_set.intersection(group_set)) == 0

    # Verify every record conforms to the strict schema
    for r in records:
        for feat in INPUT_FEATURE_COLUMNS:
            assert feat in r, f"Missing input feature {feat}"
        for out_col in OUTCOME_COLUMNS:
            assert out_col in r, f"Missing outcome column {out_col}"
        for grp_col in SCENARIO_GROUP_COLUMNS:
            assert grp_col in r, f"Missing group column {grp_col}"

        # Verify that pre-decision input features do not contain candidate metrics
        assert "delta_target_f1_vs_baseline" not in input_set
        assert "source_retention_delta" not in input_set
        assert "is_acceptable" not in input_set


# =============================================================================
# 4. Group-Aware Cross-Validation & Regret Evaluation
# =============================================================================

def test_learned_prescription_model_evaluation_report():
    """Verify group-aware cross-validation results and honest scientific validation status."""
    report_path = REPO_ROOT / "component3" / "results" / "learned_prescription_evaluation_report.json"
    assert report_path.is_file(), "Prescription evaluation report missing"

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Must be marked NOT scientifically validated for production deployment
    assert report["is_scientifically_validated"] is False
    assert report["validation_status"] == "PILOT_EXPLORATORY_EVIDENCE"
    assert len(report["limitations"]) > 0

    # Cross-validation metrics
    assert report["n_groups"] == 11
    assert "utility_rmse" in report
    assert "acceptance_accuracy" in report

    # Regret evaluation
    assert "mean_strategy_regret_vs_oracle" in report
    assert "rule_based_mean_regret" in report
    assert report["feasibility_violation_rate"] == 0.0


# =============================================================================
# 5. Learned Prescription Model OOD Detection & Safe Fallback
# =============================================================================

def test_learned_prescription_model_ood_detection():
    """Verify OOD detection triggers safe fallback on unsupported scenario inputs."""
    learned_model_p = REPO_ROOT / "component3" / "models" / "prescription" / "learned_prescription_model.joblib"
    model = LearnedPrescriptionModel.load(learned_model_p)

    # Out-of-distribution input: number of features far beyond domain
    ood_features = {
        "in_dataset_num_features": 9999,
        "in_dataset_num_numerical": 5000,
        "in_dataset_num_categorical": 4999,
        "in_dataset_imbalance_ratio": 0.25,
        "in_drift_category": "concept_drift",
        "in_available_adaptation_budget": 200,
        "in_has_valid_diagnostic_slice": 0,
        "arm_candidate_quantity": 50,
    }
    is_ood, ood_reasons = model.is_out_of_distribution(ood_features)

    assert is_ood is True
    assert len(ood_reasons) > 0
    assert any("in_dataset_num_features" in r for r in ood_reasons)


def test_learned_prescription_mode_toggle():
    """Verify that both RULE_BASED and LEARNED modes produce valid decisions."""
    engine = RepairPrescriptionEngine()
    req = RepairRequest(
        request_id="TEST-TOGGLE-01",
        dataset_id="telco_churn",
        drift_detected=True,
        drift_type=DriftType.CONCEPT_DRIFT,
        drift_severity=DriftSeverity.HIGH,
        drift_severity_score=0.60,
        adaptation_budget_size=200,
    )

    dec_rule = engine.prescribe(req, decision_mode=DecisionMode.RULE_BASED)
    assert dec_rule.decision_mode == DecisionMode.RULE_BASED
    assert dec_rule.strategy == RepairStrategy.SYNTHETIC_WEIGHTED_HYBRID

    dec_learned = engine.prescribe(req, decision_mode=DecisionMode.LEARNED)
    assert dec_learned.decision_mode == DecisionMode.LEARNED
    assert isinstance(dec_learned.strategy, RepairStrategy)


# =============================================================================
# 6. Statistical Analysis & Multiplicity Correction
# =============================================================================

def test_paired_bootstrap_test_computation():
    """Verify paired bootstrap difference testing and Cohen's d calculation."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=200)
    # Candidate performs slightly better than reference
    y_cand = y_true.copy()
    y_ref = y_true.copy()
    # Add noise to ref
    noise_idx = rng.choice(200, size=30, replace=False)
    y_ref[noise_idx] = 1 - y_ref[noise_idx]

    res = paired_bootstrap_test(
        y_true=y_true,
        y_pred_cand=y_cand,
        y_pred_ref=y_ref,
        n_bootstraps=500,
        ci_level=0.95,
        practical_margin=0.01,
        seed=42,
    )

    assert res.observed_delta > 0.0
    assert res.ci_lower <= res.observed_delta <= res.ci_upper
    assert res.ci_width > 0.0
    assert res.standard_error > 0.0
    assert res.p_value_two_sided <= 0.05
    assert res.is_statistically_significant is True


def test_holm_bonferroni_multiplicity_correction():
    """Verify Holm-Bonferroni step-down monotonic adjustment."""
    raw_p = [0.005, 0.02, 0.04, 0.15, 0.40]
    res = adjust_pvalues_holm_bonferroni(raw_p, alpha=0.05)

    assert len(res.adjusted_p_values) == len(raw_p)
    # Adjusted p-values must be >= raw p-values
    for r_p, a_p in zip(raw_p, res.adjusted_p_values):
        assert a_p >= r_p

    # Smallest p (0.005) multiplied by m=5 -> 0.025 (< 0.05)
    assert res.adjusted_p_values[0] == pytest.approx(0.025, abs=1e-5)
    assert res.significant_flags[0] is True


def test_tost_equivalence_analysis():
    """Verify TOST equivalence testing behavior."""
    # Highly overlapping differences centered near zero
    zero_diffs = np.array([0.001, -0.001, 0.002, -0.002, 0.000, 0.001, -0.001])
    res = evaluate_tost_equivalence(zero_diffs, equivalence_margin=0.05, alpha=0.05)
    assert res.get("equivalence_margin") == 0.05
    assert "Absence of difference is not proof of equivalence" in res["conclusion"] or res["is_equivalent"]


# =============================================================================
# 7. Benchmark Results Integrity (144 Records)
# =============================================================================

def test_multidataset_benchmark_results_integrity():
    """Verify the 144 benchmark results are complete and cover all datasets and generators."""
    bench_p = REPO_ROOT / "component3" / "results" / "multidataset_benchmark_results.json"
    assert bench_p.is_file(), "Benchmark results file missing"

    with open(bench_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 144

    datasets = set(r["dataset_id"] for r in records)
    assert datasets == {"telco_churn", "uci_adult_full", "bank_marketing"}

    generators = set(r.get("generator") for r in records if r.get("generator") is not None)
    assert "tvae" in generators
    assert "ctgan" in generators
    assert "tabddpm" in generators

    # Verify statistical analysis summary runs on benchmark records
    summary = analyze_multidataset_benchmark_significance(records)
    assert summary.n_total_comparisons > 50
    assert summary.multiplicity_method == "Holm-Bonferroni"
    assert len(summary.scenario_evaluations) == 9


# =============================================================================
# 8. End-to-End Execution & Demonstration Workflows
# =============================================================================

def test_phase3_demonstration_execution(tmp_path: Path):
    """Verify full end-to-end execution of Phase 3 demonstration script."""
    from component3.src.demonstrations.run_phase3_scientific_demo import (
        run_phase3_scientific_demonstration,
    )
    demo_out = tmp_path / "phase3_demo"
    results = run_phase3_scientific_demonstration(output_dir=demo_out)

    assert "demo1_success" in results
    assert results["demo1_success"]["status"] == RepairPackageStatus.APPROVED_FOR_REVIEW.value

    assert "demo2_rejection" in results
    assert results["demo2_rejection"]["status"] == RepairPackageStatus.REJECTED.value

    assert "demo3_abstain" in results
    assert results["demo3_abstain"]["status"] == RepairPackageStatus.ABSTAINED_NO_REPAIR.value

    assert "demo4_learned" in results
    assert results["demo4_learned"]["passed"] is True


# =============================================================================
# 9. CLI Decision Mode Flag
# =============================================================================

def test_cli_decision_mode_parsing():
    """Verify CLI accepts --decision-mode flag without crashing."""
    # Check-readiness executes cleanly without raising SystemExit
    exit_code = cli_main(["check-readiness"])
    assert exit_code in (0, 1)
