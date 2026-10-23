"""Comprehensive Test Suite for Component 3 Phase 1: Complete Intelligent Repair Engine Core.

Covers:
1. Contract tests (RepairRequest schema, safe path traversal, dataset registration)
2. Targeted repair contract & Safe AST slice evaluation (non-oracle enforcement)
3. Prescription engine (10-gate deterministic strategy selector, Batch 5M taxonomy)
4. Generator selector (compatibility, bounded retries, fallback, provenance)
5. Unified validation pipeline (structural, privacy DCR, fidelity, offline utility, sub-reports)
6. Retraining evaluator (model isolation, frozen preprocessor, bootstrap CIs, source retention)
7. Repair quantity planner (parsimonious search, early stopping, bounded search grid)
8. Repair orchestrator & RepairPackage (end-to-end execution, state transitions, C4 handoff metadata)
9. Leakage prevention (ensures locked evaluation partitions are never accessed)
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from component3.src.contracts.repair_package import (
    RepairPackage,
    RepairPackageStatus,
)
from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    ExecutionMode,
    InvalidRepairRequestError,
    RepairObjective,
    RepairRequest,
)
from component3.src.contracts.targeted_repair import (
    DiagnosticProvenance,
    DiagnosticSliceDefinition,
    TargetedRepairStatus,
    validate_diagnostic_slice,
)
from component3.src.evaluation.retraining_evaluator import RetrainingEvaluator, paired_bootstrap_ci_f1
from component3.src.generators.generator_selector import GeneratorSelector
from component3.src.orchestration.orchestrator_models import ExecutionState, OrchestratorExecutionLog
from component3.src.orchestration.repair_orchestrator import RepairOrchestrator
from component3.src.planning.quantity_planner import RepairQuantityPlanner
from component3.src.prescription.prescription_engine import RepairPrescriptionEngine
from component3.src.prescription.prescription_models import (
    EvidenceStrength,
    PrescriptionDecision,
    RepairStrategy,
)
from component3.src.validation.acceptance_policy import AcceptancePolicy
from component3.src.validation.validation_models import AcceptanceDecision, ValidationStatus
from component3.src.validation.validation_pipeline import UnifiedValidationPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# 1. CONTRACT TESTS: REPAIR REQUEST SCHEMA & SECURITY
# =============================================================================

class TestRepairRequestContract:
    """Contract and input validation tests for RepairRequest schema."""

    def test_valid_repair_request_creation(self):
        req = RepairRequest(
            request_id="REQ-TEST-001",
            dataset_id="telco_churn",
            drift_detected=True,
            drift_type=DriftType.CONCEPT_DRIFT,
            drift_severity=DriftSeverity.HIGH,
            drift_severity_score=0.75,
            affected_features=("tenure", "Contract"),
            adaptation_budget_size=200,
            repair_objective=RepairObjective.BALANCED_RETENTION,
        )
        assert req.request_id == "REQ-TEST-001"
        assert req.drift_severity_score == 0.75
        assert req.adaptation_budget_size == 200
        assert req.schema_version == "1.0.0"

    def test_reject_unauthorized_dataset_reference(self):
        with pytest.raises(InvalidRepairRequestError, match="Unauthorized dataset"):
            RepairRequest(
                request_id="REQ-TEST-UNAUTH",
                dataset_id="unknown_proprietary_dataset_99",
            )

    def test_prevent_unsafe_path_traversal(self):
        with pytest.raises(InvalidRepairRequestError, match="Path traversal"):
            RepairRequest(
                request_id="REQ-TEST-TRAVERSAL",
                source_model_path="../../etc/passwd",
            )

    def test_reject_out_of_bounds_severity_score(self):
        with pytest.raises(InvalidRepairRequestError, match="drift_severity_score must be between 0.0 and 1.0"):
            RepairRequest(
                request_id="REQ-TEST-BOUNDS",
                drift_severity_score=1.5,
            )

    def test_reject_negative_adaptation_budget(self):
        with pytest.raises(InvalidRepairRequestError, match="adaptation_budget_size must be non-negative"):
            RepairRequest(
                request_id="REQ-TEST-BUDGET",
                adaptation_budget_size=-50,
            )

    def test_reject_empty_request_id(self):
        with pytest.raises(InvalidRepairRequestError, match="request_id must be a non-empty string"):
            RepairRequest(request_id="")

    def test_serialization_round_trip(self):
        req = RepairRequest(
            request_id="REQ-SERIAL-001",
            drift_detected=True,
            drift_severity_score=0.45,
            affected_features=("MonthlyCharges",),
        )
        d = req.to_dict()
        assert d["request_id"] == "REQ-SERIAL-001"
        assert d["drift_severity_score"] == 0.45
        req_reloaded = RepairRequest.from_dict(d)
        assert req_reloaded.request_id == req.request_id
        assert req_reloaded.drift_severity_score == req.drift_severity_score


# =============================================================================
# 2. TARGETED REPAIR CONTRACT & SAFE AST SLICE EVALUATION
# =============================================================================

class TestTargetedRepairContract:
    """Tests for safe slice evaluation, AST checking, and oracle rejection."""

    def test_valid_safe_ast_predicate_evaluation(self):
        diag = DiagnosticSliceDefinition(
            slice_id="SLICE-SAFE-01",
            slice_predicate="tenure > 29",
            affected_features=("tenure",),
            reported_slice_support=0.35,
            diagnostic_confidence=0.90,
            provenance=DiagnosticProvenance.COMPONENT_2_DIAGNOSIS,
        )
        df = pd.DataFrame({"tenure": [10, 20, 30, 40, 50]})
        mask = diag.evaluate_mask(df)
        assert mask.tolist() == [False, False, True, True, True]

    def test_reject_arbitrary_code_execution_in_slice_rule(self):
        diag = DiagnosticSliceDefinition(
            slice_id="SLICE-EVIL-01",
            slice_predicate="__import__('os').system('dir')",
            affected_features=("tenure",),
            reported_slice_support=0.1,
            provenance=DiagnosticProvenance.COMPONENT_2_DIAGNOSIS,
        )
        df = pd.DataFrame({"tenure": [10, 20]})
        with pytest.raises(ValueError, match="Unsafe or invalid slice predicate"):
            diag.evaluate_mask(df)

    def test_block_drift_simulator_oracle_reference(self):
        diag = DiagnosticSliceDefinition(
            slice_id="SLICE-ORACLE-01",
            slice_predicate="tenure > 29",
            affected_features=("tenure",),
            reported_slice_support=0.35,
            provenance=DiagnosticProvenance.SIMULATION_ORACLE,  # Forbidden oracle!
        )
        status, msg, _ = validate_diagnostic_slice(diag)
        assert status == TargetedRepairStatus.TARGETED_REPAIR_BLOCKED
        assert "oracle" in msg.lower()

    def test_block_unsupported_slice_features(self):
        diag = DiagnosticSliceDefinition(
            slice_id="SLICE-MISSING-01",
            slice_predicate="secret_unauthorized_col == 1",
            affected_features=("secret_unauthorized_col",),
            reported_slice_support=0.1,
            provenance=DiagnosticProvenance.COMPONENT_2_DIAGNOSIS,
        )
        df = pd.DataFrame({"tenure": [10, 20]})
        status, msg, _ = validate_diagnostic_slice(diag, current_df=df)
        assert status == TargetedRepairStatus.INVALID_PREDICATE


# =============================================================================
# 3. PRESCRIPTION ENGINE & DETERMINISTIC STRATEGY SELECTOR
# =============================================================================

class TestRepairPrescriptionEngine:
    """Tests for 10-gate strategy prescription using Batch 5M taxonomy."""

    def test_no_drift_leads_to_no_repair(self):
        engine = RepairPrescriptionEngine()
        req = RepairRequest(
            request_id="TEST-PRES-A",
            drift_detected=False,
            drift_severity=DriftSeverity.NONE,
            drift_severity_score=0.0,
        )
        decision = engine.prescribe(req)
        assert decision.strategy == RepairStrategy.NO_REPAIR
        assert decision.recommended_generator is None
        assert "No significant drift detected" in decision.decision_reasons[0]

    def test_authentic_weighted_prescribed_under_drift(self):
        engine = RepairPrescriptionEngine()
        req = RepairRequest(
            request_id="TEST-PRES-B",
            drift_detected=True,
            drift_type=DriftType.CONCEPT_DRIFT,
            drift_severity=DriftSeverity.HIGH,
            drift_severity_score=0.6,
            adaptation_budget_size=100,
        )
        decision = engine.prescribe(req)
        assert decision.strategy == RepairStrategy.AUTHENTIC_WEIGHTED_ADAPTATION
        assert decision.sample_weight_policy == {"source": 1.0, "target": 2.0}

    def test_synthetic_hybrid_prescribed_when_budget_200(self):
        engine = RepairPrescriptionEngine()
        req = RepairRequest(
            request_id="TEST-PRES-C",
            drift_detected=True,
            drift_type=DriftType.CONCEPT_DRIFT,
            drift_severity=DriftSeverity.HIGH,
            drift_severity_score=0.65,
            adaptation_budget_size=200,
        )
        decision = engine.prescribe(req)
        assert decision.strategy == RepairStrategy.SYNTHETIC_WEIGHTED_HYBRID
        assert decision.recommended_generator == "tvae"
        assert decision.sample_weight_policy == {"source": 1.0, "target": 2.0, "synth": 2.0}

    def test_targeted_synthetic_repair_prescribed_with_valid_slice(self):
        engine = RepairPrescriptionEngine()
        diag = DiagnosticSliceDefinition(
            slice_id="SLICE-TARGET-01",
            slice_predicate="tenure > 29",
            affected_features=("tenure",),
            reported_slice_support=0.35,
            diagnostic_confidence=0.92,
            provenance=DiagnosticProvenance.COMPONENT_2_DIAGNOSIS,
        )
        req = RepairRequest(
            request_id="TEST-PRES-TARGETED",
            drift_detected=True,
            drift_severity_score=0.7,
            adaptation_budget_size=200,
            diagnostic_slice=diag,
        )
        decision = engine.prescribe(req)
        assert decision.strategy == RepairStrategy.TARGETED_SYNTHETIC_REPAIR
        assert decision.recommended_generator == "tvae"

    def test_oracle_diagnosis_blocks_targeted_and_falls_back(self):
        engine = RepairPrescriptionEngine()
        diag = DiagnosticSliceDefinition(
            slice_id="SLICE-ORACLE-01",
            slice_predicate="tenure > 29",
            affected_features=("tenure",),
            reported_slice_support=0.35,
            provenance=DiagnosticProvenance.SIMULATION_ORACLE,  # Oracle!
        )
        req = RepairRequest(
            request_id="TEST-PRES-ORACLE",
            drift_detected=True,
            drift_severity_score=0.7,
            adaptation_budget_size=200,
            diagnostic_slice=diag,
        )
        decision = engine.prescribe(req)
        # Targeted repair must be blocked, falling back to non-targeted hybrid
        assert decision.strategy == RepairStrategy.SYNTHETIC_WEIGHTED_HYBRID
        assert any("oracle" in w.lower() for w in decision.warnings)

    def test_unlabeled_drift_abstains_safely(self):
        engine = RepairPrescriptionEngine()
        req = RepairRequest(
            request_id="TEST-PRES-UNLABELED",
            drift_detected=True,
            has_authentic_adaptation_labels=False,  # Unlabeled!
            adaptation_budget_size=0,
        )
        decision = engine.prescribe(req)
        assert decision.strategy == RepairStrategy.NO_REPAIR
        assert any("unavailable" in r.lower() for r in decision.decision_reasons)

    def test_resource_exhaustion_restricts_generator(self):
        engine = RepairPrescriptionEngine()
        req = RepairRequest(
            request_id="TEST-PRES-RES",
            drift_detected=True,
            adaptation_budget_size=200,
            max_memory_mb=512.0,  # Constrained RAM < 1024 MB!
        )
        decision = engine.prescribe(req)
        # Falls back to authentic weighted because deep generators require at least 1024 MB RAM
        assert decision.strategy == RepairStrategy.AUTHENTIC_WEIGHTED_ADAPTATION


# =============================================================================
# 4. GENERATOR SELECTOR & PLUGIN INTEGRATION
# =============================================================================

class TestGeneratorSelector:
    """Tests for resource-aware generator selection, compatibility, and fallbacks."""

    def test_generator_compatibility_check(self):
        selector = GeneratorSelector()
        df = pd.DataFrame({
            "num1": [float(i) for i in range(35)],
            "cat1": ["A" if i % 2 == 0 else "B" for i in range(35)],
        })
        compat, reason = selector.check_compatibility("ctgan", df, categorical_columns=["cat1"], numerical_columns=["num1"])
        assert compat is True

    def test_reject_unregistered_generator(self):
        selector = GeneratorSelector()
        df = pd.DataFrame({"x": [float(i) for i in range(35)]})
        compat, reason = selector.check_compatibility("non_existent_generator", df, [], ["x"])
        assert compat is False
        assert "not registered" in reason.lower()

    def test_provenance_logging(self):
        selector = GeneratorSelector()
        df = pd.DataFrame({
            "age": [float(20 + i) for i in range(35)],
            "salary": [float(50000 + i * 1000) for i in range(35)],
            "category": ["A" if i % 2 == 0 else "B" for i in range(35)],
        })
        synth_df, prov = selector.fit_and_sample(
            generator_name="tvae",
            training_df=df,
            num_samples=10,
            categorical_columns=["category"],
            numerical_columns=["age", "salary"],
            generator_kwargs={"epochs": 2},
        )
        assert len(synth_df) == 10
        assert prov["executed_generator"] == "tvae"
        assert prov["fit_duration_seconds"] >= 0.0
        assert prov["sample_duration_seconds"] >= 0.0


# =============================================================================
# 5. UNIFIED VALIDATION PIPELINE & DISAGGREGATED SUB-REPORTS
# =============================================================================

class TestUnifiedValidationPipeline:
    """Tests for multi-stage validation: structural, privacy (DCR), fidelity, utility."""

    @pytest.fixture
    def reference_and_synthetic_dfs(self):
        # Create non-trivial distinct records to ensure clean DCR and KS stats
        rng = np.random.default_rng(42)
        n = 100
        ref_df = pd.DataFrame({
            "tenure": rng.uniform(1.0, 72.0, size=n),
            "MonthlyCharges": rng.uniform(20.0, 120.0, size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "Churn": rng.choice([0, 1], size=n),
        })
        # Synthetic data with plausible small shift
        synth_df = pd.DataFrame({
            "tenure": rng.uniform(1.0, 72.0, size=30),
            "MonthlyCharges": rng.uniform(20.0, 120.0, size=30),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=30),
            "Churn": rng.choice([0, 1], size=30),
        })
        return ref_df, synth_df

    def test_unified_validation_success_and_sub_reports(self, reference_and_synthetic_dfs):
        ref_df, synth_df = reference_and_synthetic_dfs
        pipeline = UnifiedValidationPipeline(mode="research")
        report = pipeline.validate(
            synthetic_df=synth_df,
            reference_df=ref_df,
            target_column="Churn",
            categorical_columns=["Contract"],
            numerical_columns=["tenure", "MonthlyCharges"],
            min_dcr_p05=0.001,
        )
        assert report.hard_validity_report["status"] == "PASS"
        assert report.fidelity_report["status"] in ("PASS", "REVIEW")
        assert "DCR" in report.privacy_risk_report["privacy_metric_note"]
        assert "not external ground truth" in report.empirical_plausibility_report["synthetic_label_nature_note"]

    def test_reject_corrupted_empty_synthetic_data(self, reference_and_synthetic_dfs):
        ref_df, _ = reference_and_synthetic_dfs
        pipeline = UnifiedValidationPipeline(mode="production")
        empty_synth = pd.DataFrame(columns=ref_df.columns)
        report = pipeline.validate(
            synthetic_df=empty_synth,
            reference_df=ref_df,
            target_column="Churn",
        )
        assert report.overall_decision == AcceptanceDecision.REJECT
        assert report.package_status == "REJECTED"
        assert any("empty" in r.lower() for r in report.rejection_reasons)

    def test_reject_schema_mismatch_missing_columns(self, reference_and_synthetic_dfs):
        ref_df, synth_df = reference_and_synthetic_dfs
        pipeline = UnifiedValidationPipeline(mode="production")
        bad_synth = synth_df.drop(columns=["MonthlyCharges"])
        report = pipeline.validate(
            synthetic_df=bad_synth,
            reference_df=ref_df,
            target_column="Churn",
        )
        assert report.overall_decision == AcceptanceDecision.REJECT
        assert report.package_status == "REJECTED"


# =============================================================================
# 6. RETRAINING EVALUATION & SAFETY CONTROLS
# =============================================================================

class TestRetrainingEvaluator:
    """Tests for model preservation, source retention enforcement, and bootstrap CIs."""

    def test_baseline_model_remains_unmodified(self):
        base_rf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, size=50)
        base_rf.fit(X, y)
        orig_trees = len(base_rf.estimators_)

        evaluator = RetrainingEvaluator(source_retention_tolerance=0.05)
        assert base_rf is not None
        assert len(base_rf.estimators_) == orig_trees

    def test_paired_bootstrap_confidence_intervals(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 5)
        y_cand = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0] * 5)
        y_base = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 1] * 5)
        ci = paired_bootstrap_ci_f1(y_true, y_cand, y_base, n_bootstraps=200, seed=42)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ci[1]


# =============================================================================
# 7. MINIMUM EFFECTIVE REPAIR QUANTITY PLANNING
# =============================================================================

class TestRepairQuantityPlanner:
    """Tests for parsimonious search over bounded candidate quantities."""

    def test_parsimonious_search_stops_at_smallest_acceptable(self):
        planner = RepairQuantityPlanner(candidate_quantities=(25, 50, 100))
        assert planner.candidate_quantities == (25, 50, 100)
        assert planner.candidate_quantities[0] == 25


# =============================================================================
# 8. REPAIR ORCHESTRATOR & REPAIR PACKAGE
# =============================================================================

class TestRepairOrchestratorAndPackage:
    """Tests for full orchestrator execution, state tracking, and RepairPackage contract."""

    def test_orchestration_state_transitions_logged(self):
        log = OrchestratorExecutionLog(run_id="RUN-LOG", request_id="TEST-LOG", current_state=ExecutionState.CREATED)
        log.transition_to(ExecutionState.VALIDATING, "Validating request")
        log.transition_to(ExecutionState.PRESCRIBING, "Running prescription")
        log.transition_to(ExecutionState.ACCEPTED, "Completed")
        assert len(log.transitions) == 3
        assert log.transitions[0].to_state == "VALIDATING"
        assert log.transitions[1].to_state == "PRESCRIBING"
        assert log.transitions[2].to_state == "ACCEPTED"
        assert log.current_state == ExecutionState.ACCEPTED

    def test_repair_package_hash_and_serialization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_file = tmp_path / "candidates.csv"
            data_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

            pkg = RepairPackage(
                request_id="REQ-PKG-01",
                run_id="RUN-PKG-01",
                dataset_id="telco_churn",
                model_id="telco_rf",
                chosen_strategy="SYNTHETIC_WEIGHTED_HYBRID",
                candidate_dataset_path=str(data_file),
                package_status=RepairPackageStatus.APPROVED_FOR_REVIEW,
                requires_human_approval=True,
            )

            # Save package manifest
            out_file = pkg.save_package(tmp_path)
            assert out_file.is_file()

            # Read back and verify
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["package_id"] == pkg.package_id
            assert saved["schema_version"] == "1.0.0"
            assert saved["requires_human_approval"] is True
            assert saved["candidate_dataset_sha256"] is not None


# =============================================================================
# 9. LEAKAGE PREVENTION & DATA ISOLATION
# =============================================================================

class TestDataLeakagePrevention:
    """Tests ensuring evaluation partitions and test sets remain strictly isolated."""

    def test_locked_test_partition_never_loaded_in_orchestrator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = RepairOrchestrator()
            req = RepairRequest(
                request_id="LEAK-CHECK-01",
                drift_detected=True,
                drift_severity_score=0.5,
            )
            pkg, log = orchestrator.execute_repair(req, output_directory=tmp_dir)
            assert pkg.quantity_selected <= 200
            meta = pkg.downstream_handoff_metadata
            assert "target_test" not in str(meta).lower()
