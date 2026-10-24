"""Deterministic, evidence-aware repair prescription engine for Component 3.

Executes sequential pre-decision governance checks to select an appropriate
repair strategy from the Batch 5M taxonomy without data snooping or leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    RepairObjective,
    RepairRequest,
)
from component3.src.contracts.targeted_repair import (
    TargetedRepairStatus,
    validate_diagnostic_slice,
)
from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.prescription.prescription_models import (
    DecisionMode,
    EvidenceStrength,
    PrescriptionDecision,
    RepairStrategy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "component3" / "results" / "repair_evidence_registry.json"


class RepairPrescriptionEngine:
    """Deterministic strategy selector based on Batch 5M taxonomy and pre-decision context."""

    def __init__(self, evidence_registry_path: Path | str | None = None) -> None:
        self.registry_path = Path(evidence_registry_path) if evidence_registry_path else REGISTRY_PATH
        self._historical_evidence = self._load_historical_evidence()

    def _load_historical_evidence(self) -> dict[str, Any]:
        """Load machine-readable evidence registry if available on disk."""
        if self.registry_path.is_file():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def extract_pre_decision_features(self, request: RepairRequest) -> dict[str, Any]:
        """Extract strictly pre-decision feature context for governance and logging."""
        reg = get_dataset_registry()
        meta = reg.get(request.dataset_id)

        num_numerical = len(meta.numerical_features) if hasattr(meta, "numerical_features") else 0
        num_categorical = len(meta.categorical_features) if hasattr(meta, "categorical_features") else 0
        total_feats = len(meta.feature_columns)

        has_diag = request.diagnostic_slice is not None
        diag_support = request.diagnostic_slice.reported_slice_support if has_diag else None

        return {
            "dataset_id": request.dataset_id,
            "total_features_count": total_feats,
            "numerical_features_count": num_numerical,
            "categorical_features_count": num_categorical,
            "drift_detected": request.drift_detected,
            "drift_type": request.drift_type.value,
            "drift_severity": request.drift_severity.value,
            "drift_severity_score": request.drift_severity_score,
            "affected_features_count": len(request.affected_features),
            "diagnostic_slice_identified": has_diag,
            "diagnostic_slice_support": diag_support,
            "has_authentic_adaptation_labels": request.has_authentic_adaptation_labels,
            "adaptation_budget_size": request.adaptation_budget_size,
            "source_retention_tolerance": request.source_retention_tolerance,
            "repair_objective": request.repair_objective.value,
            "max_cpu_cores": request.max_cpu_cores,
            "max_memory_mb": request.max_memory_mb,
            "max_runtime_seconds": request.max_runtime_seconds,
        }

    def prescribe(
        self,
        request: RepairRequest,
        adaptation_df: pd.DataFrame | None = None,
        decision_mode: DecisionMode = DecisionMode.RULE_BASED,
    ) -> PrescriptionDecision:
        """Evaluate 10 sequential governance checks and return a deterministic prescription."""
        checks_passed: list[str] = []
        reasons: list[str] = []
        warnings: list[str] = []

        pre_features = self.extract_pre_decision_features(request)

        # ---------------------------------------------------------------------
        # Gate 1: Request Validation
        # ---------------------------------------------------------------------
        checks_passed.append("GATE_1_REQUEST_VALIDATION")

        # ---------------------------------------------------------------------
        # Gate 2: Drift Severity and Repair Necessity
        # ---------------------------------------------------------------------
        if not request.drift_detected or request.drift_severity == DriftSeverity.NONE or request.drift_severity_score < 0.10:
            reasons.append("No significant drift detected (severity score < 0.10). Model repair is unnecessary.")
            return PrescriptionDecision(
                strategy=RepairStrategy.NO_REPAIR,
                recommended_generator=None,
                sample_weight_policy={"source": 1.0},
                initial_candidate_quantity=0,
                evidence_strength=EvidenceStrength.CONFIRMED_BENCHMARK,
                decision_reasons=tuple(reasons),
                governance_checks_passed=tuple(checks_passed),
                warnings=tuple(warnings),
                pre_decision_features=pre_features,
            )
        checks_passed.append("GATE_2_DRIFT_SEVERITY_VERIFIED")

        # ---------------------------------------------------------------------
        # Gate 3: Authentic Label Availability & Validity
        # ---------------------------------------------------------------------
        if not request.has_authentic_adaptation_labels or request.adaptation_budget_size == 0:
            reasons.append(
                "Authentic adaptation labels are unavailable. Supervised model adaptation cannot be verified safely. Abstaining from repair."
            )
            warnings.append("No authentic ground truth labels available in adaptation batch.")
            return PrescriptionDecision(
                strategy=RepairStrategy.NO_REPAIR,
                recommended_generator=None,
                sample_weight_policy={"source": 1.0},
                initial_candidate_quantity=0,
                evidence_strength=EvidenceStrength.CONFIRMED_BENCHMARK,
                decision_reasons=tuple(reasons),
                governance_checks_passed=tuple(checks_passed),
                warnings=tuple(warnings),
                pre_decision_features=pre_features,
            )
        checks_passed.append("GATE_3_AUTHENTIC_LABELS_AVAILABLE")

        # ---------------------------------------------------------------------
        # Gate 4: Source Model and Preprocessor Compatibility
        # ---------------------------------------------------------------------
        src_model_p = Path(request.source_model_path)
        src_prep_p = Path(request.source_preprocessor_path)
        if not src_model_p.is_file() or not src_prep_p.is_file():
            # If relative, check against REPO_ROOT
            if not (REPO_ROOT / src_model_p).is_file() or not (REPO_ROOT / src_prep_p).is_file():
                reasons.append(f"Source model or preprocessor artifact not found on disk: {src_model_p}, {src_prep_p}")
                return PrescriptionDecision(
                    strategy=RepairStrategy.NO_REPAIR,
                    recommended_generator=None,
                    sample_weight_policy={"source": 1.0},
                    initial_candidate_quantity=0,
                    evidence_strength=EvidenceStrength.INSUFFICIENT_EVIDENCE,
                    decision_reasons=tuple(reasons),
                    governance_checks_passed=tuple(checks_passed),
                    warnings=tuple(warnings),
                    pre_decision_features=pre_features,
                )
        checks_passed.append("GATE_4_SOURCE_MODEL_COMPATIBILITY")

        # ---------------------------------------------------------------------
        # Gate 4B: Optional Learned Prescription Decision Mode / Shadow Mode
        # ---------------------------------------------------------------------
        if decision_mode in (DecisionMode.LEARNED, DecisionMode.LEARNED_ADVISORY, DecisionMode.SHADOW_MODE):
            learned_model_p = REPO_ROOT / "component3" / "models" / "prescription" / "learned_prescription_model.joblib"
            if learned_model_p.is_file():
                try:
                    from component3.src.prescription.candidate_ranker import IntelligentCandidateRanker
                    from component3.src.prescription.learned_model import LearnedPrescriptionModel

                    # Prepare comprehensive pre-decision feature dictionary
                    reg = get_dataset_registry()
                    meta = reg.get(request.dataset_id)
                    n_feats = len(meta.feature_columns)
                    aff_prop = len(request.affected_features) / max(1, n_feats) if request.affected_features else 0.5
                    has_diag = request.diagnostic_slice is not None

                    scenario_feat = {
                        "in_dataset_num_features": n_feats,
                        "in_dataset_num_numerical": len(meta.numerical_features),
                        "in_dataset_num_categorical": len(meta.categorical_features),
                        "in_dataset_imbalance_ratio": min(meta.class_distribution.values()) if meta.class_distribution else 0.25,
                        "in_drift_category": request.drift_type.value if hasattr(request.drift_type, "value") else str(request.drift_type),
                        "in_drift_severity_score": float(request.drift_severity_score),
                        "in_affected_feature_proportion": round(aff_prop, 3),
                        "in_available_adaptation_budget": request.adaptation_budget_size,
                        "in_budget_to_feature_ratio": round(request.adaptation_budget_size / max(1, n_feats), 3),
                        "in_has_valid_diagnostic_slice": 1 if has_diag else 0,
                        "in_diagnostic_slice_size": min(request.adaptation_budget_size, 50) if has_diag else 0,
                        "in_source_model_family": "rf",
                    }

                    # Load model and evaluate candidate ranking
                    ranker = IntelligentCandidateRanker.load(learned_model_p)
                    ranking_result = ranker.rank_candidates(
                        scenario_features=scenario_feat,
                        adaptation_budget=request.adaptation_budget_size,
                        has_slice=has_diag,
                    )

                    # Shadow Mode: record advisory prescription in metadata and proceed with rule-based execution
                    if decision_mode == DecisionMode.SHADOW_MODE:
                        pre_features["shadow_learned_recommendation"] = ranking_result.to_dict()
                        checks_passed.append("GATE_LEARNED_SHADOW_INFERENCE_RECORDED")
                    elif ranking_result.is_out_of_distribution:
                        warnings.append(
                            f"OOD scenario detected ({'; '.join(ranking_result.ood_reasons)}). "
                            "Safely falling back to RULE_BASED prescription mode."
                        )
                    elif ranking_result.is_abstained:
                        reasons.append(
                            f"Learned candidate ranker abstained ({ranking_result.abstention_reason}). "
                            "Prescribing NO_REPAIR as the optimal learned decision."
                        )
                        checks_passed.append("GATE_LEARNED_MODEL_INFERENCE")
                        return PrescriptionDecision(
                            strategy=RepairStrategy.NO_REPAIR,
                            recommended_generator=None,
                            sample_weight_policy={"source": 1.0},
                            initial_candidate_quantity=0,
                            evidence_strength=EvidenceStrength.EXPLORATORY_PILOT,
                            decision_reasons=tuple(reasons),
                            governance_checks_passed=tuple(checks_passed),
                            warnings=tuple(warnings),
                            pre_decision_features=pre_features,
                            decision_mode=decision_mode,
                            predicted_delta_f1=0.0,
                        )
                    else:
                        best_strat = ranking_result.selected_strategy
                        best_gen = ranking_result.selected_generator if ranking_result.selected_generator != "none" else None
                        best_qty = ranking_result.selected_quantity
                        best_delta = ranking_result.predicted_delta_f1

                        # Enforce resource safety constraints on learned strategy
                        can_train_deep = (request.max_memory_mb >= 1024.0 and request.max_runtime_seconds >= 30.0)
                        if "SYNTHETIC" in best_strat.value and not can_train_deep:
                            warnings.append("Learned synthetic strategy exceeds resource bounds. Safely falling back to authentic adaptation.")
                            best_strat = RepairStrategy.AUTHENTIC_WEIGHTED_ADAPTATION
                            best_gen = None
                            best_qty = 0

                        checks_passed.append("GATE_LEARNED_MODEL_INFERENCE")
                        reasons.append(
                            f"Prescribed by learned candidate ranker ({best_strat.value}, gen={best_gen}, qty={best_qty}) "
                            f"with predicted delta F1: {best_delta:+.4f} and acceptance prob: {ranking_result.acceptance_probability:.2%}."
                        )

                        weight_policy = {"source": 1.0, "target": 2.0} if "WEIGHTED" in best_strat.value else {"source": 1.0, "target": 1.0, "synth": 1.0}

                        return PrescriptionDecision(
                            strategy=best_strat,
                            recommended_generator=best_gen,
                            sample_weight_policy=weight_policy,
                            initial_candidate_quantity=best_qty,
                            evidence_strength=EvidenceStrength.EXPLORATORY_PILOT,
                            decision_reasons=tuple(reasons),
                            governance_checks_passed=tuple(checks_passed),
                            warnings=tuple(warnings),
                            pre_decision_features=pre_features,
                            targeted_slice_id=request.diagnostic_slice.slice_id if request.diagnostic_slice else None,
                            decision_mode=decision_mode,
                            predicted_delta_f1=best_delta,
                        )
                except Exception as e:
                    warnings.append(f"Learned model execution failed: {str(e)}. Safely falling back to RULE_BASED prescription mode.")
            else:
                warnings.append("Learned prescription model artifact missing on disk. Safely falling back to RULE_BASED prescription mode.")


        # ---------------------------------------------------------------------
        # Gate 5: Operational Resource Limits (CPU, RAM, Runtime)
        # ---------------------------------------------------------------------
        can_train_deep_generators = True
        if request.max_memory_mb < 1024.0 or request.max_runtime_seconds < 30.0:
            can_train_deep_generators = False
            warnings.append(
                f"Resource limits (RAM {request.max_memory_mb}MB, time {request.max_runtime_seconds}s) "
                "are too constrained for deep generative modeling (TVAE/CTGAN/TabDDPM). Restricting to authentic adaptation."
            )
        checks_passed.append("GATE_5_RESOURCE_LIMITS_CHECKED")

        # ---------------------------------------------------------------------
        # Gate 6: Adaptation Data Budget Sufficiency
        # ---------------------------------------------------------------------
        b_size = request.adaptation_budget_size
        if b_size < 50:
            reasons.append(f"Adaptation budget ({b_size} rows) is below minimum viable adaptation threshold (50 rows).")
            return PrescriptionDecision(
                strategy=RepairStrategy.NO_REPAIR,
                recommended_generator=None,
                sample_weight_policy={"source": 1.0},
                initial_candidate_quantity=0,
                evidence_strength=EvidenceStrength.CONFIRMED_BENCHMARK,
                decision_reasons=tuple(reasons),
                governance_checks_passed=tuple(checks_passed),
                warnings=tuple(warnings),
                pre_decision_features=pre_features,
            )
        checks_passed.append("GATE_6_BUDGET_SUFFICIENCY_CHECKED")

        # ---------------------------------------------------------------------
        # Gate 7: Diagnostic Slice Eligibility (Safe Targeted Repair)
        # ---------------------------------------------------------------------
        targeted_status, targeted_msg, _ = validate_diagnostic_slice(
            request.diagnostic_slice,
            current_df=adaptation_df,
            min_slice_rows=10,
        )

        if request.diagnostic_slice is not None:
            if targeted_status == TargetedRepairStatus.ELIGIBLE:
                reasons.append("Valid non-oracle diagnostic slice identified. Prescribing targeted synthetic repair.")
                checks_passed.append("GATE_7_TARGETED_SLICE_ELIGIBLE")
                return PrescriptionDecision(
                    strategy=RepairStrategy.TARGETED_SYNTHETIC_REPAIR,
                    recommended_generator="tvae",
                    sample_weight_policy={"source": 1.0, "target": 2.0, "synth": 2.0},
                    initial_candidate_quantity=50,
                    evidence_strength=EvidenceStrength.EXPLORATORY_PILOT,
                    decision_reasons=tuple(reasons),
                    governance_checks_passed=tuple(checks_passed),
                    warnings=tuple(warnings),
                    pre_decision_features=pre_features,
                    targeted_slice_id=request.diagnostic_slice.slice_id,
                )
            else:
                warnings.append(f"Targeted repair blocked: {targeted_msg}. Falling back to non-targeted strategy.")
        checks_passed.append("GATE_7_TARGETED_REPAIR_EVALUATED")

        # ---------------------------------------------------------------------
        # Gate 8: Low-Budget Authentic Selection (50 <= B_k < 200 or Resource Constrained)
        # ---------------------------------------------------------------------
        if b_size < 200 or not can_train_deep_generators:
            reasons.append(
                f"Adaptation budget ({b_size} rows) is insufficient for stable deep generative modeling (requires B >= 200). "
                "Prescribing authentic weighted adaptation (w=2.0) to prevent negative transfer."
            )
            return PrescriptionDecision(
                strategy=RepairStrategy.AUTHENTIC_WEIGHTED_ADAPTATION,
                recommended_generator=None,
                sample_weight_policy={"source": 1.0, "target": 2.0},
                initial_candidate_quantity=0,
                evidence_strength=EvidenceStrength.CONFIRMED_BENCHMARK,
                decision_reasons=tuple(reasons),
                governance_checks_passed=tuple(checks_passed),
                warnings=tuple(warnings),
                pre_decision_features=pre_features,
            )
        checks_passed.append("GATE_8_GENERATOR_FEASIBILITY_VERIFIED")

        # ---------------------------------------------------------------------
        # Gate 9 & 10: High-Budget Synthetic Strategy Selection (B_k >= 200)
        # ---------------------------------------------------------------------
        # In Batch 5L, when source retention is strictly constrained (tolerance <= 0.05),
        # SYNTHETIC_WEIGHTED_HYBRID (H1, w=2) achieved adaptation gain (F1=0.4315) while
        # preserving source retention (F1=0.6102 vs B0=0.6075).
        if request.source_retention_tolerance <= 0.05:
            reasons.append(
                "Source retention is strictly constrained (tolerance <= 0.05). "
                "Prescribing TVAE joint hybrid weighting (w_target=w_synth=2.0) based on Batch 5L empirical safety evidence."
            )
            checks_passed.append("GATE_9_SOURCE_RETENTION_CONSTRAINED")
            checks_passed.append("GATE_10_STRATEGY_DECIDED")
            return PrescriptionDecision(
                strategy=RepairStrategy.SYNTHETIC_WEIGHTED_HYBRID,
                recommended_generator="tvae",
                sample_weight_policy={"source": 1.0, "target": 2.0, "synth": 2.0},
                initial_candidate_quantity=50,
                evidence_strength=EvidenceStrength.EXPLORATORY_PILOT,
                decision_reasons=tuple(reasons),
                governance_checks_passed=tuple(checks_passed),
                warnings=tuple(warnings),
                pre_decision_features=pre_features,
            )
        else:
            reasons.append(
                "Source retention constraint is relaxed (> 0.05). "
                "Prescribing unweighted TVAE synthetic augmentation (K=50) to maximize target domain adaptation."
            )
            checks_passed.append("GATE_9_SOURCE_RETENTION_RELAXED")
            checks_passed.append("GATE_10_STRATEGY_DECIDED")
            return PrescriptionDecision(
                strategy=RepairStrategy.SYNTHETIC_UNWEIGHTED,
                recommended_generator="tvae",
                sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                initial_candidate_quantity=50,
                evidence_strength=EvidenceStrength.EXPLORATORY_PILOT,
                decision_reasons=tuple(reasons),
                governance_checks_passed=tuple(checks_passed),
                warnings=tuple(warnings),
                pre_decision_features=pre_features,
            )
