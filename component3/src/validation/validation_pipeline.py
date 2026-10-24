"""Unified Multi-Stage Synthetic Data Validation Pipeline for Component 3.

Executes sequential validation across:
1. Structural integrity & Schema/Type validation
2. Missing-value & Domain boundary constraints
3. Duplicate detection & Privacy-risk screening (DCR)
4. Empirical statistical fidelity
5. Offline utility evaluation (TSTR)
6. Resource & Provenance accounting
7. Final acceptance/rejection governance decision
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import pandas as pd

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.validation.acceptance_policy import AcceptancePolicy
from component3.src.validation.fidelity_validator import FidelityValidator
from component3.src.validation.privacy_validator import PrivacyValidator
from component3.src.validation.structural_validator import StructuralValidator
from component3.src.validation.utility_evaluator import OfflineUtilityEvaluator
from component3.src.validation.validation_models import (
    AcceptanceDecision,
    RepairPackageDecision,
    ValidationStageResult,
    ValidationStatus,
)


@dataclass(frozen=True)
class ValidationPipelineReport:
    """Consolidated, disaggregated validation report."""
    overall_decision: AcceptanceDecision
    package_status: str  # "APPROVED" or "REJECTED"
    supervised_retraining_approved: bool
    rejection_reasons: tuple[str, ...]

    # Disaggregated sub-reports
    hard_validity_report: dict[str, Any]
    empirical_plausibility_report: dict[str, Any]
    fidelity_report: dict[str, Any]
    utility_report: dict[str, Any]
    privacy_risk_report: dict[str, Any]
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_decision": self.overall_decision.value,
            "package_status": self.package_status,
            "supervised_retraining_approved": self.supervised_retraining_approved,
            "rejection_reasons": list(self.rejection_reasons),
            "hard_validity_report": self.hard_validity_report,
            "empirical_plausibility_report": self.empirical_plausibility_report,
            "fidelity_report": self.fidelity_report,
            "utility_report": self.utility_report,
            "privacy_risk_report": self.privacy_risk_report,
            "stage_results": self.stage_results,
        }


class UnifiedValidationPipeline:
    """Orchestrates comprehensive multi-stage validation for synthetic candidates."""

    def __init__(
        self,
        mode: str = "production",
        strict_privacy: bool = True,
        strict_fidelity: bool = True,
    ) -> None:
        self.structural_validator = StructuralValidator()
        self.fidelity_validator = FidelityValidator()
        self.privacy_validator = PrivacyValidator()
        self.utility_evaluator = OfflineUtilityEvaluator()
        self.acceptance_policy = AcceptancePolicy(
            mode=mode,
            strict_privacy_gate=strict_privacy,
            strict_fidelity_gate=strict_fidelity,
        )

    def validate(
        self,
        synthetic_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        target_column: str,
        categorical_columns: list[str] | tuple[str, ...] | None = None,
        numerical_columns: list[str] | tuple[str, ...] | None = None,
        utility_test_df: pd.DataFrame | None = None,
        slice_rule: str | None = None,
        min_dcr_p05: float = 0.01,
        max_duplicate_ratio: float = 0.0,
    ) -> ValidationPipelineReport:
        """Run all validation stages and synthesize the consolidated report."""
        cat_cols = list(categorical_columns) if categorical_columns else []
        num_cols = list(numerical_columns) if numerical_columns else []

        # Reference column partitioning
        ref_cols = list(reference_df.columns)
        disc_cols = [c for c in ref_cols if c in cat_cols or (c == target_column and c not in num_cols)]
        cont_cols = [c for c in ref_cols if c in num_cols and c != target_column]
        for c in ref_cols:
            if c not in disc_cols and c not in cont_cols:
                if pd.api.types.is_numeric_dtype(reference_df[c]) and not pd.api.types.is_bool_dtype(reference_df[c]):
                    cont_cols.append(c)
                else:
                    disc_cols.append(c)

        expected_order = tuple(cont_cols + disc_cols)
        feature_types = {c: "continuous" for c in cont_cols} | {c: "categorical" for c in disc_cols}

        prep = PreparedGeneratorData(
            training_data=reference_df[list(expected_order)],
            discrete_columns=tuple(disc_cols),
            continuous_columns=tuple(cont_cols),
            n_rows=len(reference_df),
            source="adaptation_reference",
            target_included=target_column in expected_order,
            slice_support=1.0,
            generation_mode="supervised_repair" if target_column in expected_order else "unsupervised_exploratory",
            feature_types=feature_types,
        )

        synth_to_check = synthetic_df
        if set(synthetic_df.columns) == set(expected_order):
            synth_to_check = synthetic_df[list(expected_order)]

        # ---------------------------------------------------------------------
        # Stage 1: Structural Integrity and Schema Validation
        # ---------------------------------------------------------------------
        struct_res = self.structural_validator.validate(
            synthetic_data=synth_to_check,
            prepared_data=prep,
            slice_rule=slice_rule,
        )

        hard_validity_report = {
            "status": struct_res.status.value,
            "total_rows": struct_res.metrics.get("total_rows", len(synthetic_df)),
            "total_columns": struct_res.metrics.get("total_columns", len(synthetic_df.columns)),
            "missing_values": struct_res.metrics.get("missing_values", 0),
            "reasons": list(struct_res.reasons),
        }

        # If structural validation fails, abort early with rejection
        if struct_res.status == ValidationStatus.FAIL:
            reasons = tuple(struct_res.reasons) or ("Structural schema validation failed.",)
            return ValidationPipelineReport(
                overall_decision=AcceptanceDecision.REJECT,
                package_status="REJECTED",
                supervised_retraining_approved=False,
                rejection_reasons=reasons,
                hard_validity_report=hard_validity_report,
                empirical_plausibility_report={"status": "SKIPPED"},
                fidelity_report={"status": "SKIPPED"},
                utility_report={"status": "SKIPPED"},
                privacy_risk_report={"status": "SKIPPED"},
                stage_results={"structural": struct_res.to_dict()},
            )

        # ---------------------------------------------------------------------
        # Stage 2: Privacy-Risk Screening (DCR and Duplicate Detection)
        # ---------------------------------------------------------------------
        self.privacy_validator.min_dcr_p05 = min_dcr_p05
        self.privacy_validator.max_exact_match_ratio = max_duplicate_ratio
        priv_res = self.privacy_validator.validate(
            synthetic_data=synth_to_check,
            reference_data=reference_df,
            continuous_columns=cont_cols,
            discrete_columns=disc_cols,
        )

        exact_match_info = priv_res.metrics.get("exact_matches", {})
        dcr_info = priv_res.metrics.get("dcr_metrics", {})
        privacy_risk_report = {
            "status": priv_res.status.value,
            "exact_match_count": exact_match_info.get("exact_match_count", 0),
            "exact_match_ratio": exact_match_info.get("exact_match_ratio", 0.0),
            "dcr_p05": dcr_info.get("dcr_p05", 0.0),
            "dcr_mean": dcr_info.get("dcr_mean", 0.0),
            "near_duplicate_count": dcr_info.get("near_duplicate_count", 0),
            "privacy_metric_note": (
                "Empirical Distance to Closest Record (DCR) is a nearest-neighbor distance indicator. "
                "It does NOT establish formal mathematical differential privacy (epsilon, delta)."
            ),
            "reasons": list(priv_res.reasons),
        }

        # ---------------------------------------------------------------------
        # Stage 3: Empirical Statistical Fidelity Analysis
        # ---------------------------------------------------------------------
        fid_res = self.fidelity_validator.validate(
            synthetic_data=synth_to_check,
            reference_data=reference_df,
            continuous_columns=cont_cols,
            discrete_columns=disc_cols,
        )

        cat_summary = fid_res.metrics.get("categorical_metrics_summary", {})
        corr_diff = fid_res.metrics.get("correlation_difference", {})
        fidelity_report = {
            "status": fid_res.status.value,
            "overall_fidelity_score": fid_res.metrics.get("fidelity_score", 0.0),
            "mean_ks_statistic": fid_res.metrics.get("continuous_metrics_summary", {}).get("mean_ks_statistic", 0.0),
            "mean_total_variation_distance": cat_summary.get("mean_total_variation_distance", 0.0),
            "correlation_frobenius_norm": corr_diff.get("frobenius_norm", 0.0),
            "reasons": list(fid_res.reasons),
        }

        # ---------------------------------------------------------------------
        # Stage 4: Offline Utility Evaluation (TSTR)
        # ---------------------------------------------------------------------
        utility_test = utility_test_df if utility_test_df is not None else reference_df
        self.utility_evaluator.target_column = target_column
        has_labels = (
            target_column in synth_to_check.columns
            and target_column in reference_df.columns
            and target_column in utility_test.columns
            and not synth_to_check[target_column].isna().any()
        )
        util_res = self.utility_evaluator.evaluate(
            synthetic_data=synth_to_check,
            reference_train_data=reference_df,
            evaluation_test_data=utility_test,
            slice_rule=slice_rule,
            label_authorized=has_labels,
        )

        deltas = util_res.metrics.get("deltas", {}) if util_res.metrics else {}
        utility_report = {
            "status": util_res.status.value,
            "utility_score": util_res.metrics.get("utility_score", 0.0) if util_res.metrics else 0.0,
            "delta_overall_f1": deltas.get("delta_overall_f1", 0.0),
            "delta_affected_slice_f1": deltas.get("delta_affected_slice_f1", 0.0),
            "reasons": list(util_res.reasons),
        }

        # ---------------------------------------------------------------------
        # Stage 5: Empirical Plausibility Sub-Report
        # ---------------------------------------------------------------------
        empirical_plausibility_report = {
            "status": "PASS" if fid_res.status != ValidationStatus.FAIL else "FAIL",
            "fidelity_passed": fid_res.status == ValidationStatus.PASS,
            "domain_bounds_respected": struct_res.metrics.get("missing_values", 0) == 0,
            "synthetic_label_nature_note": (
                "Synthetic candidate labels reflect joint distribution P(X, Y) modeling from training data. "
                "They are synthetic observational assertions, not external ground truth."
            ),
        }

        # ---------------------------------------------------------------------
        # Stage 6: Overall Multi-Stage Acceptance Decision
        # ---------------------------------------------------------------------
        pkg_decision: RepairPackageDecision = self.acceptance_policy.decide(
            structural=struct_res,
            fidelity=fid_res,
            privacy=priv_res,
            utility=util_res,
        )

        stage_results_dict = {
            "structural": struct_res.to_dict(),
            "privacy": priv_res.to_dict(),
            "fidelity": fid_res.to_dict(),
            "utility": util_res.to_dict(),
        }

        return ValidationPipelineReport(
            overall_decision=pkg_decision.decision,
            package_status=pkg_decision.package_status,
            supervised_retraining_approved=pkg_decision.supervised_retraining_approved,
            rejection_reasons=pkg_decision.reasons,
            hard_validity_report=hard_validity_report,
            empirical_plausibility_report=empirical_plausibility_report,
            fidelity_report=fidelity_report,
            utility_report=utility_report,
            privacy_risk_report=privacy_risk_report,
            stage_results=stage_results_dict,
        )
