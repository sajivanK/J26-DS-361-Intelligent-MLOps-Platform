"""Minimum effective repair quantity planner for Component 3.

Evaluates an ascending search grid of synthetic augmentation budgets (e.g. 25, 50, 100, 200)
to find the smallest acceptable quantity clearing structural, privacy, and utility gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from typing import Any
import pandas as pd

from component3.src.evaluation.retraining_evaluator import (
    RetrainingEvaluationResult,
    RetrainingEvaluator,
)
from component3.src.generators.generator_selector import GeneratorSelector
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.validation.validation_models import AcceptanceDecision
from component3.src.validation.validation_pipeline import (
    UnifiedValidationPipeline,
    ValidationPipelineReport,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantityPlanResult:
    """Outcome of minimum effective repair quantity search."""
    status: str  # "ACCEPTED", "NO_ACCEPTABLE_REPAIR", "SKIPPED"
    selected_quantity: int | None
    is_acceptable: bool
    reasons: tuple[str, ...]
    evaluated_quantities: tuple[int, ...]
    search_history: list[dict[str, Any]]
    accepted_synthetic_df: pd.DataFrame | None = field(default=None, repr=False)
    final_validation_report: ValidationPipelineReport | None = None
    final_retraining_result: RetrainingEvaluationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_quantity": self.selected_quantity,
            "is_acceptable": self.is_acceptable,
            "reasons": list(self.reasons),
            "evaluated_quantities": list(self.evaluated_quantities),
            "search_history": self.search_history,
            "final_validation_report": self.final_validation_report.to_dict() if self.final_validation_report else None,
            "final_retraining_result": self.final_retraining_result.to_dict() if self.final_retraining_result else None,
            "optimality_qualification": (
                "The selected quantity represents the smallest tested acceptable candidate budget "
                "within the evaluated discrete search grid. It is NOT claimed to be a globally optimal quantity."
            ),
        }


class RepairQuantityPlanner:
    """Searches a bounded candidate grid to find the minimum effective augmentation quantity."""

    def __init__(
        self,
        candidate_quantities: tuple[int, ...] = (25, 50, 100, 200),
        max_search_budget: int = 500,
    ) -> None:
        self.candidate_quantities = tuple(sorted(q for q in candidate_quantities if q <= max_search_budget))
        self.max_search_budget = max_search_budget

    def plan_quantity(
        self,
        generator_selector: GeneratorSelector,
        generator_name: str,
        adaptation_training_df: pd.DataFrame,
        base_model: Any,
        preprocessor: TabularDataPreprocessor,
        df_source_train: pd.DataFrame,
        df_adaptation: pd.DataFrame,
        df_dev_val: pd.DataFrame | None,
        df_source_eval: pd.DataFrame | None,
        validation_pipeline: UnifiedValidationPipeline,
        retraining_evaluator: RetrainingEvaluator,
        sample_weight_policy: dict[str, float] | None = None,
        target_column: str = "Churn",
        categorical_columns: list[str] | tuple[str, ...] | None = None,
        numerical_columns: list[str] | tuple[str, ...] | None = None,
        seed: int = 42,
        pre_generated_pool: pd.DataFrame | None = None,
        min_practical_utility_margin: float = 0.01,
        reference_authentic_f1: float | None = None,
    ) -> QuantityPlanResult:


        """Evaluate candidate quantities in ascending order and stop at the first acceptable budget."""
        search_history: list[dict[str, Any]] = []
        evaluated_quantities: list[int] = []
        failure_reasons: list[str] = []

        cat_cols = list(categorical_columns) if categorical_columns else []
        num_cols = list(numerical_columns) if numerical_columns else []

        for K in self.candidate_quantities:
            evaluated_quantities.append(K)
            logger.info("Evaluating candidate repair quantity K=%d with generator '%s'...", K, generator_name)

            # Step 1: Obtain K synthetic candidate records
            try:
                if pre_generated_pool is not None and len(pre_generated_pool) >= K:
                    synth_k = pre_generated_pool.iloc[:K].copy().reset_index(drop=True)
                    gen_prov = {"source": "pre_generated_pool", "rows": K}
                else:
                    synth_k, gen_prov = generator_selector.fit_and_sample(
                        generator_name=generator_name,
                        training_df=adaptation_training_df,
                        num_samples=K,
                        categorical_columns=cat_cols,
                        numerical_columns=num_cols,
                        seed=seed + K,
                    )
            except Exception as exc:
                reason = f"Candidate generation failed at K={K}: {exc}"
                logger.warning(reason)
                search_history.append({"quantity": K, "status": "GENERATION_FAILED", "error": str(exc)})
                failure_reasons.append(reason)
                continue

            # Step 2: Validate candidates through pipeline
            val_report = validation_pipeline.validate(
                synthetic_df=synth_k,
                reference_df=adaptation_training_df,
                target_column=target_column,
                categorical_columns=cat_cols,
                numerical_columns=num_cols,
            )

            if val_report.overall_decision == AcceptanceDecision.REJECT:
                reason = f"Candidate validation rejected at K={K}: {list(val_report.rejection_reasons)}"
                logger.info(reason)
                search_history.append({
                    "quantity": K,
                    "status": "VALIDATION_REJECTED",
                    "reasons": list(val_report.rejection_reasons),
                })
                failure_reasons.append(reason)
                continue

            # Step 3: Evaluate retraining utility and source retention
            retrain_res = retraining_evaluator.evaluate_candidate(
                base_model=base_model,
                preprocessor=preprocessor,
                df_source_train=df_source_train,
                df_adaptation=df_adaptation,
                synthetic_df=synth_k,
                df_dev_val=df_dev_val,
                df_source_eval=df_source_eval,
                sample_weight_policy=sample_weight_policy,
                target_column=target_column,
                reference_authentic_f1=reference_authentic_f1,
                min_utility_margin=min_practical_utility_margin,
            )


            if not retrain_res.is_acceptable:
                reason = f"Retraining evaluation rejected at K={K}: {list(retrain_res.rejection_reasons)}"
                logger.info(reason)
                search_history.append({
                    "quantity": K,
                    "status": "RETRAINING_REJECTED",
                    "reasons": list(retrain_res.rejection_reasons),
                    "delta_target_f1": retrain_res.delta_target_f1_vs_baseline,
                    "source_delta": retrain_res.source_retention_delta,
                })
                failure_reasons.append(reason)
                continue

            # SUCCESS: Smallest acceptable quantity found!
            logger.info("Found acceptable minimum repair quantity K=%d!", K)
            search_history.append({
                "quantity": K,
                "status": "ACCEPTED",
                "delta_target_f1": retrain_res.delta_target_f1_vs_baseline,
                "target_f1": retrain_res.target_positive_f1,
                "source_f1": retrain_res.source_positive_f1,
                "source_delta": retrain_res.source_retention_delta,
            })

            return QuantityPlanResult(
                status="ACCEPTED",
                selected_quantity=K,
                is_acceptable=True,
                reasons=(f"Smallest tested acceptable quantity K={K} passed all validation and retraining gates.",),
                evaluated_quantities=tuple(evaluated_quantities),
                search_history=search_history,
                accepted_synthetic_df=synth_k,
                final_validation_report=val_report,
                final_retraining_result=retrain_res,
            )

        # No quantity passed
        reasons_tuple = tuple(failure_reasons) or ("No evaluated candidate quantity cleared all acceptance gates.",)
        return QuantityPlanResult(
            status="NO_ACCEPTABLE_REPAIR",
            selected_quantity=None,
            is_acceptable=False,
            reasons=reasons_tuple,
            evaluated_quantities=tuple(evaluated_quantities),
            search_history=search_history,
            accepted_synthetic_df=None,
            final_validation_report=None,
            final_retraining_result=None,
        )
