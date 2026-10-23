"""End-to-end repair execution orchestrator for Component 3.

Coordinates:
RepairRequest -> Request Validation -> Prescription -> Generator Selection ->
Candidate Generation -> Multi-Stage Validation -> Quantity Search ->
Retraining Evaluation -> Acceptance/Rejection -> Versioned RepairPackage.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd
import psutil

from component3.src.contracts.repair_package import (
    RepairPackage,
    RepairPackageStatus,
    compute_file_sha256,
)
from component3.src.contracts.repair_request import (
    ExecutionMode,
    InvalidRepairRequestError,
    RepairRequest,
)
from component3.src.contracts.targeted_repair import (
    TargetedRepairStatus,
    validate_diagnostic_slice,
)
from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.evaluation.retraining_evaluator import (
    RetrainingEvaluationResult,
    RetrainingEvaluator,
)
from component3.src.generators.generator_selector import (
    GeneratorExecutionError,
    GeneratorSelector,
)
from component3.src.orchestration.orchestrator_models import (
    ExecutionState,
    OrchestratorExecutionLog,
)
from component3.src.planning.quantity_planner import (
    QuantityPlanResult,
    RepairQuantityPlanner,
)
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.prescription.prescription_engine import RepairPrescriptionEngine
from component3.src.prescription.prescription_models import (
    DecisionMode,
    PrescriptionDecision,
    RepairStrategy,
)
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)
from component3.src.validation.validation_models import AcceptanceDecision
from component3.src.validation.validation_pipeline import (
    UnifiedValidationPipeline,
    ValidationPipelineReport,
)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


class RepairOrchestrator:
    """Production orchestrator driving the entire Component 3 repair workflow."""

    def __init__(
        self,
        prescription_engine: RepairPrescriptionEngine | None = None,
        generator_selector: GeneratorSelector | None = None,
        validation_pipeline: UnifiedValidationPipeline | None = None,
        retraining_evaluator: RetrainingEvaluator | None = None,
        quantity_planner: RepairQuantityPlanner | None = None,
    ) -> None:
        self.prescription_engine = prescription_engine or RepairPrescriptionEngine()
        self.generator_selector = generator_selector or GeneratorSelector()
        self.validation_pipeline = validation_pipeline or UnifiedValidationPipeline()
        self.retraining_evaluator = retraining_evaluator or RetrainingEvaluator()
        self.quantity_planner = quantity_planner or RepairQuantityPlanner()

    def execute_repair(
        self,
        request: RepairRequest,
        output_directory: Path | str,
        df_source_train: pd.DataFrame | None = None,
        df_adaptation: pd.DataFrame | None = None,
        df_dev_val: pd.DataFrame | None = None,
        df_source_eval: pd.DataFrame | None = None,
        pre_generated_candidates_df: pd.DataFrame | None = None,
        decision_mode: DecisionMode = DecisionMode.RULE_BASED,
    ) -> tuple[RepairPackage, OrchestratorExecutionLog]:
        """Execute full repair pipeline for an incoming RepairRequest."""
        run_id = f"RUN-{uuid.uuid4().hex[:10].upper()}"
        out_dir = Path(output_directory)
        resolved_out = out_dir.resolve()
        comp3_root = (REPO_ROOT / "component3").resolve()
        shared_root = (REPO_ROOT / "shared").resolve()
        from component3.src.utils.artifact_storage import get_configured_storage_root
        configured_root = get_configured_storage_root()
        import tempfile
        temp_dir = Path(tempfile.gettempdir()).resolve()

        is_allowed = (
            resolved_out.is_relative_to(comp3_root)
            or resolved_out.is_relative_to(shared_root)
            or resolved_out.is_relative_to(configured_root)
            or resolved_out.is_relative_to(temp_dir)
        )
        if not is_allowed:
            raise InvalidRepairRequestError(f"Output directory '{resolved_out}' is outside allowed storage roots.")
        out_dir.mkdir(parents=True, exist_ok=True)

        exec_log = OrchestratorExecutionLog(
            run_id=run_id,
            request_id=request.request_id,
            current_state=ExecutionState.CREATED,
        )

        t_start = time.perf_counter()
        rejection_reasons: list[str] = []

        if request.max_runtime_seconds < 1.0:
            err_msg = f"Infeasible runtime budget ({request.max_runtime_seconds}s). Minimum required is 1.0s."
            exec_log.transition_to(ExecutionState.FAILED, err_msg)
            pkg = RepairPackage(
                request_id=request.request_id,
                run_id=run_id,
                dataset_id=request.dataset_id,
                model_id=request.model_id,
                execution_status="FAILED",
                package_status=RepairPackageStatus.FAILED,
                rejection_reasons=(err_msg,),
            )
            pkg_path = pkg.save_package(out_dir)
            exec_log.package_path = str(pkg_path)
            return pkg, exec_log


        # ---------------------------------------------------------------------
        # Stage 1: VALIDATING
        # ---------------------------------------------------------------------
        exec_log.transition_to(ExecutionState.VALIDATING, "Validating input request and data references.")
        t0_val = time.perf_counter()

        reg = get_dataset_registry()
        meta = reg.get(request.dataset_id)
        feat_cols = meta.feature_columns
        cat_cols = meta.categorical_features
        num_cols = meta.numerical_features
        target_col = meta.target_column

        # Load models and preprocessor
        src_model_path = Path(request.source_model_path)
        if not src_model_path.is_absolute():
            src_model_path = REPO_ROOT / src_model_path

        src_prep_path = Path(request.source_preprocessor_path)
        if not src_prep_path.is_absolute():
            src_prep_path = REPO_ROOT / src_prep_path

        if not src_model_path.is_file() or not src_prep_path.is_file():
            err_msg = f"Source model or preprocessor file missing: {src_model_path}, {src_prep_path}"
            exec_log.transition_to(ExecutionState.FAILED, err_msg)
            exec_log.error_message = err_msg
            pkg = RepairPackage(
                request_id=request.request_id,
                run_id=run_id,
                dataset_id=request.dataset_id,
                model_id=request.model_id,
                execution_status="FAILED",
                package_status=RepairPackageStatus.FAILED,
                rejection_reasons=(err_msg,),
            )
            pkg_path = pkg.save_package(out_dir)
            exec_log.package_path = str(pkg_path)
            return pkg, exec_log

        source_model = joblib.load(src_model_path)
        source_preprocessor: TabularDataPreprocessor = joblib.load(src_prep_path)

        # Reconstruct partitions if not passed explicitly
        if df_source_train is None or df_dev_val is None or df_source_eval is None:
            raw_df = reg.load_dataset(request.dataset_id)
            canon_split_p = REPO_ROOT / "component3" / "results" / "split_manifests" / f"{request.dataset_id}_split_manifest.json"
            if canon_split_p.is_file():
                import json
                with open(canon_split_p, "r", encoding="utf-8") as f:
                    canon_manifest = json.load(f)
                canon_train = raw_df.iloc[canon_manifest["train_indices"]].copy().reset_index(drop=True)
            else:
                canon_train = raw_df

            splits, _ = ExperimentalPartitionBuilder.build_splits(
                canonical_train_df=canon_train,
                metadata=meta,
                adaptation_budgets=(50, 100, 200),
                seed=42,
            )
            df_source_train = splits.source_train.copy().reset_index(drop=True)
            df_source_eval = splits.source_eval.copy().reset_index(drop=True)

            # Isolate clean dev_val from adaptation pool
            b_budget = request.adaptation_budget_size if request.adaptation_budget_size in (50, 100, 200) else 200
            b_df = splits.nested_adapt_budgets[b_budget].copy().reset_index(drop=True)
            b_cust_ids = set(b_df["customerID"])
            unused = splits.target_adapt_pool[~splits.target_adapt_pool["customerID"].isin(b_cust_ids)].copy().reset_index(drop=True)

            b_sigs = set(tuple(str(v) for v in r) for r in b_df[feat_cols].itertuples(index=False))
            unused_sigs = [tuple(str(v) for v in r) for r in unused[feat_cols].itertuples(index=False)]
            df_dev_val = unused.drop(index=[i for i, s in enumerate(unused_sigs) if s in b_sigs]).copy().reset_index(drop=True)
            df_adaptation = b_df

            # Apply drift ground truth to adaptation & dev_val
            if request.drift_type == request.drift_type.CONCEPT_DRIFT:
                flips_adapt = df_adaptation["tenure"] > 29.0
                df_adaptation.loc[flips_adapt, target_col] = df_adaptation.loc[flips_adapt, target_col].map({"No": "Yes", "Yes": "No"})
                flips_dev = df_dev_val["tenure"] > 29.0
                df_dev_val.loc[flips_dev, target_col] = df_dev_val.loc[flips_dev, target_col].map({"No": "Yes", "Yes": "No"})

            # Clean dropped/id columns and coerce numeric fields across all working partitions
            drop_cols = [c for c in meta.drop_columns + meta.id_columns if c in df_adaptation.columns]
            if drop_cols:
                df_source_train = df_source_train.drop(columns=drop_cols, errors="ignore")
                df_source_eval = df_source_eval.drop(columns=drop_cols, errors="ignore")
                df_adaptation = df_adaptation.drop(columns=drop_cols, errors="ignore")
                df_dev_val = df_dev_val.drop(columns=drop_cols, errors="ignore")

            for df_p in (df_source_train, df_source_eval, df_adaptation, df_dev_val):
                if "TotalCharges" in df_p.columns:
                    df_p["TotalCharges"] = pd.to_numeric(df_p["TotalCharges"], errors="coerce").fillna(0.0)

        exec_log.stage_durations["validation_seconds"] = round(time.perf_counter() - t0_val, 3)

        # ---------------------------------------------------------------------
        # Stage 2: PRESCRIBING
        # ---------------------------------------------------------------------
        exec_log.transition_to(ExecutionState.PRESCRIBING, "Evaluating strategy prescription engine.")
        t0_pres = time.perf_counter()

        import inspect
        prescribe_sig = inspect.signature(self.prescription_engine.prescribe)
        prescribe_kwargs: dict[str, Any] = {"request": request, "adaptation_df": df_adaptation}
        if "decision_mode" in prescribe_sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in prescribe_sig.parameters.values()
        ):
            prescribe_kwargs["decision_mode"] = decision_mode

        prescription: PrescriptionDecision = self.prescription_engine.prescribe(**prescribe_kwargs)
        exec_log.stage_durations["prescription_seconds"] = round(time.perf_counter() - t0_pres, 3)

        chosen_strategy = prescription.strategy
        recommended_gen = prescription.recommended_generator

        # Handle NO_REPAIR / ABSTAIN early
        if chosen_strategy == RepairStrategy.NO_REPAIR:
            exec_log.transition_to(ExecutionState.ACCEPTED, "Prescription engine recommended NO_REPAIR.")
            pkg = RepairPackage(
                request_id=request.request_id,
                run_id=run_id,
                dataset_id=request.dataset_id,
                model_id=request.model_id,
                drift_summary={
                    "drift_detected": request.drift_detected,
                    "drift_type": request.drift_type.value,
                    "drift_severity": request.drift_severity.value,
                    "severity_score": request.drift_severity_score,
                },
                diagnostic_summary=request.diagnostic_slice.to_dict() if request.diagnostic_slice else {},
                chosen_strategy=chosen_strategy.value,
                generator_info={"generator": "NONE"},
                quantity_selected=0,
                candidate_model_path=str(src_model_path),
                candidate_model_sha256=compute_file_sha256(src_model_path),
                execution_status="COMPLETED",
                package_status=RepairPackageStatus.ABSTAINED_NO_REPAIR,
                rejection_reasons=prescription.decision_reasons,
                requires_human_approval=False,
                downstream_handoff_metadata={
                    "prescription_reasons": list(prescription.decision_reasons),
                    "decision_mode": prescription.decision_mode.value,
                    "predicted_delta_f1": prescription.predicted_delta_f1,
                },
            )
            pkg_path = pkg.save_package(out_dir)
            exec_log.package_path = str(pkg_path)
            return pkg, exec_log

        # Handle DRY_RUN mode
        if request.execution_mode == ExecutionMode.DRY_RUN:
            exec_log.transition_to(ExecutionState.ACCEPTED, "DRY_RUN completed successfully.")
            pkg = RepairPackage(
                request_id=request.request_id,
                run_id=run_id,
                dataset_id=request.dataset_id,
                model_id=request.model_id,
                chosen_strategy=chosen_strategy.value,
                generator_info={"generator": recommended_gen or "NONE"},
                execution_status="DRY_RUN_COMPLETED",
                package_status=RepairPackageStatus.APPROVED_FOR_REVIEW,
                downstream_handoff_metadata={"pre_decision_features": prescription.pre_decision_features},
            )
            pkg_path = pkg.save_package(out_dir)
            exec_log.package_path = str(pkg_path)
            return pkg, exec_log

        # ---------------------------------------------------------------------
        # Stage 3: GENERATING (Synthetic Strategies) or Direct EVALUATING (Authentic)
        # ---------------------------------------------------------------------
        synth_candidates_df = None
        quantity_plan_res: QuantityPlanResult | None = None
        val_report: ValidationPipelineReport | None = None
        retrain_result: RetrainingEvaluationResult | None = None

        is_synthetic_strategy = chosen_strategy in (
            RepairStrategy.SYNTHETIC_UNWEIGHTED,
            RepairStrategy.SYNTHETIC_WEIGHTED_HYBRID,
            RepairStrategy.TARGETED_SYNTHETIC_REPAIR,
        )

        if is_synthetic_strategy:
            exec_log.transition_to(ExecutionState.GENERATING, f"Generating synthetic records via {recommended_gen}.")
            t0_gen = time.perf_counter()

            # Handle targeted synthetic repair conditioning and non-oracle gating
            adaptation_gen_df = df_adaptation
            if chosen_strategy == RepairStrategy.TARGETED_SYNTHETIC_REPAIR and request.diagnostic_slice is not None:
                t_status, t_msg, slice_idx = validate_diagnostic_slice(request.diagnostic_slice, current_df=df_adaptation)
                if t_status != TargetedRepairStatus.ELIGIBLE:
                    err_msg = f"Targeted repair blocked: {t_msg}"
                    rejection_reasons.append(err_msg)
                    exec_log.transition_to(ExecutionState.FAILED, err_msg)
                    pkg = RepairPackage(
                        request_id=request.request_id,
                        run_id=run_id,
                        dataset_id=request.dataset_id,
                        model_id=request.model_id,
                        chosen_strategy=chosen_strategy.value,
                        package_status=RepairPackageStatus.BLOCKED,
                        rejection_reasons=(err_msg,),
                        requires_human_approval=True,
                    )
                    pkg_path = pkg.save_package(out_dir)
                    exec_log.package_path = str(pkg_path)
                    return pkg, exec_log

                if slice_idx is not None and int(slice_idx.sum()) >= 30:
                    adaptation_gen_df = df_adaptation[slice_idx.values].copy().reset_index(drop=True)


            # Configure evaluator with request's specific source retention tolerance
            active_evaluator = RetrainingEvaluator(
                source_retention_tolerance=request.source_retention_tolerance,
                min_practical_utility_margin=self.retraining_evaluator.min_practical_utility_margin,
            )

            # Compute matched authentic control reference if available
            auth_ctrl_res = active_evaluator.evaluate_candidate(
                base_model=source_model,
                preprocessor=source_preprocessor,
                df_source_train=df_source_train,
                df_adaptation=df_adaptation,
                synthetic_df=None,
                df_dev_val=df_dev_val,
                df_source_eval=df_source_eval,
                sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                target_column=target_col,
                min_utility_margin=-1.0,  # Evaluate without gating
            )
            ref_auth_f1 = auth_ctrl_res.target_positive_f1

            # Execute quantity planning search
            quantity_plan_res = self.quantity_planner.plan_quantity(
                generator_selector=self.generator_selector,
                generator_name=recommended_gen or "tvae",
                adaptation_training_df=adaptation_gen_df,
                base_model=source_model,
                preprocessor=source_preprocessor,
                df_source_train=df_source_train,
                df_adaptation=df_adaptation,
                df_dev_val=df_dev_val,
                df_source_eval=df_source_eval,
                validation_pipeline=self.validation_pipeline,
                retraining_evaluator=active_evaluator,
                sample_weight_policy=prescription.sample_weight_policy,
                target_column=target_col,
                categorical_columns=cat_cols,
                numerical_columns=num_cols,
                pre_generated_pool=pre_generated_candidates_df,
                reference_authentic_f1=ref_auth_f1,
            )
            exec_log.stage_durations["generation_planning_seconds"] = round(time.perf_counter() - t0_gen, 3)

            synth_candidates_df = quantity_plan_res.accepted_synthetic_df
            val_report = quantity_plan_res.final_validation_report
            retrain_result = quantity_plan_res.final_retraining_result

        else:
            # Authentic adaptation strategies
            exec_log.transition_to(ExecutionState.EVALUATING, f"Evaluating authentic adaptation strategy {chosen_strategy.value}.")
            t0_eval = time.perf_counter()

            # For authentic oversampling, perform class-balanced Random Over-Sampling
            df_adapt_retrain = df_adaptation
            if chosen_strategy == RepairStrategy.AUTHENTIC_OVERSAMPLING:
                class_counts = df_adaptation[target_col].value_counts()
                max_count = class_counts.max()
                resampled_dfs = []
                rng = np.random.default_rng(42)
                for cls_val, count in class_counts.items():
                    cls_df = df_adaptation[df_adaptation[target_col] == cls_val]
                    if count < max_count:
                        n_needed = max_count - count
                        sampled_idx = rng.choice(len(cls_df), size=n_needed, replace=True)
                        resampled_dfs.append(cls_df.iloc[sampled_idx])
                if resampled_dfs:
                    df_adapt_retrain = pd.concat([df_adaptation] + resampled_dfs, ignore_index=True)
                else:
                    dup_indices = rng.choice(len(df_adaptation), size=min(50, len(df_adaptation)), replace=True)
                    df_adapt_retrain = pd.concat([df_adaptation, df_adaptation.iloc[dup_indices]], ignore_index=True)

            active_evaluator = RetrainingEvaluator(
                source_retention_tolerance=request.source_retention_tolerance,
                min_practical_utility_margin=self.retraining_evaluator.min_practical_utility_margin,
            )
            retrain_result = active_evaluator.evaluate_candidate(
                base_model=source_model,
                preprocessor=source_preprocessor,
                df_source_train=df_source_train,
                df_adaptation=df_adapt_retrain,
                synthetic_df=None,
                df_dev_val=df_dev_val,
                df_source_eval=df_source_eval,
                sample_weight_policy=prescription.sample_weight_policy,
                target_column=target_col,
            )
            exec_log.stage_durations["evaluation_seconds"] = round(time.perf_counter() - t0_eval, 3)


        # ---------------------------------------------------------------------
        # Stage 4: Final Acceptance Decision & Package Emission
        # ---------------------------------------------------------------------
        exec_log.transition_to(ExecutionState.EVALUATING, "Finalizing acceptance decision and assembling package.")

        is_accepted = False
        package_status = RepairPackageStatus.REJECTED

        if is_synthetic_strategy and quantity_plan_res is not None:
            if quantity_plan_res.is_acceptable and retrain_result is not None and retrain_result.is_acceptable:
                is_accepted = True
                package_status = RepairPackageStatus.APPROVED_FOR_REVIEW
                exec_log.transition_to(ExecutionState.ACCEPTED, "Candidate repair passed all validation and utility gates.")
            else:
                rejection_reasons.extend(quantity_plan_res.reasons)
                if retrain_result:
                    rejection_reasons.extend(retrain_result.rejection_reasons)
                exec_log.transition_to(ExecutionState.REJECTED, "Candidate repair failed validation or retraining gates.")
        else:
            if retrain_result is not None and retrain_result.is_acceptable:
                is_accepted = True
                package_status = RepairPackageStatus.APPROVED_FOR_REVIEW
                exec_log.transition_to(ExecutionState.ACCEPTED, "Authentic adaptation candidate approved for review.")
            else:
                if retrain_result:
                    rejection_reasons.extend(retrain_result.rejection_reasons)
                exec_log.transition_to(ExecutionState.REJECTED, "Authentic adaptation candidate failed utility gates.")

        # Build final RepairPackage
        candidate_model_obj = retrain_result.candidate_model if retrain_result else None
        selected_qty = quantity_plan_res.selected_quantity if (quantity_plan_res and quantity_plan_res.selected_quantity) else 0

        pkg_id = f"PKG-{uuid.uuid4().hex[:12].upper()}"
        dataset_path_str = None
        dataset_hash = None
        if synth_candidates_df is not None and len(synth_candidates_df) > 0:
            csv_path = out_dir / f"{pkg_id.lower()}_candidates_k{len(synth_candidates_df)}.csv"
            tmp_csv = csv_path.with_suffix(".tmp")
            synth_candidates_df.to_csv(tmp_csv, index=False)
            tmp_csv.replace(csv_path)
            dataset_path_str = str(csv_path)
            dataset_hash = compute_file_sha256(csv_path)

        model_path_str = None
        model_hash = None
        if candidate_model_obj is not None:
            m_path = out_dir / f"{pkg_id.lower()}_candidate_model.joblib"
            tmp_m = m_path.with_suffix(".tmp")
            joblib.dump(candidate_model_obj, tmp_m)
            tmp_m.replace(m_path)
            model_path_str = str(m_path)
            model_hash = compute_file_sha256(m_path)

        pkg = RepairPackage(
            package_id=pkg_id,
            request_id=request.request_id,
            run_id=run_id,
            dataset_id=request.dataset_id,
            model_id=request.model_id,
            drift_summary={
                "drift_detected": request.drift_detected,
                "drift_type": request.drift_type.value,
                "drift_severity": request.drift_severity.value,
                "severity_score": request.drift_severity_score,
            },
            diagnostic_summary=request.diagnostic_slice.to_dict() if request.diagnostic_slice else {},
            chosen_strategy=chosen_strategy.value,
            generator_info={
                "generator": recommended_gen or "NONE",
                "sample_weight_policy": prescription.sample_weight_policy,
            },
            quantity_selected=selected_qty,
            candidate_dataset_path=dataset_path_str,
            candidate_dataset_sha256=dataset_hash,
            candidate_model_path=model_path_str,
            candidate_model_sha256=model_hash,
            validation_report=val_report.to_dict() if val_report else {},
            fidelity_indicators=val_report.fidelity_report if val_report else {},
            privacy_risk_indicators=val_report.privacy_risk_report if val_report else {},
            utility_metrics=retrain_result.to_dict() if retrain_result else {},
            source_retention_metrics={
                "source_f1": retrain_result.source_positive_f1 if retrain_result else None,
                "source_delta": retrain_result.source_retention_delta if retrain_result else None,
                "source_retention_delta": retrain_result.source_retention_delta if retrain_result else None,
                "source_retention_preserved": retrain_result.source_retention_preserved if retrain_result else False,
                "evidence_status": "EVALUATED" if (retrain_result and retrain_result.source_positive_f1 is not None) else "MISSING_EVIDENCE",
            },
            execution_status="COMPLETED",
            package_status=package_status,
            rejection_reasons=tuple(rejection_reasons),
            requires_human_approval=True,
            downstream_handoff_metadata={
                "pre_decision_features": prescription.pre_decision_features,
                "governance_checks_passed": list(prescription.governance_checks_passed),
                "search_history": quantity_plan_res.search_history if quantity_plan_res else [],
                "decision_mode": prescription.decision_mode.value,
                "predicted_delta_f1": prescription.predicted_delta_f1,
                "evidence_strength": prescription.evidence_strength.value,
            },
        )


        manifest_path = pkg.save_package(
            output_directory=out_dir,
            synthetic_df=synth_candidates_df,
            candidate_model=candidate_model_obj,
        )
        exec_log.package_path = str(manifest_path)
        exec_log.stage_durations["total_pipeline_seconds"] = round(time.perf_counter() - t_start, 3)

        return pkg, exec_log
