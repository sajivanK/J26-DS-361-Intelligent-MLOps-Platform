"""Public Python API for Component 3: Adaptive Intelligent Synthetic Data Repair Engine.

Provides programmatic integration entry points for the platform layer, pipelines,
and automated execution runners.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPONENT3_ROOT = REPO_ROOT / "component3"

from component3.src.adapters.c1_drift_adapter import C1DriftAdapter
from component3.src.adapters.c2_diagnosis_adapter import C2DiagnosisAdapter
from component3.src.adapters.c4_retraining_adapter import C4RetrainingAdapter
from component3.src.contracts.external_contracts import (
    ContractValidationError,
    DependencyResolutionError,
    validate_c4_flat_contract,
    validate_three_section_contract,
)
from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
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
)
from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.generators.plugin_manager import GeneratorPluginManager
from component3.src.models.model_registry import get_model_registry
from component3.src.orchestration.orchestrator_models import OrchestratorExecutionLog
from component3.src.orchestration.repair_orchestrator import RepairOrchestrator
from component3.src.prescription.knowledge_base import get_repair_knowledge_base
from component3.src.prescription.prescription_engine import RepairPrescriptionEngine
from component3.src.prescription.prescription_models import (
    DecisionMode,
    PrescriptionDecision,
)

logger = logging.getLogger(__name__)


def _parse_request(raw_request: RepairRequest | dict[str, Any] | str | Path) -> RepairRequest:
    """Helper to deserialize various inputs into a validated RepairRequest.
    
    Supports both:
    1. Modern 3-section contract: {c1_evidence, c2_diagnosis, data_context}
    2. Legacy flat RepairRequest dictionary / instance
    """
    if isinstance(raw_request, RepairRequest):
        return raw_request

    if isinstance(raw_request, (str, Path)):
        p = Path(raw_request)
        if not p.is_file():
            raise FileNotFoundError(f"RepairRequest JSON file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(raw_request, dict):
        data = dict(raw_request)
    else:
        raise TypeError(f"Unsupported request type: {type(raw_request)}")

    # -------------------------------------------------------------------------
    # Branch A: Agreed Three-Section Multi-Component Contract
    # -------------------------------------------------------------------------
    if "c1_evidence" in data or "c2_diagnosis" in data or "data_context" in data:
        ok, errs, container = validate_three_section_contract(data)
        if not ok or container is None:
            raise ContractValidationError(f"Invalid external contract payload: {errs}")

        dc = container.data_context
        c1 = container.c1_evidence
        c2 = container.c2_diagnosis

        # Dependency resolution - strict non-substitution checks
        ref_p = Path(dc.reference_dataset_uri)
        if not ref_p.is_absolute():
            ref_p = REPO_ROOT / ref_p
        if not ref_p.is_file():
            raise DependencyResolutionError(
                f"Reference dataset not found at URI '{dc.reference_dataset_uri}' (resolved: '{ref_p}'). "
                "Component 3 will not silently substitute another dataset."
            )

        cur_p = Path(dc.current_dataset_uri)
        if not cur_p.is_absolute():
            cur_p = REPO_ROOT / cur_p
        if not cur_p.is_file():
            raise DependencyResolutionError(
                f"Current drifted dataset not found at URI '{dc.current_dataset_uri}' (resolved: '{cur_p}'). "
                "Component 3 will not silently substitute another dataset."
            )

        # Model resolution - check ModelRegistry or physical path
        model_reg = get_model_registry()
        if model_reg.has(dc.model_id):
            model_entry = model_reg.get(dc.model_id)
            src_model_p = model_entry.model_path
            src_prep_p = model_entry.preprocessor_path
        else:
            candidate_p = Path(dc.model_id)
            if not candidate_p.is_absolute():
                candidate_p = REPO_ROOT / candidate_p
            if candidate_p.is_file():
                src_model_p = candidate_p
                src_prep_p = dc.preprocessing_uri or candidate_p
            else:
                raise DependencyResolutionError(
                    f"Model '{dc.model_id}' is neither registered in ModelRegistry nor found on disk. "
                    "Component 3 will not silently substitute another model."
                )

        # Adapt C2 diagnosis
        c2_adapter = C2DiagnosisAdapter()
        diag_slice = c2_adapter.adapt({
            "affected_slice_rule": c2.affected_slice_rule,
            "root_cause_feature": c2.root_cause_feature,
            "root_cause_category": c2.root_cause_category,
            "affected_features_ranked": c2.affected_features_ranked,
            "affected_value_ranges": c2.affected_value_ranges,
            "slice_support": c2.slice_support,
            "diagnosis_stability": c2.diagnosis_stability,
            "drift_impact_score": c2.drift_impact_score,
            "temporal_window": c2.temporal_window,
        })

        # Adapt C1 drift
        c1_adapter = C1DriftAdapter(default_dataset_id=dc.dataset_id)
        req = c1_adapter.adapt(
            payload={
                "predicted_accuracy_drop": c1.predicted_accuracy_drop,
                "confidence_interval": c1.confidence_interval,
                "drift_type": c1.drift_type,
                "drift_severity": c1.drift_severity,
                "affected_features": c1.affected_features,
            },
            dataset_id=dc.dataset_id,
            adaptation_budget_size=dc.adaptation_budget_size,
            has_authentic_adaptation_labels=dc.has_authentic_labels,
            source_model_path=src_model_p,
            source_preprocessor_path=src_prep_p,
        )

        import dataclasses
        req = dataclasses.replace(
            req,
            model_id=dc.model_id,
            diagnostic_slice=diag_slice,
            adaptation_data_path=str(cur_p),
            source_retention_tolerance=dc.source_retention_tolerance,
        )
        return req

    # -------------------------------------------------------------------------
    # Branch B: Legacy Flat RepairRequest Dictionary
    # -------------------------------------------------------------------------
    diag_slice = None
    if "diagnostic_slice" in data and isinstance(data["diagnostic_slice"], dict):
        d_slice = data["diagnostic_slice"]
        prov_str = d_slice.get("provenance", "COMPONENT_2_DIAGNOSIS")
        try:
            prov_enum = DiagnosticProvenance(prov_str)
        except ValueError:
            prov_enum = DiagnosticProvenance.COMPONENT_2_DIAGNOSIS

        diag_slice = DiagnosticSliceDefinition(
            slice_id=d_slice.get("slice_id", "SLICE-001"),
            slice_predicate=d_slice.get("slice_predicate", ""),
            affected_features=tuple(d_slice.get("affected_features", ())),
            reported_slice_support=float(d_slice.get("reported_slice_support", 0.25)),
            diagnostic_confidence=float(d_slice.get("diagnostic_confidence", 1.0)),
            provenance=prov_enum,
            is_non_oracle=bool(d_slice.get("is_non_oracle", prov_enum != DiagnosticProvenance.SIMULATION_ORACLE)),
        )

    # Enums conversion
    d_type = DriftType(data.get("drift_type", "CONCEPT_DRIFT")) if "drift_type" in data else DriftType.CONCEPT_DRIFT
    d_sev = DriftSeverity(data.get("drift_severity", "HIGH")) if "drift_severity" in data else DriftSeverity.HIGH
    mode = ExecutionMode(data.get("execution_mode", "EXECUTE")) if "execution_mode" in data else ExecutionMode.EXECUTE
    obj = RepairObjective(data.get("repair_objective", "BALANCED_RETENTION")) if "repair_objective" in data else RepairObjective.BALANCED_RETENTION

    return RepairRequest(
        request_id=data.get("request_id", None) or f"REQ-{data.get('dataset_id', 'AUTO')}",
        schema_version=data.get("schema_version", "1.0.0"),
        dataset_id=data.get("dataset_id", "telco_churn"),
        model_id=data.get("model_id", "telco_rf_experimental_source"),
        drift_detected=bool(data.get("drift_detected", True)),
        drift_type=d_type,
        drift_severity=d_sev,
        drift_severity_score=float(data.get("drift_severity_score", 0.65)),
        affected_features=tuple(data.get("affected_features", ())),
        adaptation_budget_size=int(data.get("adaptation_budget_size", 200)),
        has_authentic_adaptation_labels=bool(data.get("has_authentic_adaptation_labels", True)),
        diagnostic_slice=diag_slice,
        repair_objective=obj,
        execution_mode=mode,
        max_cpu_cores=int(data.get("max_cpu_cores", 4)),
        max_memory_mb=float(data.get("max_memory_mb", 4096.0)),
        max_runtime_seconds=float(data.get("max_runtime_seconds", 300.0)),
        source_retention_tolerance=float(data.get("source_retention_tolerance", 0.05)),
    )


def _load_package(package_or_path: str | Path | RepairPackage) -> dict[str, Any]:
    """Helper to load a RepairPackage dictionary."""
    if isinstance(package_or_path, RepairPackage):
        return package_or_path.to_dict()

    p = Path(package_or_path)
    if not p.is_file():
        raise FileNotFoundError(f"Package manifest file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_repair_request(request: RepairRequest | dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate a RepairRequest schema and authorized data references."""
    try:
        req = _parse_request(request)
        return {
            "valid": True,
            "request_id": req.request_id,
            "dataset_id": req.dataset_id,
            "model_id": req.model_id,
            "drift_severity": req.drift_severity.value,
            "adaptation_budget_size": req.adaptation_budget_size,
            "errors": [],
        }
    except Exception as exc:
        return {
            "valid": False,
            "request_id": getattr(request, "request_id", "UNKNOWN") if isinstance(request, RepairRequest) else "UNKNOWN",
            "errors": [str(exc)],
        }


def view_prescription_reasoning(
    request: RepairRequest | dict[str, Any] | str | Path,
    adaptation_df: pd.DataFrame | None = None,
    decision_mode: str | DecisionMode = "RULE_BASED",
) -> dict[str, Any]:
    """Evaluate prescription governance gates and return recommended strategy and reasoning."""
    req = _parse_request(request)
    mode_enum = DecisionMode(decision_mode.upper()) if isinstance(decision_mode, str) else decision_mode
    engine = RepairPrescriptionEngine()
    decision = engine.prescribe(req, adaptation_df=adaptation_df, decision_mode=mode_enum)
    return decision.to_dict()


def execute_repair(
    request: RepairRequest | dict[str, Any] | str | Path,
    output_directory: str | Path | None = None,
    df_source_train: pd.DataFrame | None = None,
    df_adaptation: pd.DataFrame | None = None,
    df_dev_val: pd.DataFrame | None = None,
    df_source_eval: pd.DataFrame | None = None,
    pre_generated_candidates_df: pd.DataFrame | None = None,
    decision_mode: str | DecisionMode = "RULE_BASED",
) -> tuple[RepairPackage, OrchestratorExecutionLog]:
    """Execute end-to-end repair pipeline for a given RepairRequest."""
    req = _parse_request(request)
    mode_enum = DecisionMode(decision_mode.upper()) if isinstance(decision_mode, str) else decision_mode
    if output_directory is None:
        target_out = COMPONENT3_ROOT / "results" / "runs" / f"run_{req.request_id.lower()}"
    else:
        target_out = Path(output_directory)

    orchestrator = RepairOrchestrator()
    return orchestrator.execute_repair(
        request=req,
        output_directory=target_out,
        df_source_train=df_source_train,
        df_adaptation=df_adaptation,
        df_dev_val=df_dev_val,
        df_source_eval=df_source_eval,
        pre_generated_candidates_df=pre_generated_candidates_df,
        decision_mode=mode_enum,
    )


def inspect_repair_status(package_or_path: str | Path | RepairPackage) -> dict[str, Any]:
    """Inspect current lifecycle and acceptance status of a repair package."""
    pkg_data = _load_package(package_or_path)
    return {
        "package_id": pkg_data.get("package_id"),
        "request_id": pkg_data.get("request_id"),
        "run_id": pkg_data.get("run_id"),
        "execution_status": pkg_data.get("execution_status"),
        "package_status": pkg_data.get("package_status"),
        "chosen_strategy": pkg_data.get("chosen_strategy"),
        "quantity_selected": pkg_data.get("quantity_selected", 0),
        "requires_human_approval": pkg_data.get("requires_human_approval", True),
        "rejection_reasons": pkg_data.get("rejection_reasons", []),
    }


def inspect_validation_results(package_or_path: str | Path | RepairPackage) -> dict[str, Any]:
    """Inspect validation sub-reports (structural, privacy, fidelity, plausibility)."""
    pkg_data = _load_package(package_or_path)
    val_report = pkg_data.get("validation_report", {})
    return {
        "package_id": pkg_data.get("package_id"),
        "structural_report": val_report.get("structural_report", {}),
        "privacy_risk_report": pkg_data.get("privacy_risk_indicators", {}),
        "fidelity_report": pkg_data.get("fidelity_indicators", {}),
        "utility_metrics": pkg_data.get("utility_metrics", {}),
        "source_retention_metrics": pkg_data.get("source_retention_metrics", {}),
    }


def inspect_package(package_or_path: str | Path | RepairPackage) -> dict[str, Any]:
    """Retrieve complete serializable dictionary representation of a RepairPackage."""
    return _load_package(package_or_path)


def check_system_readiness() -> dict[str, Any]:
    """Check system readiness across datasets, baseline models, generators, and knowledge base."""
    ds_reg = get_dataset_registry()
    model_reg = get_model_registry()
    plugin_mgr = GeneratorPluginManager()
    kb = get_repair_knowledge_base()

    datasets = ds_reg.list_datasets()
    models = [m.model_id for m in model_reg.list_models()]
    generators = list(plugin_mgr.available_generators())

    # Check model integrity
    integrity_checks: dict[str, bool] = {}
    for m in model_reg.list_models():
        integrity_checks[m.model_id] = model_reg.verify_integrity(m.model_id)

    all_integrity_ok = all(integrity_checks.values())

    return {
        "system_status": "READY" if all_integrity_ok else "DEGRADED",
        "registered_datasets": datasets,
        "registered_models": models,
        "model_integrity_verified": integrity_checks,
        "available_generators": generators,
        "knowledge_base_records": kb.metadata.get("total_experiments_recorded", 0),
        "source_eval_count": kb.get_source_eval_count(),
        "dev_val_count": kb.get_dev_val_count(),
    }


def validate_external_contract(raw_payload: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate incoming three-section contract schema without executing repair."""
    ok, errs, container = validate_three_section_contract(raw_payload)
    if not ok or container is None:
        return {
            "valid": False,
            "errors": errs,
            "contract_type": "three_section_repair_input",
        }
    return {
        "valid": True,
        "errors": [],
        "contract_type": "three_section_repair_input",
        "model_id": container.data_context.model_id,
        "dataset_id": container.data_context.dataset_id,
        "drift_detected": container.c1_evidence.is_drift_detected,
        "drift_type": container.c1_evidence.drift_type,
        "predicted_harm": container.c1_evidence.predicted_harm_proportion,
        "slice_rule": container.c2_diagnosis.affected_slice_rule,
    }


def export_c4_flat_contract(
    package: RepairPackage,
    output_path: str | Path | None = None,
    repair_version: str | None = None,
) -> dict[str, Any]:
    """Export finalized RepairPackage as flat 12-field contract for Component 4."""
    adapter = C4RetrainingAdapter()
    return adapter.export_flat_contract(
        package=package,
        output_path=output_path,
        repair_version=repair_version,
    )


def execute_contract_repair(
    request: dict[str, Any] | str | Path,
    output_c4_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    pre_generated_candidates_df: pd.DataFrame | None = None,
    decision_mode: str | DecisionMode = "RULE_BASED",
) -> tuple[dict[str, Any], RepairPackage, OrchestratorExecutionLog]:
    """Canonical integration entry point for multi-component pipeline.
    
    Receives 3-section payload (c1_evidence + c2_diagnosis + data_context),
    validates contracts, resolves dependencies, executes repair, and exports flat C4 package.
    """
    pkg, log = execute_repair(
        request=request,
        output_directory=output_directory,
        pre_generated_candidates_df=pre_generated_candidates_df,
        decision_mode=decision_mode,
    )
    c4_contract = export_c4_flat_contract(
        package=pkg,
        output_path=output_c4_path,
        repair_version=f"c3-{pkg.request_id[-8:].lower() if pkg.request_id else '001'}",
    )
    return c4_contract, pkg, log

