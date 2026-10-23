"""Lightweight FastAPI server for Component 3 (Adaptive Intelligent Synthetic Data Repair Engine).

Provides REST API endpoints for health checks and explicit repair execution.

Can be run standalone:
    python -m component3.src.server --port 8003
"""

from __future__ import annotations
from component3.src.integration_storage import upload_root, run_root

import argparse
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional

import pandas as pd
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import uvicorn
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPONENT3_ROOT = REPO_ROOT / "component3"

from component3.src.api import (
    execute_contract_repair,
    export_c4_flat_contract,
    validate_external_contract,
)
from component3.src.contracts.external_contracts import (
    ContractValidationError,
    DependencyResolutionError,
)

logger = logging.getLogger(__name__)

# Global mutex to prevent concurrent CPU-heavy model training
_REPAIR_LOCK = threading.Lock()

app = FastAPI(
    title="Component 3 API Server",
    description="Adaptive Intelligent Synthetic Data Repair Engine (Mathusan K)",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

def _get_default_telco_demo_contract() -> dict[str, Any]:
    """Returns the verified, canonical Telco high-covariate-shift demonstration contract."""
    from sklearn.model_selection import train_test_split
    from component3.src.datasets.dataset_registry import get_dataset_registry
    from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule
    import uuid
    frame = pd.read_csv(COMPONENT3_ROOT / "data/raw/telco_churn.csv")
    frame["TotalCharges"] = pd.to_numeric(frame.TotalCharges, errors="coerce").fillna(0.)
    reference, current = train_test_split(frame, test_size=.3, stratify=frame.Churn, random_state=40)
    directory = upload_root() / "scenarios" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    reference.to_csv(directory / "reference.csv", index=False)
    current.to_csv(directory / "current.csv", index=False)
    metadata = get_dataset_registry().get("telco_churn")
    rule = "Contract == 'Month-to-month' and tenure <= 12"
    return {
        "c1_evidence": {
            "drift_detected": True,
            "drift_type": "covariate_shift",
            "predicted_accuracy_drop": 18.0,
            "confidence_interval": [12.0, 24.0],
            "drift_severity": "HIGH",
            "affected_features": ["Contract", "tenure", "MonthlyCharges"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "Contract",
            "root_cause_category": "covariate_shift",
            "affected_slice_rule": "Contract == 'Month-to-month' and tenure <= 12",
            "affected_features_ranked": ["Contract", "MonthlyCharges", "tenure"],
            "slice_support": float(evaluate_slice_rule(current, rule).mean()),
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.85,
        },
        "data_context": {
            "model_id": "telco_rf_experimental_source",
            "dataset_id": "telco_churn",
            "reference_dataset_uri": str(directory / "reference.csv"),
            "current_dataset_uri": str(directory / "current.csv"),
            "target_column": "Churn",
            "preprocessing_uri": "component3/models/experimental/telco_preprocessor_experimental_source.joblib",
            "adaptation_budget_size": 200,
            "source_retention_tolerance": 0.05,
            "label_status": "AVAILABLE",
            "feature_columns": metadata.feature_columns,
            "categorical_features": metadata.categorical_features,
            "generator_epochs": 1,
        },
    }


def _execute_repair_job(payload: dict[str, Any]) -> JSONResponse | dict[str, Any]:
    """Internal helper to execute a validated synthetic repair job with concurrency locking."""
    # Request payload size guard (500 KB limit)
    try:
        raw_size = len(json.dumps(payload))
        if raw_size > 500_000:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "PAYLOAD_TOO_LARGE",
                    "error_type": "PayloadTooLargeError",
                    "detail": f"Payload size ({raw_size} bytes) exceeds maximum allowed limit (500,000 bytes).",
                },
            )
    except Exception:
        pass

    # Acquire concurrency lock to prevent overlapping CPU training jobs
    acquired = _REPAIR_LOCK.acquire(blocking=False)
    if not acquired:
        return JSONResponse(
            status_code=429,
            content={
                "status": "BUSY",
                "error_type": "WorkerBusyError",
                "detail": "A synthetic repair job is already running on this worker. Concurrent CPU execution is blocked to preserve system stability.",
            },
        )

    start_time = time.perf_counter()
    try:
        # Validate incoming contract
        val_res = validate_external_contract(payload)
        if not val_res.get("valid"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "CONTRACT_VALIDATION_ERROR",
                    "errors": val_res.get("errors", []),
                    "c4_flat_contract": None,
                },
            )

        # Run real repair orchestrator
        c4_flat, package, orch_log = execute_contract_repair(
            request=payload,
            output_directory=run_root(),
            decision_mode="RULE_BASED",
        )

        duration = time.perf_counter() - start_time

        # Map package status to clear API outcome status
        pkg_status = package.package_status.value if hasattr(package.package_status, "value") else str(package.package_status)
        if pkg_status == "APPROVED_FOR_REVIEW":
            outcome_status = "COMPLETED_APPROVED"
        elif pkg_status == "REJECTED":
            outcome_status = "COMPLETED_REJECTED"
        elif pkg_status == "ABSTAINED_NO_REPAIR":
            outcome_status = "ABSTAINED_NO_REPAIR"
        else:
            outcome_status = f"COMPLETED_{pkg_status}"

        from component3.src.live_repair import response_for_package
        return {
            **response_for_package(package, orch_log),
            "status": outcome_status,
            "run_id": package.run_id,
            "package_id": package.package_id,
            "dataset": package.dataset_id,
            "execution_status": package.execution_status,
            "package_status": pkg_status,
            "chosen_strategy": str(package.chosen_strategy),
            "generator": c4_flat["generator"],
            "quantity_selected": package.quantity_selected,
            "duration_seconds": round(duration, 3),
            "c4_flat_contract": c4_flat,
            "candidate_dataset_path": str(package.candidate_dataset_path) if package.candidate_dataset_path else None,
            "candidate_dataset_sha256": package.candidate_dataset_sha256,
        }

    except ContractValidationError as cve:
        return JSONResponse(
            status_code=422,
            content={
                "status": "CONTRACT_VALIDATION_ERROR",
                "error_type": "ContractValidationError",
                "detail": str(cve),
            },
        )
    except DependencyResolutionError as dre:
        return JSONResponse(
            status_code=400,
            content={
                "status": "DEPENDENCY_RESOLUTION_ERROR",
                "error_type": "DependencyResolutionError",
                "detail": str(dre),
                "resolution_recommendation": "Deliver required external dataset batch before running repair.",
            },
        )
    except Exception as exc:
        logger.exception("Live repair failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "status": "INTERNAL_EXECUTION_ERROR",
                "error_type": "ExecutionError",
                "detail": "An internal execution error occurred during repair processing. Please inspect server logs.",
            },
        )
    finally:
        _REPAIR_LOCK.release()


@app.get("/api/c3/health")
def health():
    """Returns engine health, supported generators, and operational metadata."""
    return {
        "status": "ok",
        "component": "c3",
        "title": "C3 Synthetic Data Repair",
        "owner": "Mathusan K",
        "version": "2.0.0",
        "contract_version": "1.1",
        "supported_generators": ["TVAE", "CTGAN", "TabDDPM"],
        "default_decision_mode": "RULE_BASED",
        "supported_tasks": ["tabular_binary_classification"],
        "supported_datasets": "Caller-supplied tabular datasets with explicit schema and labeled independent evaluation data.",
        "storage_root": str(run_root()),
    }


@app.get("/api/c3/packages/{run_id}/dataset")
def download_dataset(run_id: str):
    if not re.fullmatch(r"c3-[0-9a-f]{32}", run_id):
        return JSONResponse(status_code=404, content={"detail": "Unknown repair run."})
    from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
    directory = run_root() / run_id
    manifests = list(directory.glob("*_manifest.json"))
    if len(manifests) != 1:
        return JSONResponse(status_code=404, content={"detail": "Package not found."})
    package = RepairPackage(**json.loads(manifests[0].read_text(encoding="utf-8")))
    if package.package_status != RepairPackageStatus.APPROVED_FOR_REVIEW:
        return JSONResponse(status_code=409, content={"detail": "Package is not approved."})
    path = Path(package.candidate_dataset_path or "").resolve()
    if not path.is_relative_to(directory.resolve()) or not package.verify_artifact_integrity()[0]:
        return JSONResponse(status_code=409, content={"detail": "Artifact integrity check failed."})
    return FileResponse(path, filename=f"{run_id}_repair.csv", media_type="text/csv")


@app.post("/api/c3/repair")
def run_repair(payload: Optional[dict[str, Any]] = Body(None)):
    """Executes a real Component 3 repair workflow.
    
    Requires an explicit non-empty contract payload containing c1_evidence, c2_diagnosis, and data_context.
    Missing or empty payloads return HTTP 400 Bad Request to prevent accidental execution.
    """
    if not payload or not isinstance(payload, dict) or len(payload) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "status": "MISSING_PAYLOAD",
                "error_type": "EmptyPayloadError",
                "detail": "Repair payload cannot be empty. An explicit 3-section contract ({c1_evidence, c2_diagnosis, data_context}) is required. For the Telco demonstration, call POST /api/c3/demo/repair.",
            },
        )

    return _execute_repair_job(payload)


@app.post("/api/c3/demo/repair")
def run_demo_repair(payload: Optional[dict[str, Any]] = Body(None)):
    """Explicitly triggers the verified Telco Churn high-covariate-shift demonstration repair.
    
    Accepts optional payload overrides, or defaults to the verified Telco high-drift scenario.
    """
    if not payload or not isinstance(payload, dict) or len(payload) == 0:
        payload = _get_default_telco_demo_contract()
    return _execute_repair_job(payload)


def main():
    parser = argparse.ArgumentParser(description="Component 3 REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="Port (default: 8003)")
    args = parser.parse_args()

    print(f"Starting Component 3 Server at http://{args.host}:{args.port}")
    print(f"Health endpoint: http://{args.host}:{args.port}/api/c3/health")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
