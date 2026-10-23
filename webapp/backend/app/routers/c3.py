"""C3 real-input API and explicitly separate research demonstrations."""
from pathlib import Path
import json
import re
import threading
import uuid

from fastapi import APIRouter, Body, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from .. import runs
from ..components import BY_NAME
from ..components.c3_results import RECORDED
from ..settings import REPO_ROOT, MAX_UPLOAD_MB, c3_data_context
from component3.src.contracts.external_contracts import ContractValidationError, DependencyResolutionError
from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
from component3.src.live_repair import execute_live_repair, response_for_package

router = APIRouter(prefix="/api/c3", tags=["C3 Synthetic Data Repair"])
_LOCK = threading.Lock()


@router.post("/artifacts")
def upload_artifact(role: str = Form(...), file: UploadFile = File(...)):
    roles = {"model": ("models", ".joblib", ".pkl", ".pickle"),
             "preprocessor": ("models", ".joblib", ".pkl", ".pickle"),
             **{key: ("data", ".csv") for key in ("reference_dataset", "current_dataset", "source_evaluation", "evaluation_dataset")}}
    if role not in roles:
        raise HTTPException(422, "Unknown artifact role.")
    category, *extensions = roles[role]
    extension = Path(file.filename or "").suffix.lower()
    if extension not in extensions:
        raise HTTPException(422, f"Expected one of {extensions} for {role}.")
    root = REPO_ROOT / "component3" / category / "integration_uploads"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid.uuid4().hex}{extension}"
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"Artifact exceeds {MAX_UPLOAD_MB} MB; configure an authorized local path for larger artifacts.")
                output.write(chunk)
        if size == 0:
            raise HTTPException(422, "Artifact is empty.")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    from component3.src.contracts.repair_package import compute_file_sha256
    return {"role": role, "uri": str(path.relative_to(REPO_ROOT)), "sha256": compute_file_sha256(path), "bytes": size}


@router.get("/info")
def info():
    comp = BY_NAME["c3"]
    return {"name": comp.name, "title": comp.title, "owner": comp.owner,
        "source": comp.source, "loaded": comp.loaded, "error": comp.error,
        "generators": ["TVAE", "CTGAN", "TabDDPM"], "decision_modes": ["RULE_BASED"],
        "engine_version": "2.0.0", "contract_version": "1.1",
        "supported_tasks": ["tabular_binary_classification"],
        "model_requirements": "Fitted sklearn-compatible binary classifier or Pipeline; clone, fit, predict and classes_ required.",
        "dataset_requirements": "Explicit feature schema and labeled independent adaptation/evaluation data; no dataset-name restrictions.",
        "upstream_sources": {name: BY_NAME[name].source for name in ("c1", "c2", "c4")},
        "quality_gates": {"fidelity_ks_max": 0.35, "privacy_dcr_min": 0.05,
                          "utility_gain_min": 0.01, "target_coverage_min": 0.8, "max_exact_copies": 0}}


@router.get("/benchmarks")
def benchmarks(dataset: str | None = None):
    if dataset is None:
        return RECORDED
    key = "uci_adult" if dataset == "uci" else dataset
    if key not in RECORDED:
        raise HTTPException(404, "No recorded benchmark for this dataset; real repair accepts explicit dataset metadata.")
    return {dataset: RECORDED[key]}


@router.get("/upstream/latest")
def latest_upstream():
    latest = runs.latest()
    if not latest:
        raise HTTPException(404, "No platform run available. Supply a C1/C2 contract or run the upstream components.")
    sources = latest.get("component_sources", {})
    for step in latest.get("steps", []):
        sources.setdefault(step["component"], step.get("source"))
    outputs = latest.get("outputs", {})
    if not outputs.get("c1") or not outputs.get("c2_to_c3"):
        raise HTTPException(422, "Latest run does not contain both C1 and C2 evidence.")
    if sources.get("c1") != "real" or sources.get("c2") != "real":
        raise HTTPException(409, "Latest run contains placeholder C1/C2 evidence. Connect the actual upstream components or import their contract JSON.")
    context = latest.get("context", {})
    context = {**c3_data_context(context.get("model_id", "")), **context}
    context.setdefault("dataset_id", context.get("dataset"))
    return {"c1_evidence": outputs["c1"], "c2_diagnosis": outputs["c2_to_c3"],
            "data_context": context, "upstream_run_id": latest.get("run_id")}


@router.post("/repair")
def repair(response: Response, payload: dict = Body(...)):
    if not _LOCK.acquire(blocking=False):
        raise HTTPException(429, "A C3 repair is already running on this worker.")
    try:
        package, log = execute_live_repair(payload)
        result = response_for_package(package, log)
        response.headers["X-Run-Id"] = package.run_id
        return result
    except ContractValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except DependencyResolutionError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _LOCK.release()


@router.get("/packages/{run_id}/dataset")
def download_package(run_id: str):
    if not re.fullmatch(r"c3-[0-9a-f]{32}", run_id):
        raise HTTPException(404, "Unknown repair run.")
    root = REPO_ROOT / "component3/results/live_runs" / run_id
    manifests = list(root.glob("*_manifest.json"))
    if len(manifests) != 1:
        raise HTTPException(404, "Repair package not found.")
    package = RepairPackage(**json.loads(manifests[0].read_text(encoding="utf-8")))
    if package.package_status != RepairPackageStatus.APPROVED_FOR_REVIEW:
        raise HTTPException(409, "Only approved repair packages can be downloaded.")
    path = Path(package.candidate_dataset_path or "").resolve()
    if not path.is_relative_to(root.resolve()):
        raise HTTPException(409, "Package artifact location mismatch.")
    valid, errors = package.verify_artifact_integrity()
    if not valid:
        raise HTTPException(409, "; ".join(errors))
    return FileResponse(path, filename=f"{run_id}_repair.csv", media_type="text/csv")


@router.get("/scenarios")
def scenarios(dataset: str = "uci"):
    if dataset != "uci":
        raise HTTPException(404, "No configured research scenarios for this dataset.")
    from component3.src.engine_uci import UCI_SCENARIOS
    return list(UCI_SCENARIOS.values())


@router.get("/scenario/{scenario_id}/evidence")
def scenario_evidence(scenario_id: str, dataset: str = "uci"):
    if dataset != "uci":
        raise HTTPException(404, "Unknown research dataset.")
    from component3.src.engine_uci import get_scenario_evidence
    try:
        evidence = get_scenario_evidence(scenario_id)
        return {key: evidence[key] for key in ("c1", "c2", "scenario", "ground_truth")} | {"mode": "simulation"}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/demo/repair")
def demo_repair(response: Response, payload: dict = Body(default={})):
    if payload.get("dataset", "uci") != "uci":
        raise HTTPException(422, "This explicit scenario demonstration supports uci only. Use /repair with supplied artifacts for other datasets.")
    if not _LOCK.acquire(blocking=False):
        raise HTTPException(429, "A C3 repair is already running on this worker.")
    try:
        from component3.src.engine_uci import execute_c3_repair_flow
        result = execute_c3_repair_flow(scenario_key=payload.get("scenario", "education_num_marginal"),
            custom_c1=payload.get("c1"), custom_c2=payload.get("c2"))
        response.headers["X-Run-Id"] = result["run_id"]
        return result
    except (ContractValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except DependencyResolutionError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _LOCK.release()
