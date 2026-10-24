"""Drift Lab endpoints (platform test bench, owner: Sajivan K)."""
from typing import List, Literal
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import contracts, runs
from ..components import C1
from ..lab import LAB, N_FEATURES, LabError
from ..orchestrator import run_pipeline
from ..settings import DATASETS, REPO_ROOT, C3_UPLOAD_DIR, c3_data_context

router = APIRouter(prefix="/api/lab", tags=["Drift Lab"])

DRIFT_HELP = {
    "none": "No change: a random batch from unseen test data. C1 should stay quiet.",
    "marginal_single": "Pick 1 feature. Numbers shift up by strength x the feature's spread; "
                       "categories are pushed toward their rarest value.",
    "marginal_multi": "Pick 2 to 5 features; each shifts as above.",
    "interaction": "Pick exactly 2 features. The second is shuffled between some rows: each column "
                   "looks normal, but their relationship breaks.",
    "combined": "Pick 3 features: the first shifts, the other two get an interaction break.",
}


class LabRun(BaseModel):
    dataset: str = "uci"
    drift_type: Literal["none", "marginal_single", "marginal_multi", "interaction", "combined"] = "none"
    features: List[str] = Field(default_factory=list)
    magnitude: float = 0.55
    batch_size: int = 1500
    seed: int = 42
    mode: Literal["c1", "c1c2", "pipeline"] = "c1"   # c1c2: C1 then C2, stops before C3
    save_batch: bool = False          # also save the batch CSV (with labels) next to the run


def _engine(dataset):
    if dataset not in DATASETS:
        raise HTTPException(404, f"unknown dataset '{dataset}'; use one of {list(DATASETS)}")
    try:
        return C1.engine(dataset)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.get("/options")
def options(dataset: str = "uci"):
    """What the lab can build for one dataset: feature groups, batch sizes, drift types."""
    eng = _engine(dataset)
    info = LAB.feature_info(dataset, eng)
    sizes = list(eng.sizes)
    return {"dataset": dataset, "label": DATASETS[dataset]["label"],
            "harm_metric": getattr(eng, "harm_metric", "accuracy_drop"),
            "batch_sizes": sizes, "default_batch_size": sizes[len(sizes) // 2],
            "features": {k: info[k] for k in ("high", "low", "other")}, "note": info["note"],
            "drift_types": [{"id": k, "min": v[0], "max": v[1], "help": DRIFT_HELP[k]} for k, v in N_FEATURES.items()]}


@router.post("/run")
def run(req: LabRun):
    """Build a batch with a known drift, then run C1 only, C1 -> C2, or the full pipeline (labels removed)."""
    eng = _engine(req.dataset)
    try:
        batch, truth, injection, labelled = LAB.make_batch(eng, req.dataset, req.drift_type, req.features,
                                                 req.magnitude, req.batch_size, req.seed)
    except LabError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, f"lab data not found: {e}")

    run_id = runs.new_run_id()
    directory = C3_UPLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"lab_{run_id}.csv"
    labelled.to_csv(path, index=False)
    context = {**c3_data_context(DATASETS[req.dataset]["model_id"]),
               "dataset": req.dataset, "dataset_id": req.dataset,
               "model_id": DATASETS[req.dataset]["model_id"], "current_dataset_uri": str(path),
               "batch_id": path.stem, "label_status": "AVAILABLE", "label_source": "drift_lab_ground_truth_available_to_c3_only"}
    out = {"run_id": run_id, "truth": truth, "injection": injection, "mode": req.mode}
    if req.mode == "c1":
        c1 = C1.run(batch, {"context": context})["c1"]
        out["c1"] = c1
        out["contract_errors"] = contracts.validate("c1", c1)
    else:
        result = run_pipeline(batch, context, stop_after="c2" if req.mode == "c1c2" else None)
        names = ", ".join(req.features) or "clean"
        result["origin"] = {"kind": "drift_lab", "rows": len(batch), "run_ref": run_id,
                            "label": f"Drift Lab: {req.drift_type.replace('marginal_', '')} ({names})",
                            "truth": truth}
        if req.mode == "pipeline":
            runs.save_latest(result)
        out["pipeline"] = result
        out["c1"] = result["outputs"].get("c1")

    record = {"run_id": run_id, "source": "drift_lab", "mode": req.mode, "dataset": req.dataset,
              "injection": injection, "truth": truth, "upload": None, "c1": out["c1"],
              "reference_prediction_stats": C1.reference_stats(req.dataset), "pipeline": out.get("pipeline")}
    out["summary"] = runs.save_run(record, labelled if req.save_batch else None)
    return out
