"""Full pipeline endpoints: one batch through C1 -> C2 -> C3 -> C4, and the latest run."""
from fastapi import APIRouter, File, Form, UploadFile

from .. import runs
from ..orchestrator import run_pipeline
from ..uploads import read_batch

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run")
def run(dataset: str = Form("uci"), file: UploadFile = File(...)):
    """Run the whole platform on an uploaded CSV batch. Each step reports real/fake and contract checks."""
    batch, context, removed = read_batch(file, dataset)
    result = run_pipeline(batch, context)
    result["labels_removed"] = removed
    result["origin"] = {"kind": "upload", "label": context["current_dataset_uri"], "rows": len(batch)}
    runs.save_latest(result)
    return result


@router.get("/latest")
def latest():
    """The most recent full pipeline run (upload or Drift Lab), or null if there has been none."""
    return runs.latest()
