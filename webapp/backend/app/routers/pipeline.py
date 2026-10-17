"""Full pipeline endpoint: one batch through C1 -> C2 -> C3 -> C4."""
from fastapi import APIRouter, File, Form, UploadFile

from ..orchestrator import run_pipeline
from ..uploads import read_batch

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run")
def run(dataset: str = Form("uci"), file: UploadFile = File(...)):
    """Run the whole platform on an uploaded CSV batch. Each step reports real/fake and contract checks."""
    batch, context, removed = read_batch(file, dataset)
    result = run_pipeline(batch, context)
    result["labels_removed"] = removed
    return result
