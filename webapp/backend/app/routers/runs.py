"""Saved runs: list, one run, its batch CSV, and a CSV export of all runs (Run history page)."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .. import runs

router = APIRouter(prefix="/api/runs", tags=["Run history"])


@router.get("")
def list_runs():
    """One summary per saved run, newest first."""
    return runs.list_runs()


@router.get("/export.csv")
def export():
    """All run summaries as CSV (for the report)."""
    return Response(runs.export_csv(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="platform_runs.csv"'})


@router.get("/{run_id}")
def get_run(run_id: str):
    """The full saved record of one run."""
    rec = runs.get_run(run_id)
    if rec is None:
        raise HTTPException(404, f"run '{run_id}' not found")
    return rec


@router.get("/{run_id}/batch")
def get_batch(run_id: str):
    """The batch CSV of a run, if it was saved (Drift Lab, "Save the batch CSV")."""
    path = runs.batch_path(run_id)
    if path is None:
        raise HTTPException(404, f"no saved batch for run '{run_id}'")
    return FileResponse(path, media_type="text/csv", filename=f"batch_{run_id}.csv")
