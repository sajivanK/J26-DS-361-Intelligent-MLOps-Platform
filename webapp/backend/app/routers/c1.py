"""C1 endpoints (owner: Sajivan K). Each member can add a routers/cX.py like this for their own page."""
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from .. import contracts, runs
from ..components import C1
from ..uploads import read_batch

router = APIRouter(prefix="/api/c1", tags=["C1 Drift Detection"])


@router.get("/engine-info")
def engine_info(dataset: str = "uci"):
    """Engine 1.2.3 details for one dataset: calibration, interval margins, batch sizes, features."""
    try:
        return C1.info(dataset)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.post("/detect")
def detect(response: Response, dataset: str = Form("uci"), file: UploadFile = File(...)):
    """Run C1 on an uploaded CSV batch (prepared feature format). Returns the contract v1.1 output."""
    batch, context, removed = read_batch(file, dataset)
    try:
        out = C1.run(batch, {"context": context})["c1"]
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except (ValueError, TypeError) as e:          # missing features, NaN, too few rows
        raise HTTPException(422, str(e))
    if removed:
        out["warnings"] = list(out.get("warnings", [])) + [f"label column(s) {removed} removed before detection"]
    errs = contracts.validate("c1", out)
    if errs:                                       # should never happen; surfaced loudly if it does
        raise HTTPException(500, {"message": "C1 output does not match contract v1.1", "errors": errs})
    # saved for the Run history; the run id travels in a header so the body stays the exact contract output
    summary = runs.save_run({"source": "c1_upload", "mode": "c1", "dataset": dataset, "injection": None,
                             "truth": None, "c1": out, "pipeline": None,
                             "upload": {"file": context["current_dataset_uri"], "rows": len(batch),
                                        "labels_removed": removed},
                             "reference_prediction_stats": C1.reference_stats(dataset)})
    response.headers["X-Run-Id"] = summary["run_id"]
    return out
