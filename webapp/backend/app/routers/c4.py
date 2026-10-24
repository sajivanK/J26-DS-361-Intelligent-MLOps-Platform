"""C4 page endpoints, separate from the live pipeline placeholder."""
import logging
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from ..components import c4_worker_client as service

router = APIRouter(prefix="/api/c4", tags=["C4 Release Control"])
log = logging.getLogger(__name__)

class CaseRequest(BaseModel):
    case_id: str = "real-outer-case"
    method: Literal["lightgbm", "linear"] = "lightgbm"
    family: Literal["global", "adaptive"] = "adaptive"
    model_config = ConfigDict(extra="forbid")

def call(fn, *args):
    try:
        return fn(*args)
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown recorded C4 case.")
    except service.C4Unavailable as exc:
        log.exception("C4 artifact loading failed")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        log.exception("C4 evaluation failed")
        raise HTTPException(status_code=500, detail="C4 evaluation failed. Inspect the backend log.")

@router.get("/info")
def info():
    return call(service.catalog)

@router.get("/research-results")
def research_results():
    return call(service.recorded_results)

@router.post("/release/evaluate")
def evaluate(request: CaseRequest):
    return call(service.evaluate, request.case_id, request.method, request.family)

@router.post("/acceptance/verify")
def acceptance():
    return call(service.fixtures)


@router.get("/research-results/hybrid")
def hybrid_research_results():
    return call(service.hybrid_results)


class RevisedCaseRequest(BaseModel):
    case_id: str = "revised-outer-case"
    method: Literal["hybrid", "lightgbm", "linear", "constant"] = "hybrid"
    family: Literal["global", "adaptive"] = "adaptive"
    model_config = ConfigDict(extra="forbid")


@router.post("/release/revised/evaluate")
def revised_evaluate(request: RevisedCaseRequest):
    return call(service.revised_evaluate, request.case_id, request.method, request.family)

@router.post("/sandbox/evaluate")
def sandbox_evaluate(upstream: dict):
    return call(service._request, "/api/c4/sandbox/evaluate", upstream)
