"""C4-only worker. Run with the original C4 Python environment."""
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from app.components import c4_service as service

app = FastAPI(title="C4 Recorded Pilot Worker", version="1.0.0")
log = logging.getLogger(__name__)


class CaseRequest(BaseModel):
    case_id: str = "real-outer-case"
    method: Literal["lightgbm", "linear"] = "lightgbm"
    family: Literal["global", "adaptive"] = "adaptive"
    model_config = ConfigDict(extra="forbid")


def invoke(function, *args):
    try:
        return function(*args)
    except KeyError:
        raise HTTPException(404, "Unknown recorded C4 case.")
    except service.C4Unavailable:
        log.exception("C4 artifacts unavailable")
        raise HTTPException(503, "C4 artifacts unavailable. Inspect the worker log.")
    except Exception:
        log.exception("C4 worker evaluation failed")
        raise HTTPException(500, "C4 evaluation failed. Inspect the worker log.")


@app.get("/health")
def health():
    return {"status": "running", "service": "c4-worker",
            "artifacts_verified": False}


@app.get("/api/c4/info")
def info():
    return service.catalog()


@app.get("/api/c4/research-results")
def research_results():
    return invoke(service.recorded_results)


@app.post("/api/c4/release/evaluate")
def evaluate(request: CaseRequest):
    return invoke(service.evaluate, request.case_id, request.method, request.family)


@app.post("/api/c4/acceptance/verify")
def acceptance():
    return invoke(service.fixtures)


@app.get("/api/c4/research-results/hybrid")
def hybrid_research_results():
    return invoke(service.hybrid_results)


class RevisedCaseRequest(BaseModel):
    case_id: str = "revised-outer-case"
    method: Literal["hybrid", "lightgbm", "linear", "constant"] = "hybrid"
    family: Literal["global", "adaptive"] = "adaptive"
    model_config = ConfigDict(extra="forbid")


@app.post("/api/c4/release/revised/evaluate")
def revised_evaluate(request: RevisedCaseRequest):
    return invoke(service.revised_evaluate, request.case_id, request.method, request.family)

@app.post("/api/c4/pipeline")
def pipeline_evaluate(upstream: dict):
    from app.components.c4_pipeline_service import (
        PipelineInputError,
        PipelineUnavailable,
        evaluate_pipeline,
    )
    try:
        return evaluate_pipeline(upstream)
    except PipelineInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PipelineUnavailable as exc:
        log.exception("C4 live pipeline unavailable")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("C4 live pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="C4 live evaluation failed. Inspect the worker log.",
        ) from exc

@app.post("/api/c4/sandbox/evaluate")
def sandbox_evaluate(upstream: dict):
    from app.components.c4_pipeline_service import (
        PipelineInputError, PipelineUnavailable,
    )
    from app.components.c4_sandbox_service import evaluate_sandbox

    try:
        return evaluate_sandbox(upstream)
    except PipelineInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (PipelineUnavailable, FileNotFoundError) as exc:
        log.exception("C4 sandbox prerequisites unavailable")
        raise HTTPException(
            503, "Sandbox prerequisites unavailable. Inspect worker logs."
        ) from exc
    except ValueError as exc:
        log.exception("C4 sandbox input or artifact validation failed")
        raise HTTPException(
            422, "Sandbox validation failed. Inspect worker logs."
        ) from exc
    except Exception as exc:
        log.exception("C4 sandbox evaluation failed")
        raise HTTPException(
            500, "Sandbox evaluation failed. Inspect worker logs."
        ) from exc
