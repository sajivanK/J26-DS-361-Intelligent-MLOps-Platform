"""Platform status: are the four components loaded, and are they real or fake?"""
from fastapi import APIRouter

from ..components import PIPELINE
from ..settings import DATASETS

router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/health")
def health():
    comps = [c.status() for c in PIPELINE]
    return {"status": "ok" if all(c["loaded"] for c in comps) else "degraded", "components": comps}


@router.get("/datasets")
def datasets():
    return [{"id": k, **v} for k, v in DATASETS.items()]
