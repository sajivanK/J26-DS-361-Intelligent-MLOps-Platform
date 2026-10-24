"""Platform status: are the four components loaded, and are they real or fake?"""
from fastapi import APIRouter

from ..components import PIPELINE
from ..settings import DATASETS

router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/health")
def health():
    from ..main import ROUTERS_MISSING          # member page APIs that could not be imported
    comps = [c.status() for c in PIPELINE]
    ok = all(c["loaded"] for c in comps) and not ROUTERS_MISSING
    return {"status": "ok" if ok else "degraded", "components": comps, "routers_missing": ROUTERS_MISSING}


@router.get("/datasets")
def datasets():
    return [{"id": k, **v} for k, v in DATASETS.items()]
