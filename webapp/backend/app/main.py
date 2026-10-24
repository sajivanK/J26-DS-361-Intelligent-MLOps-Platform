"""
Intelligent MLOps Platform: backend API (FastAPI).
J26-DS-361 | Sajivan K (C1), Tharmika N (C2), Mathusan K (C3), Kesigan M (C4)

Run from the repo root:
    uvicorn app.main:app --app-dir webapp/backend --port 8000
Then open http://127.0.0.1:8000/docs

Member page routers (c2, c3, ...) are loaded one by one: if one cannot be imported on this machine
(e.g. a library it needs is not installed), the backend still starts without that page's API, prints
the reason, and lists it in /api/health under "routers_missing". One member's setup must never stop
the whole platform.
"""
import importlib
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .components import PIPELINE
from .routers import alerts, c1, health, lab, pipeline, runs
from .routers import c4 as c4_page
from .settings import FRONTEND_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    for comp in PIPELINE:          # load every component once, before the first request
        comp.safe_load()
    yield


app = FastAPI(title="J26-DS-361 Intelligent MLOps Platform API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=FRONTEND_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Run-Id"])

app.include_router(health.router)
app.include_router(c1.router)

MEMBER_ROUTERS = ("c2", "c3")  # C4 is registered explicitly below
ROUTERS_MISSING = {}
for _name in MEMBER_ROUTERS:
    try:
        app.include_router(importlib.import_module(f".routers.{_name}", __package__).router)
    except Exception as _e:  # noqa: BLE001
        ROUTERS_MISSING[_name] = f"{type(_e).__name__}: {_e}"
        traceback.print_exc()
        print(f"[{_name}] page API NOT loaded: {ROUTERS_MISSING[_name]}", flush=True)

app.include_router(pipeline.router)
app.include_router(lab.router)
app.include_router(runs.router)
app.include_router(c4_page.router)
app.include_router(alerts.router)

# Persistent C4 recorded-assessment workspace.
from .routers import c4_workspace
app.include_router(c4_workspace.router)
