"""
Intelligent MLOps Platform: backend API (FastAPI).
J26-DS-361 | Sajivan K (C1), Tharmika N (C2), Mathusan K (C3), Kesigan M (C4)

Run from the repo root:
    uvicorn app.main:app --app-dir webapp/backend --port 8000
Then open http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .components import PIPELINE
from .routers import c1, health, lab, pipeline, runs
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
app.include_router(pipeline.router)
app.include_router(lab.router)
app.include_router(runs.router)
app.include_router(c4_page.router)
