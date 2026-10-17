"""
C2 Diagnosis API .
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Start:   python3.11 -m uvicorn c2.api:app --port 8002
Docs:    http://localhost:8002/docs   (interactive, try requests in the browser)

C2 is stateful: Temporal SHAP (Novelty 1) needs the history of monitoring
cycles, so one running server keeps one engine for the monitored model.

Endpoints
  GET  /health     engine status
  POST /warmup     clean production batches from the first cycles after
                   deployment; sets the SHAP history and the performance noise band
                   body: {"batches": [[row, ...], [row, ...]]}
  POST /diagnose   one monitoring cycle
                   body: {"c1_output": {...C1 v1.1 output...},
                          "production_data": [row, ...]}
                   returns {"to_c3": {...}, "to_c4": {...}, "details": {...},
                            "contract_problems": {"to_c3": [], "to_c4": []}}
  POST /reset      clear the cycle history

Rows are pandas to_dict(orient="records") records with every model feature and
the label column "target" (the performance evidence and localization need
labels). Both outputs are validated against the team contracts before returning.
"""
import time
import warnings

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from c2.src.config import TARGET_COL
from c2.src.outputs import C3_SCHEMA, C4_SCHEMA, C2DiagnosisEngine, validate_output

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="C2 XAI Root Cause Diagnosis Engine", version="1.0")
ENGINE = {"engine": None, "warmed_up": False, "cycles": 0}


class WarmupRequest(BaseModel):
    batches: list[list[dict]]


class DiagnoseRequest(BaseModel):
    c1_output: dict
    production_data: list[dict]


def _engine():
    if ENGINE["engine"] is None:
        ENGINE["engine"] = C2DiagnosisEngine()
    return ENGINE["engine"]


def _frame(rows):
    eng = _engine()
    df = pd.DataFrame(rows)
    needed = eng.a["feature_names"] + [TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(422, f"production_data is missing columns: {missing}")
    if len(df) < 100:
        raise HTTPException(422, "need at least 100 rows per batch")
    return df[needed]


@app.on_event("startup")
def load_engine():
    t = time.time()
    _engine()
    print(f"C2 engine loaded in {time.time() - t:.1f}s")


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": ENGINE["engine"] is not None,
            "warmed_up": ENGINE["warmed_up"], "cycles_diagnosed": ENGINE["cycles"]}


@app.post("/warmup")
def warmup(req: WarmupRequest):
    if not req.batches:
        raise HTTPException(422, "send at least one clean batch")
    frames = [_frame(b) for b in req.batches]
    _engine().warmup(frames)
    ENGINE["warmed_up"], ENGINE["cycles"] = True, 0
    return {"status": "warmed up", "clean_batches": len(frames)}


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    if not ENGINE["warmed_up"]:
        raise HTTPException(409, "call /warmup with clean batches first")
    df = _frame(req.production_data)
    t = time.time()
    result = _engine().diagnose(df, req.c1_output)
    ENGINE["cycles"] += 1
    ref = _engine().a["ref_df"]
    result["contract_problems"] = {
        "to_c3": validate_output(result["to_c3"], C3_SCHEMA, ref),
        "to_c4": validate_output(result["to_c4"], C4_SCHEMA, ref),
    }
    result["details"]["latency_s"] = round(time.time() - t, 3)
    return result


@app.post("/reset")
def reset():
    if ENGINE["engine"] is not None:
        ENGINE["engine"].monitor.reset()
        ENGINE["engine"].root_history = []
    ENGINE["warmed_up"], ENGINE["cycles"] = False, 0
    return {"status": "reset"}