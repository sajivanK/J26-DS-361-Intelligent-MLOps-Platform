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
                   c1_output may be missing or malformed (NFR7): C2 then
                   diagnoses from its own evidence and reports it in
                   details.c1_status / details.warnings.
                   also stores the run in the run history and returns its number
  GET  /runs                 run history table (newest first) + temporal SHAP
                             z per feature across all stored runs
  GET  /runs/latest          full record of the latest run
  GET  /runs/{n}             full record of run n (temporal SHAP, evidence,
                             localization, outputs)
  GET  /runs/{n}/report      business report (FR7): OpenAI-written if
                             OPENAI_API_KEY is set, claim-checked, template
                             fallback; cached, ?refresh=true rewrites it
  GET  /runs/{n}/report.md   the report as a Markdown download
  GET  /audit      last N audit-log records (FR9), ?last=5
  POST /reset      clear the cycle history and the run history

Errors are always structured JSON: {"error": <code>, "message": ..., "problems": [...]}

Rows are pandas to_dict(orient="records") records with every model feature and
the label column "target" (the performance evidence and localization need
labels). Both outputs are validated against the team contracts before returning.
"""
import json
import time
import warnings
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from c2.src.config import TARGET_COL
from c2.src.history import RunHistory, build_run_record
from c2.src.outputs import C3_SCHEMA, C4_SCHEMA, C2DiagnosisEngine, validate_output
from c2.src.plain import PlainLanguage
from c2.src.report import build_report, report_markdown

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="C2 XAI Root Cause Diagnosis Engine", version="1.0")
ENGINE = {"engine": None, "warmed_up": False, "cycles": 0}
HISTORY = RunHistory()
PLAIN = PlainLanguage()


class WarmupRequest(BaseModel):
    batches: list[list[dict]]


class DiagnoseRequest(BaseModel):
    c1_output: Optional[dict] = None
    production_data: list[dict]
    source: str = ""               # shown in the run history, e.g. "Drift Lab"


def _error(status, code, message, problems=None):
    return HTTPException(status, {"error": code, "message": message,
                                  "problems": problems or []})


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
        raise _error(422, "bad_production_data", "production_data is missing columns",
                     [f"missing column '{c}'" for c in missing])
    if len(df) < 100:
        raise _error(422, "bad_production_data", "need at least 100 rows per batch",
                     [f"got {len(df)} rows"])
    bad = [c for c in needed if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise _error(422, "bad_production_data", "columns must be numeric (preprocessed)",
                     [f"column '{c}' is not numeric" for c in bad])
    return df[needed]


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    body = exc.detail if isinstance(exc.detail, dict) else {
        "error": "request_error", "message": str(exc.detail), "problems": []}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(Exception)
def unexpected_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": "diagnosis_failed", "message": f"{type(exc).__name__}: {exc}",
        "problems": []})


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
        raise _error(422, "bad_warmup", "send at least one clean batch")
    frames = [_frame(b) for b in req.batches]
    _engine().warmup(frames)
    ENGINE["warmed_up"], ENGINE["cycles"] = True, 0
    return {"status": "warmed up", "clean_batches": len(frames)}


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    if not ENGINE["warmed_up"]:
        raise _error(409, "not_warmed_up", "call /warmup with clean batches first")
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
    record = HISTORY.add(build_run_record(
        _engine(), result, source=req.source, latency_s=time.time() - t,
        c1_output=req.c1_output, n_rows=len(df)))
    result["run"] = record["run"]
    return result


def _run(n):
    try:
        return HISTORY.get(n)
    except KeyError as e:
        raise _error(404, "run_not_found", str(e.args[0]))


@app.get("/runs")
def runs():
    return {"runs": HISTORY.summaries(), "temporal_shap": HISTORY.z_matrix()}


@app.get("/runs/latest")
def run_latest():
    if not HISTORY.runs:
        raise _error(404, "run_not_found", "no runs yet")
    return HISTORY.latest()


@app.get("/runs/{n}")
def run_detail(n: int):
    return _run(n)


@app.get("/runs/{n}/report")
def run_report(n: int, refresh: bool = False):
    record = _run(n)
    if record["report"] is None or refresh:
        HISTORY.set_report(n, build_report(record, HISTORY.runs, PLAIN))
    return record["report"]


@app.get("/runs/{n}/report.md")
def run_report_md(n: int):
    report = run_report(n)
    return PlainTextResponse(report_markdown(report, _run(n)), media_type="text/markdown",
                             headers={"Content-Disposition":
                                      f'attachment; filename="c2-report-run-{n}.md"'})


@app.get("/audit")
def audit(last: int = 5):
    path = _engine().log_path
    if path is None or not path.exists():
        return {"records": []}
    lines = path.read_text().splitlines()[-max(last, 1):]
    return {"records": [json.loads(x) for x in lines]}


@app.post("/reset")
def reset():
    if ENGINE["engine"] is not None:
        ENGINE["engine"].monitor.reset()
        ENGINE["engine"].root_history = []
    ENGINE["warmed_up"], ENGINE["cycles"] = False, 0
    HISTORY.clear()
    return {"status": "reset"}