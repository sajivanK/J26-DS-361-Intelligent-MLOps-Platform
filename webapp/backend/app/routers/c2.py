"""
C2 endpoints (owner: Tharmika N): the C2 Root Cause Diagnosis page.

    GET  /api/c2/info               engine status, run options, recorded research results
    POST /api/c2/run                build a batch with a known drift, get C1's output for it, run C2
    GET  /api/c2/runs               run history (newest first) + temporal SHAP z across all runs
    GET  /api/c2/runs/latest        full record of the latest run
    GET  /api/c2/runs/{n}           full record of run n (temporal SHAP, evidence, group, outputs)
    GET  /api/c2/runs/{n}/report    business report (FR7), written by the configured language model
                                    and claim-checked, template fallback; ?refresh=true rewrites it
    GET  /api/c2/runs/{n}/report.md the report as a Markdown download
    GET  /api/c2/monitor            Monitoring tab: every batch C1 checked with C2's temporal SHAP
                                    pass (catches up on saved runs first), KPIs, watchlist, z matrix
    POST /api/c2/monitor/check      run the temporal SHAP check now on every saved C1 run not yet monitored
    POST /api/c2/monitor/check-latest  {rows}: C1 (when loaded) and the temporal SHAP check on the newest
                                    rows C1 has checked (500, 1000, 2000 or 5000)
    POST /api/c2/monitor/{b}/diagnose  full diagnosis of monitored batch b (reuses its monitor pass)
    POST /api/c2/reset              start a new monitoring period (empty history and monitoring log,
                                    fresh warm-up)

The page builds its batches with C2's own scenario builder (the shared injector on test.csv rows),
so it works even when the C1 engine files are not on this machine. Neither C1 nor C2 sees the
labels: they stay with the answer key (true accuracy drop, injected features), which the page
shows only to check C2 afterwards.

C1 output: C2 always receives a C1 JSON (contract v1.1). It comes from the real C1 engine when that
is loaded; otherwise from the team's fake C1 builder (shared/contracts/c1_fake_output.py), which
builds a contract-valid C1 output from the injection answer key, as the research guide prescribes
until integration. Each run stores the exact C1 JSON and where it came from.
"""
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .. import contracts
from .. import runs as platform_runs
from ..components import BY_NAME, C1
from ..components.c2_results import RECORDED
from ..lab import N_FEATURES
from ..settings import DATASETS, TARGET_COL

router = APIRouter(prefix="/api/c2", tags=["C2 Root Cause Diagnosis"])
C2 = BY_NAME["c2"]          # the one C2 engine shared with the pipeline

DRIFT_HELP = {
    "none": "A clean batch from unseen test data: C2 should find nothing harmful.",
    "marginal_single": "Pick 1 feature: its values shift (numbers up, categories toward a rare value).",
    "marginal_multi": "Pick 2 to 5 features; each shifts.",
    "interaction": "Pick 2 features: their relationship breaks while each looks normal alone.",
    "combined": "Pick 3: the first shifts, the other two get an interaction break.",
}


class C2Run(BaseModel):
    drift_type: Literal["none", "marginal_single", "marginal_multi", "interaction", "combined"] = "marginal_single"
    features: List[str] = Field(default_factory=lambda: ["capital_gain"])
    magnitude: float = 0.6
    batch_size: int = 1000
    seed: Optional[int] = None          # default: a new seed per run, so every batch is different


def _c2():
    if not C2.loaded or C2.engine is None:
        raise HTTPException(503, f"C2 engine is not loaded on this machine: {C2.error or C2.engine_error}")
    return C2


@router.get("/info")
def info():
    c2 = _c2()
    a = c2.engine.a
    from c2.src.report import writer_name
    import os
    return {**c2.status(), "dataset": "uci", "label": DATASETS["uci"]["label"],
            "features": {"high": a["high_importance"], "low": a["low_importance"],
                         "all": a["feature_names"], "types": a["feature_types"]},
            "drift_types": [{"id": k, "min": v[0], "max": v[1], "help": DRIFT_HELP[k]}
                            for k, v in N_FEATURES.items()],
            "batch_sizes": [500, 1000, 2000],
            "c1_loaded": C1.loaded,
            "report_writer": writer_name() if os.environ.get("OPENAI_API_KEY") else "template",
            "recorded": RECORDED}


def _check(req, features):
    lo, hi = N_FEATURES[req.drift_type]
    if not lo <= len(req.features) <= hi:
        raise HTTPException(422, f"{req.drift_type} needs {lo if lo == hi else f'{lo} to {hi}'} "
                                 f"features, got {len(req.features)}")
    unknown = [f for f in req.features if f not in features]
    if unknown or len(set(req.features)) != len(req.features):
        raise HTTPException(422, f"unknown or repeated features: {unknown or req.features}")
    if req.drift_type != "none" and not 0.05 <= req.magnitude <= 1.0:
        raise HTTPException(422, "strength must be between 0.05 and 1.0")
    if not 200 <= req.batch_size <= 5000:
        raise HTTPException(422, "batch size must be between 200 and 5000")


@router.post("/run")
def run(req: C2Run):
    """One monitoring cycle on a new batch with a known drift (the answer key is kept for checking)."""
    c2 = _c2()
    from c2.src.scenarios import make_scenario
    _check(req, c2.engine.a["feature_names"])
    feats, pairs = list(req.features), None
    if req.drift_type == "combined":          # first feature shifts, the other two get an interaction break
        feats, pairs = [req.features[0]], [(req.features[1], req.features[2])]
    seed = req.seed if req.seed is not None else 7000 + len(c2.history.runs)
    s = make_scenario(req.drift_type, feats, 0.0 if req.drift_type == "none" else req.magnitude,
                      seed=seed, batch_size=req.batch_size, interaction_pairs=pairs)
    labelled, gt = s["drifted_df"], s["ground_truth"]
    batch = labelled.drop(columns=[TARGET_COL])

    c1, c1_note, c1_source = None, None, None
    uri = f"c2_page://run/{len(c2.history.runs) + 1}"
    if C1.loaded:
        try:
            ctx = {"dataset": "uci", "model_id": DATASETS["uci"]["model_id"], "current_dataset_uri": uri}
            c1, c1_source = C1.run(batch, {"context": ctx})["c1"], "real"
        except Exception as e:  # noqa: BLE001  fall back to the fake C1 below (NFR7)
            c1_note = f"C1 failed ({type(e).__name__}: {e}), so the team's fake C1 output was used"
    else:
        c1_note = "C1 engine not on this machine: C2 received the team's fake C1 output (built from the answer key)"
    if c1 is None:                            # the team's contract-valid stand-in for C1
        from contracts.c1_fake_output import build_fake_c1_output
        c1 = build_fake_c1_output(gt, seed=seed, model_id=DATASETS["uci"]["model_id"],
                                  current_dataset_uri=uri)
        c1_source = "fake"

    names = " + ".join(req.features) or "clean"
    kind = req.drift_type.replace("marginal_", "")
    source = f"Drift Lab · {names}" + ("" if req.drift_type == "none" else f" {kind} {req.magnitude:g}")
    truth = {"true_harm_pp": round(float(gt["actual_accuracy_drop"]), 2),
             "true_causes": s["true_causes"], "harm_metric": "accuracy_drop",
             "note": "same rows before vs after injection; the answer key never reaches C2"}
    injection = {"drift_type": req.drift_type, "features": req.features, "magnitude": req.magnitude,
                 "batch_size": len(batch), "seed": seed}
    # C2 gets the batch WITHOUT labels, like in production; the labels stay in the answer key
    result, rec = c2.diagnose(batch, c1, source=source, truth=truth,
                              injection=injection, c1_source=c1_source)
    errors = {k: contracts.validate(k, result[f"to_{k[-2:]}"]) for k in ("c2_to_c3", "c2_to_c4")}
    return {"run": rec, "contract_errors": errors, "c1_source": c1_source, "c1_note": c1_note}


@router.get("/runs")
def runs():
    c2 = _c2()
    return {"runs": c2.history.summaries(), "temporal_shap": c2.history.z_matrix()}


def _record(n):
    try:
        return _c2().history.get(n)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


@router.get("/runs/latest")
def run_latest():
    rec = _c2().history.latest()
    if rec is None:
        raise HTTPException(404, "no runs yet")
    return rec


@router.get("/runs/{n}")
def run_detail(n: int):
    return _record(n)


@router.get("/runs/{n}/report")
def run_report(n: int, refresh: bool = False):
    from c2.src.report import build_report
    c2, rec = _c2(), _record(n)
    if rec["report"] is None or refresh:
        c2.history.set_report(n, build_report(rec, c2.history.runs, c2.plain))
    return rec["report"]


@router.get("/runs/{n}/report.md")
def run_report_md(n: int):
    from c2.src.report import report_markdown
    report = run_report(n)
    return PlainTextResponse(report_markdown(report, _record(n)), media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="c2-report-run-{n}.md"'})


# ── monitoring: every batch ───────────────────────────────────────────────────
def _monitor_view(c2, limit=30):
    log = c2.monitor_log
    out = log.summary(last=max(5, min(limit, 60)))
    out.update({"batches": log.rows()[:200], "window": c2.engine.monitor.window,
                "persist_k": c2.engine.monitor.persist_k, "c1_loaded": C1.loaded,
                "z_line": round(float(c2.engine.monitor.z_thresh), 3), "last_check": c2.last_check,
                "warmup_batches": len(c2.engine.monitor.history) - len(log.records)})
    return out


@router.get("/monitor")
def monitor(limit: int = 30):
    """Every batch C1 checked and what C2's temporal SHAP monitor saw (newest first)."""
    c2 = _c2()
    try:
        c2.catch_up()
    except Exception as e:  # noqa: BLE001  show what is there; say why the rest is missing
        print(f"[c2] monitor catch-up failed: {e}", flush=True)
    return _monitor_view(c2, limit)


@router.post("/monitor/check")
def monitor_check(limit: int = 30):
    """Run the temporal SHAP check now: every saved C1 run (Drift Lab, C1 page, pipeline) that
    C2 has not monitored yet, oldest first."""
    c2 = _c2()
    try:
        c2.catch_up(limit=1000)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"monitoring check failed: {type(e).__name__}: {e}")
    return _monitor_view(c2, limit)


class LatestRows(BaseModel):
    rows: Literal[500, 1000, 2000, 5000] = 1000


@router.post("/monitor/check-latest")
def monitor_check_latest(req: LatestRows):
    """Manual check of the newest rows C1 has checked (pooled from the most recent saved C1 runs):
    C1's verdict when its engine is loaded, then C2's temporal SHAP pass."""
    c2 = _c2()
    try:
        rec, already = c2.check_latest(req.rows)
    except ValueError as e:
        raise HTTPException(409, str(e))
    view = _monitor_view(c2)
    view.update({"checked": {k: v for k, v in rec.items() if k != "shap"}, "already": already})
    return view


@router.post("/monitor/{batch}/diagnose")
def monitor_diagnose(batch: int):
    """Full diagnosis of a monitored batch. The batch is rebuilt from its saved run (Drift Lab
    recipe and seed, or the saved CSV); its monitoring pass is reused, so it is not counted twice."""
    c2 = _c2()
    try:
        mrec = c2.monitor_log.get(batch)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))
    if mrec.get("c2_run"):
        return {"run": c2.history.get(mrec["c2_run"]), "already": True, "contract_errors": {}}
    if mrec.get("batch_file"):                    # a manual check: the batch kept with the log
        import pandas as pd
        from pathlib import Path
        if not Path(mrec["batch_file"]).is_file():
            raise HTTPException(409, "the CSV of this manual check is no longer there")
        df = pd.read_csv(mrec["batch_file"])
        c1 = mrec.get("c1_output")
        result, run = c2.diagnose(df, c1, source=f"Monitoring · batch #{batch} · {mrec.get('source', '')}",
                                  truth=None, injection=None, c1_source="real" if c1 else None)
        errors = {k: contracts.validate(k, result[f"to_{k[-2:]}"]) for k in ("c2_to_c3", "c2_to_c4")}
        return {"run": run, "already": False, "contract_errors": errors}
    rec = platform_runs.get_run(mrec["run_ref"]) if mrec.get("run_ref") else None
    if rec is None:
        raise HTTPException(409, "this batch did not come from a saved run, so it cannot be rebuilt")
    df = c2.batch_for_run(rec)
    if df is None:
        raise HTTPException(409, "the batch of this run was not kept, so C2 cannot diagnose it")
    truth = None
    if rec.get("truth") and rec.get("injection"):
        inj, t = rec["injection"], rec["truth"]
        causes = list(inj.get("shifted") or [])
        for pair in inj.get("interaction_pairs") or []:
            causes += [f for f in pair if f not in causes]
        truth = {"true_harm_pp": round(float(t.get("true_harm_paired_pp", t.get("true_harm_pp", 0.0))), 2),
                 "true_causes": causes, "main_cause": t.get("main_cause"), "harm_metric": t.get("harm_metric"),
                 "note": "Drift Lab answer key; never shown to C2"}
    result, run = c2.diagnose(df, rec.get("c1"), source=f"Monitoring · batch #{batch}", truth=truth,
                              injection=rec.get("injection"), c1_source="real" if rec.get("c1") else None,
                              run_ref=rec["run_id"])
    errors = {k: contracts.validate(k, result[f"to_{k[-2:]}"]) for k in ("c2_to_c3", "c2_to_c4")}

    def change(r):                      # same shape the platform's drift alerts use
        r["c2"] = {"c2_to_c3": result["to_c3"], "c2_to_c4": result["to_c4"]}
        r["c2_review"] = {"status": "diagnosed", "at": platform_runs.now_iso(), "c2_run": run.get("run"),
                          "root": result["to_c3"]["root_cause_feature"],
                          "category": result["to_c3"]["root_cause_category"], "contract_errors": errors}
    try:
        platform_runs.update_run(rec["run_id"], change)
    except Exception as e:  # noqa: BLE001  the diagnosis itself is stored either way
        print(f"[c2] could not save the diagnosis with run {rec['run_id']}: {e}", flush=True)
    return {"run": run, "already": False, "contract_errors": errors}


@router.post("/reset")
def reset():
    c2 = _c2()
    c2.reset()
    return {"status": "reset", "warmup_sec": c2.warmup_sec}
