"""
Saved runs: every Drift Lab run, every batch checked on the C1 page and every uploaded pipeline run is
saved as ONE JSON file, so it can be opened again later (Run history, run detail page).
J26-DS-361 | Intelligent MLOps Platform | Sajivan K (lead)

Where (outside the repo, settings.RUNS_DIR, default C:\\J26\\data\\platform_runs):
    runs/<run_id>.json        one file per run (the full record)
    batches/<run_id>.csv      the batch itself, only when "Save the batch CSV" was ticked (labels included)
    latest_pipeline_run.json  the latest full pipeline run (Overview page)
Saving is best-effort: a failure is printed and never breaks a run.

Outcome of a run (does C1's repair decision agree with the truth?), at the 1 pp severity threshold:
    correct alert   C1 says repair, truth >= 1 pp       false alert   C1 says repair, truth < 1 pp
    missed harm     C1 says no repair, truth >= 1 pp    correct quiet C1 says no repair, truth < 1 pp
    no truth        uploaded batch, no labels

Drift alerts for C2 (v0.3): a run is a drift alert when C1 says harm_severity is not "none". It is
"waiting" until C2 has diagnosed that batch (in the pipeline, or with Diagnose on the C2 page) or
someone dismisses it. Only runs saved from v0.3 on take part (older runs carry no alert state).
C2's answer is graded against the Drift Lab answer key (truth.main_cause) when there is one.
"""
import csv
import io
import json
import threading
import uuid
from datetime import datetime, timezone

from .settings import RUNS_DIR

SEVERITY_START_PP = 1.0        # harm_severity is "none" below 1 pp (C1 contract v1.1)
MIN_DETECTORS_FOR_ALARM = 3    # the "classic monitor" rule shown in the UI

_RUNS = RUNS_DIR / "runs"
_BATCHES = RUNS_DIR / "batches"
_LATEST = RUNS_DIR / "latest_pipeline_run.json"
_lock = threading.Lock()
_latest = None
_index = {}                    # run_id -> summary (built from the files on first use)
_index_loaded = False


def new_run_id():
    return uuid.uuid4().hex[:10]


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


# ── outcome and summary ───────────────────────────────────────────────────────
def outcome(truth_pp, severity):
    if truth_pp is None:
        return "no truth"
    harmful, alert = truth_pp >= SEVERITY_START_PP, severity != "none"
    if alert:
        return "correct alert" if harmful else "false alert"
    return "missed harm" if harmful else "correct quiet"


def c2_result(rec):
    """C2's diagnosis of this run, if any: from the pipeline (real C2 only) or from Diagnose later."""
    review = rec.get("c2_review") or {}
    if review.get("status") == "diagnosed":
        return {"root": review.get("root"), "category": review.get("category"), "source": "real"}
    pipe = rec.get("pipeline") or {}
    step = next((s for s in pipe.get("steps", []) if s.get("component") == "c2"), None)
    out = (pipe.get("outputs") or {}).get("c2_to_c3")
    if step and out and step.get("source") == "real" and step.get("status") == "ok":
        return {"root": out.get("root_cause_feature"), "category": out.get("root_cause_category"), "source": "real"}
    return None


def c2_grade(rec, c2):
    """True / False against the answer key, None when there is nothing to grade."""
    truth, inj = rec.get("truth") or {}, rec.get("injection") or {}
    if not c2 or not inj:
        return None
    if inj.get("drift_type") == "interaction":
        pair = (inj.get("interaction_pairs") or [[]])[0]
        return c2["category"] in ("interaction", "combined") and c2["root"] in pair
    main = truth.get("main_cause")
    return None if main is None else c2["root"] == main


def alert_state(rec, c2=None):
    if not rec.get("alerts_v"):
        return None
    from .orchestrator import repair_gate      # same rule as the pipeline (late import: no cycle)
    if not repair_gate(rec.get("c1") or {}):
        return None
    review = (rec.get("c2_review") or {}).get("status")
    if review in ("diagnosed", "dismissed"):
        return review
    return "diagnosed" if (c2 or c2_result(rec)) else "waiting"


def summarise(rec):
    c1 = rec.get("c1") or {}
    inj = rec.get("injection") or {}
    up = rec.get("upload") or {}
    truth = (rec.get("truth") or {}).get("true_harm_pp")
    lo, hi = [v * 100 for v in c1.get("harm_interval", [None, None])] if c1.get("harm_interval") else (None, None)
    pred = c1["predicted_harm"] * 100 if "predicted_harm" in c1 else None
    sev = c1.get("harm_severity", "none")
    return {
        "run_id": rec["run_id"], "saved_at": rec["saved_at"], "source": rec["source"], "mode": rec["mode"],
        "dataset": rec["dataset"], "drift_type": inj.get("drift_type"), "features": inj.get("features", []),
        "magnitude": inj.get("magnitude"), "seed": inj.get("seed"), "file": up.get("file"),
        "batch_size": c1.get("batch_size") or inj.get("batch_size") or up.get("rows"),
        "truth_pp": truth, "c1_pp": pred, "lo_pp": lo, "hi_pp": hi, "severity": sev,
        "extent": c1.get("marginal_extent"), "interaction": c1.get("interaction_suspected"),
        "detectors_fired": c1.get("detectors_fired", 0),
        "classic_alarm": c1.get("detectors_fired", 0) >= MIN_DETECTORS_FOR_ALARM,
        "inside": None if truth is None or lo is None else bool(lo <= truth <= hi),
        "outcome": outcome(truth, sev), "pipeline_status": (rec.get("pipeline") or {}).get("status"),
        "c3_status": ((rec.get("pipeline") or {}).get("outputs", {}).get("c3") or {}).get("execution_status"),
        "has_batch": bool(rec.get("batch_csv")),
        "true_cause": (rec.get("truth") or {}).get("main_cause"),
        "c2_root": (c2 := c2_result(rec)) and c2["root"],
        "c2_category": c2 and c2["category"],
        "c2_ok": c2_grade(rec, c2),
        "alert": alert_state(rec, c2),
    }


# ── saving and reading ────────────────────────────────────────────────────────
def _load_index():
    global _index_loaded
    if _index_loaded:
        return
    if _RUNS.exists():
        for f in _RUNS.glob("*.json"):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                _index[rec["run_id"]] = summarise(rec)
            except (OSError, ValueError, KeyError) as e:
                print(f"[runs] skipped {f.name}: {e}", flush=True)
    _index_loaded = True


def save_run(rec, batch_with_labels=None):
    """Save one run record (dict with run_id, source, mode, dataset, ...). Returns the summary."""
    rec.setdefault("run_id", new_run_id())
    rec.setdefault("saved_at", now_iso())
    rec.setdefault("alerts_v", 1)
    with _lock:
        _load_index()
        try:
            if batch_with_labels is not None:
                _BATCHES.mkdir(parents=True, exist_ok=True)
                batch_with_labels.to_csv(_BATCHES / f"{rec['run_id']}.csv", index=False)
                rec["batch_csv"] = f"{rec['run_id']}.csv"
            _RUNS.mkdir(parents=True, exist_ok=True)
            (_RUNS / f"{rec['run_id']}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
        except OSError as e:
            print(f"[runs] could not save run {rec['run_id']}: {e}", flush=True)
        summary = summarise(rec)
        _index[rec["run_id"]] = summary
    return summary


def list_runs():
    with _lock:
        _load_index()
        return sorted(_index.values(), key=lambda s: s["saved_at"], reverse=True)


def get_run(run_id):
    f = _RUNS / f"{run_id}.json"
    if not run_id.isalnum() or not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def batch_path(run_id):
    f = _BATCHES / f"{run_id}.csv"
    return f if run_id.isalnum() and f.exists() else None


EXPORT_COLUMNS = ["run_id", "saved_at", "source", "mode", "dataset", "drift_type", "features", "magnitude",
                  "seed", "file", "batch_size", "truth_pp", "c1_pp", "lo_pp", "hi_pp", "inside", "severity",
                  "outcome", "extent", "interaction", "detectors_fired", "classic_alarm",
                  "true_cause", "c2_root", "c2_category", "c2_ok", "alert"]


def update_run(run_id, change):
    """Apply change(rec) to a saved run and save it again. Returns the new summary, or None."""
    rec = get_run(run_id)
    if rec is None:
        return None
    change(rec)
    return save_run(rec)


def export_csv():
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for s in list_runs():
        w.writerow({**s, "features": " ".join(s["features"] or [])})
    return buf.getvalue()


# ── latest full pipeline run (Overview page) ──────────────────────────────────
def save_latest(run):
    global _latest
    with _lock:
        _latest = run
        try:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            _LATEST.write_text(json.dumps(run, default=str), encoding="utf-8")
        except OSError as e:
            print(f"[runs] could not save latest run: {e}", flush=True)


def latest():
    global _latest
    with _lock:
        if _latest is None and _LATEST.exists():
            try:
                _latest = json.loads(_LATEST.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _latest = None
        return _latest
