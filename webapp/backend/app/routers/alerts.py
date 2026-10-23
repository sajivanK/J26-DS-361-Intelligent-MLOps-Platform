"""
Drift alerts for C2 (platform, owner: Sajivan K; requested by Tharmika N for the C2 page).

A drift alert is a saved run where C1 said harm_severity is not "none" and C2 has not diagnosed that
batch yet (Drift Lab "Run C1 only", the C1 page, uploads). Full pipeline and "Run C1 -> C2" runs are
diagnosed already, so they never wait. The sidebar shows the number waiting on the C2 tab.

    GET  /api/alerts                     {"waiting": n, "items": [...]}, newest first
    POST /api/alerts/{run_id}/diagnose   run C2 on the same batch now; the result is saved with the run
    POST /api/alerts/{run_id}/dismiss    stop counting it without a diagnosis

The batch for Diagnose comes from: the saved batch CSV, or (Drift Lab) the same settings and seed
rebuilt exactly, or (uploads) the copy of the uploaded file. C2 receives it WITHOUT labels.
"""
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from .. import contracts, runs
from ..components import BY_NAME, C1
from ..lab import LAB, LabError
from ..settings import LABEL_COLUMNS

router = APIRouter(prefix="/api/alerts", tags=["Drift alerts"])

try:
    from ..components.c2 import SUPPORTED_DATASETS as C2_DATASETS
except ImportError:  # pragma: no cover
    C2_DATASETS = ("uci",)


def _items():
    out = []
    for s in runs.list_runs():
        if s.get("alert") is None:
            continue
        item = dict(s)
        if item["alert"] == "waiting" and s["dataset"] not in C2_DATASETS:
            item["alert"] = "unsupported"
        out.append(item)
    return out


@router.get("")
def alerts():
    items = _items()
    return {"waiting": sum(1 for i in items if i["alert"] == "waiting"), "items": items,
            "c2_datasets": list(C2_DATASETS)}


def _batch(rec):
    """The batch of a saved run, without labels, or None if it cannot be recovered."""
    path = runs.batch_path(rec["run_id"])
    if path is not None:
        df = pd.read_csv(path)
    elif rec.get("source") == "drift_lab" and rec.get("injection"):
        inj = rec["injection"]
        try:
            eng = C1.engine(rec["dataset"])
            df, *_ = LAB.make_batch(eng, rec["dataset"], inj["drift_type"], inj["features"],
                                    inj["magnitude"], inj["batch_size"], inj["seed"])
        except (LabError, RuntimeError, KeyError) as e:
            raise HTTPException(409, f"could not rebuild the Drift Lab batch: {e}")
    else:
        f = (rec.get("upload") or {}).get("file")
        if not f or not Path(f).is_file():
            return None
        df = pd.read_csv(f)
    return df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])


def _c2_truth(rec):
    """The Drift Lab answer key in the shape the C2 page expects (C2's own scenario format):
    true_harm_pp = same rows before vs after (paired), true_causes = the injected features.
    main_cause (the feature that truly costs the most) is added for reference."""
    truth, inj = rec.get("truth"), rec.get("injection")
    if not truth or not inj:
        return None
    causes = list(inj.get("shifted") or [])
    for pair in inj.get("interaction_pairs") or []:
        causes += [f for f in pair if f not in causes]
    return {"true_harm_pp": round(float(truth.get("true_harm_paired_pp", truth.get("true_harm_pp", 0.0))), 2),
            "true_causes": causes, "main_cause": truth.get("main_cause"),
            "harm_metric": truth.get("harm_metric", "accuracy_drop"),
            "note": "Drift Lab answer key: same rows before vs after injection; never shown to C2"}


@router.post("/{run_id}/diagnose")
def diagnose(run_id: str):
    rec = runs.get_run(run_id)
    if rec is None:
        raise HTTPException(404, f"run '{run_id}' not found")
    if rec["dataset"] not in C2_DATASETS:
        raise HTTPException(422, f"C2 supports {', '.join(C2_DATASETS)} only")
    if not rec.get("c1"):
        raise HTTPException(422, "this run has no C1 output to diagnose")
    c2 = BY_NAME["c2"]
    if not c2.loaded or getattr(c2, "engine", None) is None:
        raise HTTPException(503, f"C2 is not loaded on this machine: {c2.error or getattr(c2, 'engine_error', '')}")
    batch = _batch(rec)
    if batch is None:
        raise HTTPException(409, "the batch of this run was not saved, so C2 cannot diagnose it")
    result, hist = c2.diagnose(batch, rec["c1"], source=f"Drift alert · run {run_id}",
                               truth=_c2_truth(rec), injection=rec.get("injection"), c1_source="real")
    errors = {"c2_to_c3": contracts.validate("c2_to_c3", result["to_c3"]),
              "c2_to_c4": contracts.validate("c2_to_c4", result["to_c4"])}

    def change(r):
        r["c2"] = {"c2_to_c3": result["to_c3"], "c2_to_c4": result["to_c4"]}
        r["c2_review"] = {"status": "diagnosed", "at": runs.now_iso(), "c2_run": hist.get("run"),
                          "root": result["to_c3"]["root_cause_feature"],
                          "category": result["to_c3"]["root_cause_category"], "contract_errors": errors}
    summary = runs.update_run(run_id, change)
    return {"summary": summary, "c2_run": hist.get("run"), "contract_errors": errors,
            "c2_to_c3": result["to_c3"], "c2_to_c4": result["to_c4"]}


@router.post("/{run_id}/dismiss")
def dismiss(run_id: str):
    summary = runs.update_run(run_id, lambda r: r.update(c2_review={"status": "dismissed", "at": runs.now_iso()}))
    if summary is None:
        raise HTTPException(404, f"run '{run_id}' not found")
    return {"summary": summary}
