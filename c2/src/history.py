"""
Run history: every diagnosis run with its full temporal SHAP analysis.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

The C2 page shows the latest run and lets the user open any earlier one, so
each run is stored as one JSON line in c2/logs/runs.jsonl (git-ignored). A run
record keeps aggregates only, never data rows (NFR5):

    temporal_shap   z per feature (combined, distribution, behaviour channels),
                    accumulated EWMA z, alarm line, persistent features,
                    mean |SHAP| normal vs this run
    evidence        label-free Drift Impact Score per feature, the four signals
                    (E explanation, D drift, DI decision change, PS prediction shift)
                    raw and normalised, the share of decisions each feature changes,
                    C1's estimated accuracy drop when C1 is real
    localization    levels used, rows at risk (decisions the drift changes),
                    risk-weighted precision / recall, share of the group in training data
    outputs         the exact to_c3 / to_c4 messages and warnings
    report          the business report, cached after it is first generated

The audit log (FR9) is separate: it is the compliance trail, this is what the
page and the report read.

Usage:
    hist = RunHistory()
    rec = hist.add(build_run_record(engine, result, source="Drift Lab", latency_s=1.1))
    hist.summaries(); hist.get(3); hist.latest()
"""
import json
import uuid

import numpy as np

from c2.src.config import C2_DIR

DEFAULT_HISTORY = C2_DIR / "logs" / "runs.jsonl"
MALIGNANT_PP = 2.0
RECORD_VERSION = 2          # 2 = label-free records; version 1 (labelled) runs are set aside
_RAW_KEYS = ("E", "D", "DI", "PS", "decision_change", "flip_share")


def run_state(alarm, malignant):
    """clean / harmless / suspected / confirmed, as shown on the page."""
    if alarm:
        return "confirmed" if malignant else "harmless"
    return "suspected" if malignant else "clean"


def _r(d, n=4):
    return {k: round(float(v), n) for k, v in d.items()}


def build_run_record(engine, result, source="", latency_s=None, c1_output=None,
                     n_rows=None):
    """One history record from C2DiagnosisEngine.diagnose() and its last_evidence."""
    ev = engine.last_evidence
    shap_r, dis, loc = ev["shap"], ev["dis"], ev["localization"]
    to_c3 = result["to_c3"]
    rule = to_c3["affected_slice_rule"]
    ref = engine.a["ref_df"]
    try:
        ref_share = float(len(ref.query(rule)) / len(ref))
    except Exception:  # noqa: BLE001
        ref_share = None
    c1 = c1_output or {}
    raw = dis["signals_raw"]
    d = result["details"]
    return {
        "record_version": RECORD_VERSION,
        "timestamp": result["to_c4"]["diagnosis_timestamp"],
        "diagnosis_version": result["to_c4"]["diagnosis_version"],
        "source": source,
        "latency_s": None if latency_s is None else round(latency_s, 3),
        "n_rows": n_rows,
        "state": run_state(shap_r["alarm"], d["is_malignant"]),
        "c1": {"status": result["details"]["c1_status"],
               "harm_severity": c1.get("harm_severity"),
               "predicted_harm": c1.get("predicted_harm"),
               "marginal_extent": c1.get("marginal_extent"),
               "interaction_suspected": c1.get("interaction_suspected")},
        "temporal_shap": {
            "z_thresh": round(float(shap_r["z_thresh"]), 3),
            "z": _r(shap_r["cycle_z"]),
            "z_distribution": _r(shap_r["z_distribution"]),
            "z_behaviour": _r(shap_r["z_behaviour"]),
            "ewma": _r(shap_r["ranking"]),
            "alarm": bool(shap_r["alarm"]),
            "persistent_features": list(shap_r["persistent_features"]),
            "shift_pattern": shap_r["shift_pattern"],
            "mean_abs_shap_ref": _r(shap_r["mean_abs_shap_ref"]),
            "mean_abs_shap_now": _r(shap_r["mean_abs_shap_current"]),
        },
        "evidence": {
            "drift_impact_scores": dis["drift_impact_scores"],
            "signals": {f: _r(s) for f, s in dis["signals_normalised"].items()},
            "signals_raw": {f: _r({k: v for k, v in s.items() if k in _RAW_KEYS and v is not None})
                            for f, s in raw.items()},
            "decision_change_by_feature": {f: round(max(float(s["decision_change"]), 0.0), 4)
                                           for f, s in raw.items()},
            "weights": list(dis["weights"]),
            "labels_used": False,
            "decision_change": d["decision_change"],
            "decision_change_part": d["decision_change_part"],
            "c1_estimated_harm_pp": d["c1_estimated_harm_pp"],
            "estimated_harm_pp": d["estimated_harm_pp"],
            "is_malignant": d["is_malignant"],
            "harm_basis": d["harm_basis"],
        },
        "localization": {
            "levels_used": loc["levels_used"],
            "level2_rule": loc.get("level2_rule"),
            "rows_at_risk": loc.get("rows_at_risk"),
            "est_precision": loc.get("est_precision"),
            "est_recall": loc.get("est_recall"),
            "ref_share": None if ref_share is None else round(ref_share, 4),
            "note": loc.get("note", ""),
        },
        "to_c3": to_c3,
        "to_c4": result["to_c4"],
        "warnings": result["details"]["warnings"],
        "report": None,
    }


def decision_points(record):
    """Share of the model's decisions that change because of the root feature, in %."""
    return round(100 * float(record["evidence"]["decision_change"]), 1)


def harm_points(record):
    """Estimated accuracy loss in pp, never below 0: the harm predictor's estimate for the
    root feature, else real C1's batch estimate. None when neither exists (no labels, so
    accuracy is never measured)."""
    ev = record["evidence"]
    v = ev.get("estimated_harm_pp")
    v = ev.get("c1_estimated_harm_pp") if v is None else v
    return None if v is None else round(max(float(v), 0.0), 2)


def c1_points(record):
    """C1's predicted harm in pp (contract v1.1: a proportion). None without C1, and None for the
    team's fake C1: it copies the true drop from the answer key, so it is not a prediction."""
    if record.get("c1_source") == "fake":
        return None
    p = (record.get("c1") or {}).get("predicted_harm")
    return None if p is None else round(100 * float(p), 2)


class RunHistory:

    def __init__(self, path=DEFAULT_HISTORY):
        self.path = path
        self.session = uuid.uuid4().hex[:8]
        self.runs = []
        if path is not None and path.exists():
            with open(path) as f:
                runs = [json.loads(line) for line in f if line.strip()]
            old = [r for r in runs if r.get("record_version") != RECORD_VERSION]
            if old:                     # labelled-era runs: kept aside, not mixed in
                backup = path.with_name(path.stem + "_v1_backup.jsonl")
                with open(backup, "a") as f:
                    for r in old:
                        f.write(json.dumps(r) + "\n")
                runs = [r for r in runs if r.get("record_version") == RECORD_VERSION]
                print(f"[c2] {len(old)} older (labelled) runs moved to {backup.name}")
            self.runs = runs
            if old:
                self._save()

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in self.runs:
                f.write(json.dumps(r, default=_np_default) + "\n")
        tmp.replace(self.path)

    def add(self, record):
        record = dict(record, run=len(self.runs) + 1, session=self.session)
        self.runs.append(record)
        self._save()
        return record

    def get(self, run):
        if not 1 <= run <= len(self.runs):
            raise KeyError(f"run {run} not found (history has {len(self.runs)} runs)")
        return self.runs[run - 1]

    def latest(self):
        return self.runs[-1] if self.runs else None

    def set_report(self, run, report):
        self.get(run)["report"] = report
        self._save()

    def clear(self):
        self.runs = []
        self.session = uuid.uuid4().hex[:8]
        self._save()

    def summaries(self):
        """One short row per run for the history table (newest first)."""
        out = []
        for r in reversed(self.runs):
            c3 = r["to_c3"]
            out.append({
                "run": r["run"], "timestamp": r["timestamp"], "source": r["source"],
                "state": r["state"],
                "root_cause_feature": None if r["state"] == "clean" else c3["root_cause_feature"],
                "decision_change_pct": decision_points(r),
                "estimated_harm_pp": harm_points(r),
                "c1_predicted_pp": c1_points(r),
                "diagnosis_stability": c3["diagnosis_stability"],
                "alarm": r["temporal_shap"]["alarm"],
                "has_report": r["report"] is not None,
                "truth_pp": (r.get("truth") or {}).get("true_harm_pp"),   # Drift Lab answer key
            })
        return out

    def z_matrix(self):
        """Temporal SHAP across all stored runs: {runs: [...], z: {feature: [...]}}."""
        if not self.runs:
            return {"runs": [], "z_thresh": None, "z": {}}
        feats = list(self.runs[-1]["temporal_shap"]["z"])
        return {"runs": [r["run"] for r in self.runs],
                "times": [r["timestamp"] for r in self.runs],
                "z_thresh": self.runs[-1]["temporal_shap"]["z_thresh"],
                "z": {f: [r["temporal_shap"]["z"].get(f) for r in self.runs] for f in feats}}


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")