"""
Run history: every diagnosis run with its full temporal SHAP analysis.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

The C2 page shows the latest run and lets the user open any earlier one, so
each run is stored as one JSON line in c2/logs/runs.jsonl (git-ignored). A run
record keeps aggregates only, never data rows (NFR5):

    temporal_shap   z per feature (combined, distribution, behaviour channels),
                    accumulated EWMA z, alarm line, persistent features,
                    mean |SHAP| normal vs this run
    evidence        Drift Impact Score per feature, the four signals (E, D, P, G)
                    raw and normalised, estimated harm per feature
    localization    levels used, harmed rows estimate, precision / recall
                    estimates, share of the group in training data
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
    return {
        "timestamp": result["to_c4"]["diagnosis_timestamp"],
        "diagnosis_version": result["to_c4"]["diagnosis_version"],
        "source": source,
        "latency_s": None if latency_s is None else round(latency_s, 3),
        "n_rows": n_rows,
        "state": run_state(shap_r["alarm"], dis["is_malignant"]),
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
            "signals_raw": {f: _r({k: v for k, v in s.items() if k != "harm_pp"})
                            for f, s in raw.items()},
            "harm_pp_by_feature": {f: round(float(s["harm_pp"]), 2) for f, s in raw.items()},
            "weights": list(dis["weights"]),
            "estimated_harm_pp": dis["estimated_harm_pp"],
            "is_malignant": dis["is_malignant"],
        },
        "localization": {
            "levels_used": loc["levels_used"],
            "level2_rule": loc.get("level2_rule"),
            "harmed_rows_estimate": loc.get("harmed_rows_estimate"),
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


class RunHistory:

    def __init__(self, path=DEFAULT_HISTORY):
        self.path = path
        self.session = uuid.uuid4().hex[:8]
        self.runs = []
        if path is not None and path.exists():
            with open(path) as f:
                self.runs = [json.loads(line) for line in f if line.strip()]

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(exist_ok=True)
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
                "estimated_harm_pp": max(r["evidence"]["estimated_harm_pp"], 0.0),
                "diagnosis_stability": c3["diagnosis_stability"],
                "alarm": r["temporal_shap"]["alarm"],
                "has_report": r["report"] is not None,
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