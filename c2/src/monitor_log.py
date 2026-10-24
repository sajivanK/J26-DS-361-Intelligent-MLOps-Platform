"""
Monitoring log: one record per production batch that C2's temporal SHAP monitor saw.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Temporal SHAP (Novelty 1) is a monitor, so it looks at EVERY batch C1 checks, not only the
ones C1 flags. Each batch gets one light pass (SHAP on the monitoring sample) and a record:

    z per feature, the alarm line, which features are above it (this batch) and which are
    persistent (above it in at least 2 of the last 3 batches), the monitor state
    (normal / watch / alarm), the feature to watch, what C1 said about the same batch, and
    the C2 diagnosis run if one was made later.

The full temporal result is stored with the record, so a later Diagnose of the same batch
reuses it instead of counting the batch twice. Aggregates only, never data rows (NFR5).

Usage:
    log = MonitorLog(Path("c2/logs/monitor.jsonl"))
    rec = log.add(make_record(shap_result, key, c1=..., source=..., run_ref=...))
    log.summary()          # KPIs, watchlist, z matrix for the Monitoring tab
"""
import json
from datetime import datetime, timezone

import numpy as np

RECORD_VERSION = 1


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def monitor_state(shap_r):
    """alarm: a feature persists above the line; watch: above it in this batch only."""
    if shap_r.get("alarm"):
        return "alarm"
    return "watch" if shap_r.get("significant_this_cycle") else "normal"


def _round(d, n=3):
    return {k: round(float(v), n) for k, v in d.items()}


def _safe(o):
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def make_record(shap_r, key, c1=None, source="", run_ref=None, n_rows=None, answer_key=None,
                hits=None, extra=None):
    """One monitoring record. hits: {feature: batches above the line in the last 3}."""
    z = shap_r["cycle_z"]
    state = monitor_state(shap_r)
    persistent = list(shap_r.get("persistent_features") or [])
    significant = list(shap_r.get("significant_this_cycle") or [])
    if state == "alarm":
        feature = max(persistent, key=lambda f: z[f])
    elif state == "watch":
        feature = max(significant, key=lambda f: z[f])
    else:
        feature = max(z, key=z.get)
    c1 = c1 or None
    sev = (c1 or {}).get("harm_severity")
    return {
        "record_version": RECORD_VERSION,
        "timestamp": now_iso(),
        "batch_key": key,
        "run_ref": run_ref,
        "source": source,
        "n_rows": n_rows,
        "state": state,
        "feature": feature,
        "feature_z": round(float(z[feature]), 2),
        "feature_hits": int((hits or {}).get(feature, 1 if feature in significant else 0)),
        "z": _round(z, 2),
        "z_thresh": round(float(shap_r["z_thresh"]), 3),
        "significant": significant,
        "persistent": persistent,
        "c1": None if c1 is None else {
            "drift_detected": c1.get("drift_detected"), "harm_severity": sev,
            "predicted_harm": c1.get("predicted_harm"), "affected_features": c1.get("affected_features", []),
            "marginal_extent": c1.get("marginal_extent"), "interaction_suspected": c1.get("interaction_suspected")},
        # C2's monitor alarms, the feature is above the line in THIS batch, and C1 called the
        # batch harmless: e.g. interaction drift (not the tail of an episode C1 already flagged)
        "c1_missed": bool(state == "alarm" and feature in significant and c1 is not None
                          and sev in (None, "none")),
        "answer_key": answer_key,
        "c2_run": None,
        "shap": _safe(shap_r),
        **(extra or {}),                # e.g. batch_file and c1_output of a manual check
    }


class MonitorLog:

    def __init__(self, path):
        self.path = path
        self.state_path = None if path is None else path.with_name(path.stem + "_state.json")
        # period_start None = every saved C1 run counts (also runs from before C2 monitored them);
        # only Reset monitoring starts a new period, from that moment on
        self.records, self.period_start, self.ignore_runs = [], None, set()
        if path is not None and path.exists():
            with open(path) as f:
                self.records = [json.loads(line) for line in f if line.strip()]
        if self.state_path is not None and self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            if st.get("reset"):
                self.period_start = st.get("period_start")
                self.ignore_runs = set(st.get("ignore_runs") or [])

    def _save_state(self):
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"period_start": self.period_start,
                                               "reset": self.period_start is not None,
                                               "ignore_runs": sorted(self.ignore_runs)}))

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
        tmp.replace(self.path)

    def add(self, record):
        record = dict(record, batch=len(self.records) + 1)
        self.records.append(record)
        self._save()
        return record

    def find(self, key):
        return next((r for r in reversed(self.records) if r.get("batch_key") == key), None)

    def get(self, batch):
        if not 1 <= batch <= len(self.records):
            raise KeyError(f"batch {batch} not found (log has {len(self.records)} batches)")
        return self.records[batch - 1]

    def seen_run_refs(self):
        out = {r["run_ref"] for r in self.records if r.get("run_ref")}
        for r in self.records:
            out.update(r.get("also_runs") or [])
        return out

    def add_run_ref(self, rec, run_ref):
        """Another saved run with exactly the same batch: remember it, do not count the batch twice."""
        if run_ref and run_ref != rec.get("run_ref") and run_ref not in (rec.get("also_runs") or []):
            rec.setdefault("also_runs", []).append(run_ref)
            self._save()

    def set_c2_run(self, key, run, root=None):
        rec = self.find(key)
        if rec is not None:
            rec["c2_run"] = run
            rec["c2_root"] = root
            self._save()
        return rec

    def clear(self, ignore_runs=()):
        """New monitoring period. ignore_runs: the saved runs that exist now (they belong to the
        old period), so they are not monitored again even if saved in the same second."""
        self.records = []
        self.period_start = now_iso()
        self.ignore_runs = set(ignore_runs)
        self._save()
        self._save_state()

    # ── what the Monitoring tab shows ─────────────────────────────────────────
    def summary(self, last=30):
        recs = [r for r in self.records if r.get("state") != "unavailable"]
        states = [r["state"] for r in recs]
        kpis = {"total": len(recs), "normal": states.count("normal"), "watch": states.count("watch"),
                "alarm": states.count("alarm"), "c1_missed": sum(1 for r in recs if r.get("c1_missed")),
                "unavailable": len(self.records) - len(recs)}
        window = recs[-last:]
        feats = list(window[-1]["z"]) if window else []
        matrix = {"batches": [r["batch"] for r in window], "times": [r["timestamp"] for r in window],
                  "z_thresh": window[-1]["z_thresh"] if window else None,
                  "c1": [((r.get("c1") or {}).get("harm_severity") if r.get("c1") else None,
                          (r.get("c1") or {}).get("drift_detected")) for r in window],
                  "states": [r["state"] for r in window],
                  "missed": [bool(r.get("c1_missed")) for r in window],
                  "z": {f: [r["z"].get(f) for r in window] for f in feats}}
        watch = []
        if recs:
            last_rec, line = recs[-1], recs[-1]["z_thresh"]
            for f in feats:
                series = [r["z"].get(f) for r in recs[-8:]]
                n3 = sum(1 for r in recs[-3:] if r["z"].get(f, 0) > line)
                watch.append({"feature": f, "z": last_rec["z"].get(f), "trend": series, "last3": n3,
                              "state": "alarm" if n3 >= 2 else "watch" if n3 == 1 else "normal"})
            watch.sort(key=lambda w: -(w["z"] or 0))
        alarm = None
        missed = [r for r in recs[-5:] if r.get("c1_missed")]
        if missed:
            r = missed[-1]
            alarm = {"batch": r["batch"], "features": r["persistent"],
                     "z": {f: r["z"][f] for f in r["persistent"]},
                     "since": next((x["batch"] for x in recs if x.get("c1_missed")
                                    and set(x["persistent"]) & set(r["persistent"])), r["batch"]),
                     "count": sum(1 for x in recs if x.get("c1_missed")
                                  and set(x["persistent"]) & set(r["persistent"]))}
        return {"kpis": kpis, "matrix": matrix, "watchlist": watch, "c2_alarm": alarm,
                "period_start": self.period_start}

    def rows(self, newest_first=True):
        """Batch log rows (the stored temporal result is left out)."""
        out = [{k: v for k, v in r.items() if k != "shap"} for r in self.records]
        return list(reversed(out)) if newest_first else out
