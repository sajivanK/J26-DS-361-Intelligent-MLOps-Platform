"""
C2 XAI Root Cause Diagnosis (owner: Tharmika N, IT22247940). REAL component.

load()  loads the C2 engine (c2/src: temporal SHAP, label-free Drift Impact Score, rows-at-risk
        localization) for the UCI Adult model and warms it up on two clean batches from test.csv
        (labels dropped), the "first clean cycles after deployment": they set the temporal SHAP
        history and the clean baselines. Takes about 10 to 30 s at backend startup.
run()   one pipeline cycle on the batch exactly as the platform passes it: WITHOUT labels. C2 never
        needs labels: it finds the responsible feature, the share of decisions it changes and the
        group at risk from the model's predictions and the reference data. C1's output from the
        same run is checked against the C1 contract and used (hints, and its predicted harm as the
        accuracy estimate).

Fails safe: if the C2 engine cannot load (a missing library or data file on this machine), the
component still loads in fallback mode, says why in /api/health, shows as "fake", and answers the
pipeline with the contract example, so the team pipeline never stops because of C2.

C2 is stateful: each diagnosis is compared with the runs before it (temporal SHAP), so one engine
serves the pipeline and the C2 page, one request at a time (lock). Every diagnosis is stored in the
C2 run history (settings.C2_RUNS_FILE) with its full temporal SHAP analysis.

Monitoring (every batch): temporal SHAP is a monitor, so it sees EVERY batch C1 checks, not only
the ones C1 flags. Batches that reach C2 directly (pipeline, Diagnose) are observed on the way in;
the others (Drift Lab "Run C1 only", runs stopped by the repair gate, C1 page checks) are picked up
from the platform's saved runs in time order (catch_up). Each gets one light SHAP pass and a record
in the monitoring log (monitor.jsonl next to the run history). A later Diagnose of the same batch
reuses that pass, so no batch is counted twice. The monitor's memory survives a backend restart.
IEEE-CIS is not supported by C2 yet: for that dataset the contract-example fake is returned and
/api/health says so.
"""
import hashlib
import sys
import threading
import time
import traceback

from ..settings import C2_AUDIT_LOG, C2_RUNS_FILE, REPO_ROOT, TARGET_COL
from . import fakes
from .base import Component

sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_DATASETS = ("uci",)
C2_MONITOR_FILE = C2_RUNS_FILE.with_name("monitor.jsonl")     # monitoring log, next to the run history
WARMUP_BATCHES, WARMUP_ROWS = 2, 1000


class C2Component(Component):
    name = "c2"
    title = "C2 Root Cause Diagnosis"
    owner = "Tharmika N"
    produces = ("c2_to_c3", "c2_to_c4")
    source = "real"

    def __init__(self):
        super().__init__()
        self.engine = self.history = self.plain = self.monitor_log = None
        self.lock = threading.Lock()
        self.warmup_sec = None
        self.fake_runs = 0
        self.engine_error = None
        self.last_check = None

    def load(self):
        try:
            from c2.src.history import RunHistory
            from c2.src.monitor_log import MonitorLog
            from c2.src.outputs import C2DiagnosisEngine
            from c2.src.plain import PlainLanguage
            self.engine = C2DiagnosisEngine(log_path=C2_AUDIT_LOG)
            self.history = RunHistory(C2_RUNS_FILE)
            self.monitor_log = MonitorLog(C2_MONITOR_FILE)
            self.plain = PlainLanguage()
            self.warmup()
            self._restore_monitor()
            self.source, self.engine_error = "real", None
        except Exception as e:  # noqa: BLE001  fall back, never stop the team pipeline
            traceback.print_exc()
            self.engine = None
            self.source = "fake"
            self.engine_error = (f"{type(e).__name__}: {e}. Install C2's libraries with "
                                 "pip install -r c2/requirements.txt -c c1/requirements.txt and check "
                                 "the data files in shared/uci_adult (or C2_REFERENCE_CSV, C2_TEST_CSV, "
                                 "C2_MODEL_PKL)")
            print(f"[c2] engine not loaded, using the contract example: {self.engine_error}", flush=True)

    def warmup(self):
        """Clean labelled batches from test.csv: temporal SHAP history + performance noise band."""
        t0 = time.time()
        test = self.engine.a["test_df"]
        batches = [test.sample(WARMUP_ROWS, random_state=1000 + i) for i in range(WARMUP_BATCHES)]
        with self.lock:
            self.engine.warmup(batches)
        self.warmup_sec = round(time.time() - t0, 1)

    def _restore_monitor(self):
        """After a restart: give the temporal monitor back its memory of this period's batches,
        so persistence (2 of the last 3) continues where it stopped."""
        import numpy as np
        mon = self.engine.monitor
        with self.lock:
            for r in self.monitor_log.records:
                sh = r.get("shap")
                if not sh:
                    continue
                cycle = {"z": sh["cycle_z"], "significant": sh.get("significant_this_cycle", [])}
                z = np.array([cycle["z"][f] for f in mon.features])
                mon.ewma = z if mon.ewma is None else mon.alpha * z + (1 - mon.alpha) * mon.ewma
                mon.history.append(cycle)
                mon.top1_history.append(sh.get("root_cause_feature"))

    def reset(self):
        """Start a new monitoring period: empty run history and monitoring log, fresh warm-up."""
        from .. import runs as platform_runs
        try:
            old_runs = [r["run_id"] for r in platform_runs.list_runs()]
        except Exception:  # noqa: BLE001
            old_runs = []
        with self.lock:
            self.history.clear()
            self.monitor_log.clear(ignore_runs=old_runs)
        self.warmup()

    # ── monitoring: every batch ────────────────────────────────────────────────
    @staticmethod
    def batch_key(df):
        """Fingerprint of a batch (no data stored): the same rows give the same key."""
        import pandas as pd
        cols = sorted(c for c in df.columns if c != TARGET_COL)
        # normalised first, so the same rows give the same key whether they come from memory, a
        # rebuilt Drift Lab batch or a CSV file (int vs float columns, last-digit float noise)
        norm = pd.DataFrame({c: (pd.to_numeric(df[c], errors="coerce").astype("float64").round(6)
                                 if pd.api.types.is_numeric_dtype(df[c]) else df[c].astype(str))
                             for c in cols})
        h = pd.util.hash_pandas_object(norm, index=False).values
        return hashlib.sha1(h.tobytes()).hexdigest()[:16]

    def _hits(self):
        recent = self.engine.monitor.history[-3:]
        out = {}
        for c in recent:
            for f in c.get("significant", []):
                out[f] = out.get(f, 0) + 1
        return out

    def _observe_locked(self, df, key, c1=None, source="", run_ref=None, answer_key=None, extra=None):
        from c2.src.monitor_log import make_record
        shap_r = self.engine.observe(df)
        rec = make_record(shap_r, key, c1=c1, source=source, run_ref=run_ref, n_rows=len(df),
                          answer_key=answer_key, hits=self._hits(), extra=extra)
        return self.monitor_log.add(rec)

    def observe(self, batch, c1=None, source="", run_ref=None, answer_key=None, extra=None):
        """Monitoring pass for one batch (no diagnosis). Returns its monitoring record."""
        df = batch.drop(columns=[TARGET_COL]) if TARGET_COL in batch.columns else batch
        key = self.batch_key(df)
        with self.lock:
            rec = self.monitor_log.find(key)
            if rec is not None:
                self.monitor_log.add_run_ref(rec, run_ref)
                return rec
            return self._observe_locked(df, key, c1, source, run_ref, answer_key, extra)

    def check_latest(self, n_rows):
        """Manual check of the newest rows C1 has checked: the batches of the most recent saved C1
        runs, newest first, pooled until n_rows. C1 gives its verdict on the pooled batch when its
        engine is loaded; then C2's temporal SHAP pass. The batch is kept next to the monitoring log
        so Diagnose can rebuild it later."""
        import pandas as pd
        from .. import runs as platform_runs
        from ..settings import DATASETS
        from . import BY_NAME
        feats = self.engine.a["feature_names"]
        parts, used, have, skipped = [], [], 0, 0
        for srun in sorted(platform_runs.list_runs(), key=lambda x: x.get("saved_at") or "", reverse=True):
            if have >= n_rows:
                break
            if srun.get("dataset") not in SUPPORTED_DATASETS:
                continue
            rec = platform_runs.get_run(srun["run_id"])
            if not rec or not rec.get("c1"):
                continue
            try:
                df = self.batch_for_run(rec)
            except Exception:  # noqa: BLE001  e.g. a Drift Lab run while the C1 engine is not loaded
                df = None
            if df is None or any(f not in df.columns for f in feats):
                skipped += 1
                continue
            take = df[feats].tail(n_rows - have)
            parts.append(take)
            used.append(srun["run_id"])
            have += len(take)
        if have < 50:
            raise ValueError("fewer than 50 rows are available from saved C1 runs"
                             + (f" ({skipped} run(s) could not be rebuilt: start the backend with the C1 engine loaded)" if skipped else "")
                             + ". Run some batches in the Drift Lab or the C1 page first.")
        df = pd.concat(list(reversed(parts)), ignore_index=True)       # oldest rows first, like a real stream
        key = self.batch_key(df)
        with self.lock:
            old = self.monitor_log.find(key)
        if old is not None:
            return old, True
        c1, c1_note = None, None
        c1c = BY_NAME.get("c1")
        if c1c is not None and getattr(c1c, "loaded", False):
            try:
                ctx = {"dataset": "uci", "model_id": DATASETS["uci"]["model_id"],
                       "current_dataset_uri": f"c2_monitor://latest/{have}"}
                c1 = c1c.run(df, {"context": ctx})["c1"]
            except Exception as e:  # noqa: BLE001  monitor anyway, without C1's verdict
                c1_note = f"C1 failed: {type(e).__name__}: {e}"
        else:
            c1_note = "C1 engine not loaded, so this batch has no C1 verdict"
        folder = C2_MONITOR_FILE.parent / "monitor_batches"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.csv"
        df.to_csv(path, index=False)
        label = f"Latest {have:,} rows · {len(used)} C1 run{'s' if len(used) != 1 else ''}"
        rec = self.observe(df, c1, source=label, run_ref=None,
                           extra={"batch_file": str(path), "c1_output": c1, "c1_note": c1_note,
                                  "from_runs": used, "rows_requested": n_rows})
        return rec, False

    def batch_for_run(self, rec):
        """The batch of a saved platform run, without labels, or None if it was not kept:
        the saved CSV, or a Drift Lab batch rebuilt exactly from its recipe and seed."""
        import pandas as pd
        from pathlib import Path
        from .. import runs as platform_runs
        from ..lab import LAB
        from ..settings import LABEL_COLUMNS
        from . import BY_NAME
        path = platform_runs.batch_path(rec["run_id"])
        uri = str((rec.get("c1") or {}).get("current_dataset_uri") or "")
        if path is not None:
            df = pd.read_csv(path)
        elif uri.lower().endswith(".csv") and Path(uri).is_file():
            df = pd.read_csv(uri)            # the batch file C1 checked (Drift Lab keeps it for C3)
        elif rec.get("source") == "drift_lab" and rec.get("injection"):
            inj = rec["injection"]
            df, *_ = LAB.make_batch(BY_NAME["c1"].engine(rec["dataset"]), rec["dataset"], inj["drift_type"],
                                    inj["features"], inj["magnitude"], inj["batch_size"], inj["seed"])
        else:
            f = (rec.get("upload") or {}).get("file")
            if not f or not Path(f).is_file():
                return None
            df = pd.read_csv(f)
        return df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])

    @staticmethod
    def answer_key_of(rec):
        t, inj = rec.get("truth") or {}, rec.get("injection") or {}
        if not t:
            return None
        return {"main_cause": t.get("main_cause"), "true_harm_pp": t.get("true_harm_paired_pp", t.get("true_harm_pp")),
                "drift_type": inj.get("drift_type"), "features": inj.get("features")}

    SOURCE_NAMES = {"drift_lab": "Drift Lab", "c1_upload": "C1 page upload", "pipeline_upload": "Pipeline upload"}

    def source_name(self, rec):
        name = self.SOURCE_NAMES.get(rec.get("source"), rec.get("source") or "batch")
        mode = {"c1": "C1 only", "c1c2": "C1 → C2", "pipeline": "full pipeline"}.get(rec.get("mode"))
        return f"{name} · {mode}" if rec.get("source") == "drift_lab" and mode else name

    def catch_up(self, limit=200):
        """Observe, in time order, the saved platform runs of this period that C2 has not seen
        (e.g. Drift Lab "Run C1 only", or a pipeline stopped by the repair gate). A run whose
        batch was not kept is listed as unavailable so the log stays complete."""
        if self.engine is None:
            return 0
        from .. import runs as platform_runs
        seen = self.monitor_log.seen_run_refs()
        start = self.monitor_log.period_start
        todo = []
        for srun in platform_runs.list_runs():
            if srun.get("dataset") not in SUPPORTED_DATASETS or srun["run_id"] in seen:
                continue
            if srun["run_id"] in self.monitor_log.ignore_runs:
                continue                          # saved before the last Reset monitoring
            if start and not self.monitor_log.ignore_runs and _utc(srun.get("saved_at")) < start:
                continue
            todo.append(srun)
        done, waiting, gone, n0 = 0, 0, 0, len(self.monitor_log.records)
        for srun in sorted(todo, key=lambda x: x["saved_at"])[:limit]:
            rec = platform_runs.get_run(srun["run_id"])
            if not rec or not rec.get("c1"):
                continue
            try:
                df = self.batch_for_run(rec)
            except Exception as e:  # noqa: BLE001  e.g. C1 engine not loaded for a Drift Lab rebuild:
                # not recorded, so the next check tries again once the engine is there
                print(f"[c2] monitor: could not rebuild batch of run {srun['run_id']}: {e}", flush=True)
                waiting += 1
                continue
            if df is None:
                gone += 1
                with self.lock:
                    self.monitor_log.add({"record_version": 1, "timestamp": rec.get("saved_at"),
                                          "batch_key": None, "run_ref": rec["run_id"], "source": self.source_name(rec),
                                          "state": "unavailable", "c1": {"harm_severity": rec["c1"].get("harm_severity"),
                                                                         "drift_detected": rec["c1"].get("drift_detected")},
                                          "note": "batch not kept with this run, so C2 could not monitor it",
                                          "c2_run": None})
            else:
                self.observe(df, rec["c1"], source=self.source_name(rec), run_ref=rec["run_id"],
                             answer_key=self.answer_key_of(rec))
            done += 1
        self.last_check = {"at": _now_utc(), "new": len(self.monitor_log.records) - n0 - gone, "unavailable": gone, "waiting": waiting,
                           "saved_runs": len(todo)}
        return self.last_check

    def details(self):
        if self.engine is None:
            return {"engine_error": self.engine_error, "fallback": "contract example",
                    "fake_runs": self.fake_runs}
        return {"datasets": list(SUPPORTED_DATASETS), "warmup_sec": self.warmup_sec,
                "runs_stored": len(self.history.runs), "batches_monitored": len(self.monitor_log.records), "z_thresh": round(self.engine.monitor.z_thresh, 3),
                "plain_language_decoded": self.plain.decoded,
                "fake_runs_unsupported_dataset": self.fake_runs, "labels_used": False}

    def diagnose(self, batch, c1=None, source="", truth=None, injection=None, c1_source=None,
                 run_ref=None):
        """One label-free diagnosis, stored in the run history. Returns (engine result, record).
        c1_source: "real" (C1's predicted harm is used), "fake" (team fake C1: hints only,
        its harm copies the answer key) or None. If the monitor already saw this batch, its
        temporal SHAP pass is reused; otherwise the batch is observed now (so it joins the log)."""
        from c2.src.history import build_run_record
        df = batch.drop(columns=[TARGET_COL]) if TARGET_COL in batch.columns else batch
        try:
            self.catch_up()                       # earlier batches first, so time order holds
        except Exception as e:  # noqa: BLE001  monitoring must never block a diagnosis
            print(f"[c2] monitor catch-up skipped: {e}", flush=True)
        key = self.batch_key(df)
        with self.lock:
            t0 = time.time()
            mrec = self.monitor_log.find(key)
            if mrec is None:
                mrec = self._observe_locked(df, key, c1=c1, source=source, run_ref=run_ref)
            result = self.engine.diagnose(df, c1, c1_trusted=(c1_source == "real"), shap_result=mrec["shap"])
            rec = build_run_record(self.engine, result, source=source, latency_s=time.time() - t0,
                                   c1_output=c1, n_rows=len(df))
            rec["truth"], rec["injection"] = truth, injection
            rec["c1_source"] = c1_source if c1 else None      # "real", "fake" or None
            rec["c1_output"] = c1 or None                     # the exact C1 JSON C2 received
            c3 = result["to_c3"]
            rec["plain"] = {"root_cause": self.plain.feature(c3["root_cause_feature"]),
                            "group": self.plain.rule(c3["affected_slice_rule"]),
                            "decoded": self.plain.decoded}
            rec["batch_key"], rec["monitor_batch"] = key, mrec["batch"]
            rec = self.history.add(rec)
            self.monitor_log.set_c2_run(key, rec["run"], c3["root_cause_feature"])
        return result, rec

    def run(self, batch, upstream):
        ctx = upstream["context"]
        if self.engine is None or ctx.get("dataset") not in SUPPORTED_DATASETS:
            self.fake_runs += 1
            return fakes.fake_c2(upstream)
        origin = ctx.get("current_dataset_uri") or "pipeline run"
        # the saved platform run this batch belongs to (Drift Lab puts its id in the URI)
        run_ref = origin.rsplit("/", 1)[-1] if origin.startswith("drift_lab://") else ctx.get("run_id")
        result, _ = self.diagnose(batch, upstream.get("c1"), source=f"Pipeline · {origin}",
                                  c1_source="real" if upstream.get("c1") else None, run_ref=run_ref)
        return {"c2_to_c3": result["to_c3"], "c2_to_c4": result["to_c4"]}


def _now_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(iso):
    """Saved-run timestamps carry a local offset; compare in UTC with the period start."""
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""
