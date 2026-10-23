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
IEEE-CIS is not supported by C2 yet: for that dataset the contract-example fake is returned and
/api/health says so.
"""
import sys
import threading
import time
import traceback

from ..settings import C2_AUDIT_LOG, C2_RUNS_FILE, REPO_ROOT, TARGET_COL
from . import fakes
from .base import Component

sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_DATASETS = ("uci",)
WARMUP_BATCHES, WARMUP_ROWS = 2, 1000


class C2Component(Component):
    name = "c2"
    title = "C2 Root Cause Diagnosis"
    owner = "Tharmika N"
    produces = ("c2_to_c3", "c2_to_c4")
    source = "real"

    def __init__(self):
        super().__init__()
        self.engine = self.history = self.plain = None
        self.lock = threading.Lock()
        self.warmup_sec = None
        self.fake_runs = 0
        self.engine_error = None

    def load(self):
        try:
            from c2.src.history import RunHistory
            from c2.src.outputs import C2DiagnosisEngine
            from c2.src.plain import PlainLanguage
            self.engine = C2DiagnosisEngine(log_path=C2_AUDIT_LOG)
            self.history = RunHistory(C2_RUNS_FILE)
            self.plain = PlainLanguage()
            self.warmup()
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

    def reset(self):
        """Start a new monitoring period: empty run history, fresh warm-up."""
        with self.lock:
            self.history.clear()
        self.warmup()

    def details(self):
        if self.engine is None:
            return {"engine_error": self.engine_error, "fallback": "contract example",
                    "fake_runs": self.fake_runs}
        return {"datasets": list(SUPPORTED_DATASETS), "warmup_sec": self.warmup_sec,
                "runs_stored": len(self.history.runs), "z_thresh": round(self.engine.monitor.z_thresh, 3),
                "plain_language_decoded": self.plain.decoded,
                "fake_runs_unsupported_dataset": self.fake_runs, "labels_used": False}

    def diagnose(self, batch, c1=None, source="", truth=None, injection=None, c1_source=None):
        """One label-free diagnosis, stored in the run history. Returns (engine result, record).
        c1_source: "real" (C1's predicted harm is used), "fake" (team fake C1: hints only,
        its harm copies the answer key) or None."""
        from c2.src.history import build_run_record
        df = batch.drop(columns=[TARGET_COL]) if TARGET_COL in batch.columns else batch
        with self.lock:
            t0 = time.time()
            result = self.engine.diagnose(df, c1, c1_trusted=(c1_source == "real"))
            rec = build_run_record(self.engine, result, source=source, latency_s=time.time() - t0,
                                   c1_output=c1, n_rows=len(df))
            rec["truth"], rec["injection"] = truth, injection
            rec["c1_source"] = c1_source if c1 else None      # "real", "fake" or None
            rec["c1_output"] = c1 or None                     # the exact C1 JSON C2 received
            c3 = result["to_c3"]
            rec["plain"] = {"root_cause": self.plain.feature(c3["root_cause_feature"]),
                            "group": self.plain.rule(c3["affected_slice_rule"]),
                            "decoded": self.plain.decoded}
            rec = self.history.add(rec)
        return result, rec

    def run(self, batch, upstream):
        ctx = upstream["context"]
        if self.engine is None or ctx.get("dataset") not in SUPPORTED_DATASETS:
            self.fake_runs += 1
            return fakes.fake_c2(upstream)
        origin = ctx.get("current_dataset_uri") or "pipeline run"
        result, _ = self.diagnose(batch, upstream.get("c1"), source=f"Pipeline · {origin}",
                                  c1_source="real" if upstream.get("c1") else None)
        return {"c2_to_c3": result["to_c3"], "c2_to_c4": result["to_c4"]}
