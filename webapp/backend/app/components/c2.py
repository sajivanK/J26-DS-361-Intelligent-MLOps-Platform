"""
C2 XAI Root Cause Diagnosis (owner: Tharmika N, IT22247940). REAL component.

load()  loads the C2 engine (c2/src: temporal SHAP, Drift Impact Score, hierarchical localization)
        for the UCI Adult model and warms it up on two clean labelled batches from test.csv, the
        "first clean cycles after deployment": they set the temporal SHAP history and the
        performance noise band. Takes about 10 to 30 s at backend startup.
run()   one pipeline cycle. C2 needs the batch's labels (the accuracy impact, the harm estimate
        and the harmed group are measured with them). The platform removes labels before
        monitoring, so until delayed labels are passed as upstream["labels"] (a list or Series in
        the batch's row order) the pipeline step returns the contract example, and /api/health
        counts it. The C2 page (routers/c2.py) builds its own labelled Drift Lab batches and runs
        the real diagnosis.

C2 is stateful: each diagnosis is compared with the runs before it (temporal SHAP), so one engine
serves the pipeline and the C2 page, one request at a time (lock). Every diagnosis is stored in the
C2 run history (settings.C2_RUNS_FILE) with its full temporal SHAP analysis.
IEEE-CIS is not supported by C2 yet: for that dataset the contract-example fake is returned and
/api/health says so.
"""
import sys
import threading
import time

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
        self.unlabelled_runs = 0

    def load(self):
        from c2.src.history import RunHistory
        from c2.src.outputs import C2DiagnosisEngine
        from c2.src.plain import PlainLanguage
        self.engine = C2DiagnosisEngine(log_path=C2_AUDIT_LOG)
        self.history = RunHistory(C2_RUNS_FILE)
        self.plain = PlainLanguage()
        self.warmup()

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
            return {}
        return {"datasets": list(SUPPORTED_DATASETS), "warmup_sec": self.warmup_sec,
                "runs_stored": len(self.history.runs), "z_thresh": round(self.engine.monitor.z_thresh, 3),
                "plain_language_decoded": self.plain.decoded,
                "fake_runs_unsupported_dataset": self.fake_runs,
                "pipeline_runs_without_labels": self.unlabelled_runs}

    def diagnose(self, batch, c1=None, labels=None, source="", truth=None, injection=None):
        """One diagnosis, stored in the run history. Returns (engine result, history record)."""
        from c2.src.history import build_run_record
        df = batch.copy()
        if labels is not None:
            df[TARGET_COL] = list(labels)
        with self.lock:
            t0 = time.time()
            result = self.engine.diagnose(df, c1)
            rec = build_run_record(self.engine, result, source=source, latency_s=time.time() - t0,
                                   c1_output=c1, n_rows=len(df))
            rec["truth"], rec["injection"] = truth, injection
            c3 = result["to_c3"]
            rec["plain"] = {"root_cause": self.plain.feature(c3["root_cause_feature"]),
                            "group": self.plain.rule(c3["affected_slice_rule"]),
                            "decoded": self.plain.decoded}
            rec = self.history.add(rec)
        return result, rec

    def run(self, batch, upstream):
        ctx = upstream["context"]
        if ctx.get("dataset") not in SUPPORTED_DATASETS:
            self.fake_runs += 1
            return fakes.fake_c2(upstream)
        labels = upstream.get("labels")
        if labels is None:                   # labels removed by the platform: not diagnosable yet
            self.unlabelled_runs += 1
            return fakes.fake_c2(upstream)
        origin = ctx.get("current_dataset_uri") or "pipeline run"
        result, _ = self.diagnose(batch, upstream.get("c1"), labels, source=f"Pipeline · {origin}")
        return {"c2_to_c3": result["to_c3"], "c2_to_c4": result["to_c4"]}
