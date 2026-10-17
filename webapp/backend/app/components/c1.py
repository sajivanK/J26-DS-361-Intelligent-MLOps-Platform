"""
C1 Drift Detection Engine (owner: Sajivan K, IT23172296). REAL component.

Loads the engine 1.2.3 for each dataset at startup and warms it up (the first LSDD call builds its
permutation test), so each request then takes about 0.2 s (UCI) or 1.6 s (IEEE-CIS) on the laptop.
detect() is not written for concurrent use, so calls to the same engine are serialised with a lock.
"""
import sys
import threading
import time

from ..settings import C1_ENGINES, DATASETS, REPO_ROOT
from .base import Component

sys.path.insert(0, str(REPO_ROOT))
from c1.src.engine import ENGINE_VERSION  # noqa: E402
from c1.src.engine_io import load_engine  # noqa: E402


class C1Component(Component):
    name = "c1"
    title = "C1 Drift Detection"
    owner = "Sajivan K"
    produces = ("c1",)
    source = "real"

    def __init__(self):
        super().__init__()
        self.engines, self.engine_status = {}, {}
        self._locks = {k: threading.Lock() for k in C1_ENGINES}

    def load(self):
        for key, path in C1_ENGINES.items():
            t0 = time.time()
            try:
                eng = load_engine(path)
                t1 = time.time()
                n = min(eng.sizes[0], len(eng.profile.ref))
                eng.detect(eng.profile.ref.sample(n, random_state=1), seed=1)   # warm-up
                self.engines[key] = eng
                self.engine_status[key] = {"loaded": True, "path": path, "load_sec": round(t1 - t0, 1),
                                           "warmup_sec": round(time.time() - t1, 1)}
            except Exception as e:  # noqa: BLE001  report per dataset, keep the other engine
                self.engine_status[key] = {"loaded": False, "path": path, "error": f"{type(e).__name__}: {e}"}
            print(f"[c1] engine {key}: {self.engine_status[key]}", flush=True)
        if not self.engines:
            raise RuntimeError("no C1 engine could be loaded")

    def details(self):
        return {"engine_version": ENGINE_VERSION, "engines": self.engine_status}

    def engine(self, dataset):
        if dataset not in C1_ENGINES:
            raise KeyError(f"unknown dataset '{dataset}'; use one of {list(C1_ENGINES)}")
        if dataset not in self.engines:
            raise RuntimeError(f"C1 engine '{dataset}' is not loaded: "
                               f"{self.engine_status.get(dataset, {}).get('error', 'not loaded')}")
        return self.engines[dataset]

    def info(self, dataset):
        eng = self.engine(dataset)
        return {"dataset": dataset, "label": DATASETS[dataset]["label"], "model_id": DATASETS[dataset]["model_id"],
                "engine_version": ENGINE_VERSION, "min_rows": 100, **eng.summary()}

    def run(self, batch, upstream):
        ctx = upstream["context"]
        eng = self.engine(ctx["dataset"])
        with self._locks[ctx["dataset"]]:
            out = eng.detect(batch, model_id=ctx["model_id"],
                             current_dataset_uri=ctx.get("current_dataset_uri", ""), seed=0)
        return {"c1": out}
