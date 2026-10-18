"""
The one interface every component follows.
J26-DS-361 | Intelligent MLOps Platform

    load()                      load models once, at backend startup
    run(batch, upstream) -> {}  batch: the production batch (pandas DataFrame, labels removed)
                                upstream: everything produced before you, keyed by output name:
                                  upstream["context"]   {"dataset", "model_id", "current_dataset_uri", "run_id"}
                                  upstream["c1"]        C1 output (contract v1.1)
                                  upstream["c2_to_c3"], upstream["c2_to_c4"], upstream["c3"]
                                returns {output_name: output_dict}, one entry per contract you produce
                                (names in app/contracts.py CONTRACT_FILES)

Set `source = "real"` once your component runs your actual code instead of the fake.
"""
import time
import traceback


class Component:
    name = "base"                 # c1, c2, c3, c4
    title = ""
    owner = ""
    produces = ()                 # contract output names this component returns
    source = "fake"               # "fake" or "real"

    def __init__(self):
        self.loaded = False
        self.error = None
        self.load_sec = None

    # override these two ---------------------------------------------------
    def load(self):
        """Load models and data. Raise an exception if something is missing."""

    def run(self, batch, upstream):
        raise NotImplementedError

    # optional: extra details for /api/health ---------------------------------
    def details(self):
        return {}

    # used by the platform; no need to override --------------------------------
    def safe_load(self):
        t0 = time.time()
        try:
            self.load()
            self.loaded, self.error = True, None
        except Exception as e:  # noqa: BLE001  one broken component must not stop the backend
            self.loaded, self.error = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        self.load_sec = round(time.time() - t0, 1)
        state = "loaded" if self.loaded else f"FAILED ({self.error})"
        print(f"[{self.name}] {self.title}: {state} in {self.load_sec} s, source={self.source}", flush=True)

    def status(self):
        return {"name": self.name, "title": self.title, "owner": self.owner, "source": self.source,
                "loaded": self.loaded, "error": self.error, "load_sec": self.load_sec,
                "produces": list(self.produces), **self.details()}
