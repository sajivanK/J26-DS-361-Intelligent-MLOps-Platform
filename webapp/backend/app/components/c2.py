"""
C2 XAI Root Cause Diagnosis (owner: Tharmika N, IT22247940).

CURRENTLY FAKE. To plug in the real C2:
  1. load your models in load() (paths in app/settings.py, files outside the repo)
  2. replace the body of run() with your diagnosis; return BOTH outputs:
       {"c2_to_c3": {...}, "c2_to_c4": {...}}   (fields: shared/contracts/c2_output_to_c3/_c4_schema.json)
  3. set source = "real"
Inputs available in run(): batch (DataFrame, no labels), upstream["c1"] (C1 contract v1.1),
upstream["context"]["dataset"] ("uci" / "ieee").
"""
from . import fakes
from .base import Component


class C2Component(Component):
    name = "c2"
    title = "C2 Root Cause Diagnosis"
    owner = "Tharmika N"
    produces = ("c2_to_c3", "c2_to_c4")
    source = "fake"

    def load(self):
        pass

    def run(self, batch, upstream):
        return fakes.fake_c2(upstream)
