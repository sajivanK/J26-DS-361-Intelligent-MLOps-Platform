"""
C4 Release Controller (owner: Kesigan M, IT23210974).

CURRENTLY FAKE. To plug in the real C4:
  1. load your forecaster / conformal config in load() (paths in app/settings.py, outside the repo)
  2. replace the body of run(); return {"c4": {...}}   (fields: shared/contracts/c4_output_schema.json)
  3. set source = "real"
Inputs available in run(): batch, upstream["c1"], upstream["c2_to_c4"], upstream["c3"], upstream["context"].
"""
from . import fakes
from .base import Component


class C4Component(Component):
    name = "c4"
    title = "C4 Release Controller"
    owner = "Kesigan M"
    produces = ("c4",)
    source = "fake"

    def load(self):
        pass

    def run(self, batch, upstream):
        return fakes.fake_c4(upstream)
