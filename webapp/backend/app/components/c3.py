"""
C3 Synthetic Data Repair (owner: Mathusan K, IT23174580).

CURRENTLY FAKE. To plug in the real C3:
  1. load your generators / prescription engine in load() (paths in app/settings.py, outside the repo)
  2. replace the body of run(); return {"c3": {...}}   (fields: shared/contracts/c3_output_schema.json)
  3. set source = "real"
Inputs available in run(): batch, upstream["c1"], upstream["c2_to_c3"], upstream["context"].
"""
from . import fakes
from .base import Component


class C3Component(Component):
    name = "c3"
    title = "C3 Synthetic Data Repair"
    owner = "Mathusan K"
    produces = ("c3",)
    source = "fake"

    def load(self):
        pass

    def run(self, batch, upstream):
        return fakes.fake_c3(upstream)
