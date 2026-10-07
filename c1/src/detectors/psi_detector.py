"""PSI detector: Population Stability Index per feature (all features).

Continuous: bins are reference quantiles (fixed at onboarding).
Categorical: one bin per reference category plus one for unseen categories.
Empty bins are floored at PSI_EPS so the log stays finite.
"""
import numpy as np
from ..profile import _bin_props, _cat_props, PSI_EPS
from ._common import aggregate


def _psi(ref_p, cur_p):
    r = np.clip(ref_p, PSI_EPS, None)
    c = np.clip(cur_p, PSI_EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_detector(profile, batch):
    per = {}
    for f in profile.continuous:
        per[f] = _psi(profile.psi_ref_props[f],
                      _bin_props(batch[f].values, profile.psi_edges[f]))
    for f in profile.categorical:
        per[f] = _psi(profile.psi_ref_props[f],
                      _cat_props(batch[f].values, profile.categories[f]))
    return aggregate(per, "psi"), per
