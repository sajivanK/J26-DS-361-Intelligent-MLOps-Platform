"""
C1 detectors: six heterogeneous drift detectors, type-aware.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

run_all_detectors(profile, batch_df, seed) returns
    scores:      dict of the 9 detector signals (DETECTOR_SIGNALS)
    per_feature: dict {"ks": {...}, "chi2": {...}, "psi": {...}}
"""
from .ks_detector import ks_detector
from .chi2_detector import chi2_detector
from .psi_detector import psi_detector
from .mmd_detector import mmd_detector
from .lsdd_detector import lsdd_detector
from .clf_detector import clf_detector

DETECTOR_SIGNALS = ["ks_mean", "ks_max", "chi2_mean", "chi2_max",
                    "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]


def run_all_detectors(profile, batch_df, seed=0, use_lsdd=True):
    scores, per = {}, {}
    for name, (s, p) in (("ks", ks_detector(profile, batch_df)),
                         ("chi2", chi2_detector(profile, batch_df, seed=seed)),
                         ("psi", psi_detector(profile, batch_df))):
        scores.update(s)
        per[name] = p
    scores.update(mmd_detector(profile, batch_df, seed=seed))
    scores.update(lsdd_detector(profile, batch_df, seed=seed) if use_lsdd
                  else {"lsdd": float("nan")})
    scores.update(clf_detector(profile, batch_df, seed=seed))
    return scores, per
