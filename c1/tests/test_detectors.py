"""
Unit tests for C1 detectors and prediction stats (synthetic data).
Run from the repo root:  python -m pytest c1/tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from c1.src.profile import ReferenceProfile                      # noqa: E402
from c1.src.detectors import run_all_detectors, DETECTOR_SIGNALS  # noqa: E402
from c1.src.detectors.ks_detector import ks_detector               # noqa: E402
from c1.src.detectors.chi2_detector import chi2_detector           # noqa: E402
from c1.src.detectors.psi_detector import psi_detector             # noqa: E402
from c1.src.detectors.mmd_detector import mmd_detector             # noqa: E402
from c1.src.detectors.clf_detector import clf_detector             # noqa: E402
from c1.src.prediction_stats import prediction_stats               # noqa: E402
from c1.tests.synthetic import make_frame, CONT, CAT               # noqa: E402

FT = {**{c: "continuous" for c in CONT}, **{c: "categorical" for c in CAT}}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    full = make_frame(9000, rng)
    ref = full.iloc[:6000].reset_index(drop=True)
    cur = full.iloc[6000:].reset_index(drop=True).drop(columns="target")
    return ref, cur, ReferenceProfile(ref, FT)


def test_profile_types_from_json_not_guessed(data):
    ref, _, prof = data
    assert prof.continuous == CONT
    assert prof.categorical == list(CAT)
    enc = prof.encode(ref.head(5))
    assert enc.shape == (5, len(CONT) + sum(len(v) for v in prof.categories.values()))


def test_profile_rejects_missing_type(data):
    ref, _, _ = data
    bad = dict(FT)
    bad.pop("race")
    with pytest.raises(ValueError):
        ReferenceProfile(ref, bad)


def test_point_mass_feature_keeps_its_own_psi_bin(data):
    _, _, prof = data
    # capital_gain is ~92% one value: the pile must be bin 0, not merged
    assert prof.psi_ref_props["capital_gain"][0] > 0.85


def test_no_drift_scores_small(data):
    _, cur, prof = data
    s, per = run_all_detectors(prof, cur, seed=1, use_lsdd=False)
    assert set(DETECTOR_SIGNALS) <= set(s)
    assert s["ks_max"] < 0.06
    assert s["chi2_max"] < 0.06
    assert s["psi_max"] < 0.05
    assert 0.4 < s["clf_auc"] < 0.6
    assert set(per["ks"]) == set(CONT) and set(per["chi2"]) == set(CAT)


def test_continuous_shift_hits_that_feature(data):
    _, cur, prof = data
    d = cur.copy()
    d["age"] = d["age"] + 0.8
    s, per = run_all_detectors(prof, d, seed=1, use_lsdd=False)
    assert max(per["ks"], key=per["ks"].get) == "age"
    assert s["ks_max"] > 0.25
    # the max is not diluted the way the mean is
    assert s["ks_max"] > 2 * s["ks_mean"]
    assert s["clf_auc"] > 0.6


def test_categorical_shift_seen_by_chi2_not_ks(data):
    _, cur, prof = data
    d = cur.copy()
    rng = np.random.default_rng(3)
    d["occupation"] = rng.choice(14, size=len(d))  # very different distribution
    ks, _ = ks_detector(prof, d)
    chi, per = chi2_detector(prof, d, seed=1)
    assert max(per, key=per.get) == "occupation"
    assert chi["chi2_max"] > 0.2
    assert ks["ks_max"] < 0.06


def test_cramers_v_bounded_and_symmetric_scale(data):
    _, cur, prof = data
    d = cur.copy()
    d["sex"] = 1 - d["sex"]  # flip both categories
    _, per = chi2_detector(prof, d, seed=1)
    assert 0.0 <= per["sex"] <= 1.0


def test_point_mass_shift_detected_by_psi(data):
    _, cur, prof = data
    d = cur.copy()
    d["capital_gain"] = d["capital_gain"] + 0.5
    _, per = psi_detector(prof, d)
    assert per["capital_gain"] > 1.0


def test_unseen_category_handled(data):
    _, cur, prof = data
    d = cur.copy()
    d.loc[:50, "native_country"] = 999
    s, _ = psi_detector(prof, d)
    assert np.isfinite(s["psi_max"])
    assert prof.encode(d).shape[1] == len(CONT) + sum(len(v) for v in prof.categories.values())


def test_interaction_break_invisible_to_marginals(data):
    _, cur, prof = data
    d = cur.copy()
    d["education_num"] = np.random.default_rng(5).permutation(d["education_num"].values)
    ks, _ = ks_detector(prof, d)
    assert ks["ks_max"] < 0.06  # marginals unchanged


def test_mmd_increases_with_shift(data):
    _, cur, prof = data
    small, big = cur.copy(), cur.copy()
    small["age"] += 0.2
    big["age"] += 1.0
    assert mmd_detector(prof, big)["mmd"] > mmd_detector(prof, small)["mmd"] \
        > mmd_detector(prof, cur)["mmd"]


def test_detectors_deterministic(data):
    _, cur, prof = data
    a, _ = run_all_detectors(prof, cur, seed=7, use_lsdd=False)
    b, _ = run_all_detectors(prof, cur, seed=7, use_lsdd=False)
    a.pop("lsdd"), b.pop("lsdd")  # NaN when LSDD is off (NaN != NaN)
    assert a == b


def test_prediction_stats_maxprob_vs_posprob():
    class M:
        def predict_proba(self, X):
            p = np.array([0.1, 0.9, 0.5, 0.3])
            return np.c_[1 - p, p]
    s = prediction_stats(M(), pd.DataFrame({"x": range(4)}))
    assert s["mean_pos_prob"] == pytest.approx(0.45)
    assert s["conf_mean"] == pytest.approx((0.9 + 0.9 + 0.5 + 0.7) / 4)
    assert s["conf_pct_lt_06"] == pytest.approx(0.25)     # only the 0.5 row
    assert s["pct_pos_prob_lt_06"] == pytest.approx(0.75)  # 0.1, 0.5, 0.3
