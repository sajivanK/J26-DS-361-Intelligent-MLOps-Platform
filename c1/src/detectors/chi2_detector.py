"""Chi-squared detector: per CATEGORICAL feature, reported as bias-corrected
Cramer's V on a BALANCED table.

Why not plain Cramer's V on reference vs batch?
  1. Imbalance: the reference (~26k rows) is far larger than a batch (500 to
     3,000), and plain V then shrinks with batch size for the same drift.
     Fix: subsample the reference to the batch size (seeded) so every table
     is balanced and scores are comparable across batch sizes.
  2. Small-sample bias: with many categories (native_country has 41) plain V
     is well above 0 even with no drift. Fix: Bergsma (2013) bias correction
     (verify citation details before using in the thesis).
Yates' correction is off so 2-category features are treated like the others.
Checked on synthetic data: no-drift V stays near 0 for 2, 7 and 41
categories, and a fixed shift gives the same V at batch sizes 500 and 3,000.
"""
import numpy as np
from scipy.stats import chi2_contingency
from ._common import aggregate


def bias_corrected_cramers_v(table):
    n = table.sum()
    r, k = table.shape
    if k < 2 or n < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    k_c = k - (k - 1) ** 2 / (n - 1)
    r_c = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_c - 1, r_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _table(ref_vals, batch_vals):
    cats = np.union1d(np.unique(ref_vals), np.unique(batch_vals))
    return np.array([[np.sum(ref_vals == c) for c in cats],
                     [np.sum(batch_vals == c) for c in cats]], dtype=float)


def chi2_detector(profile, batch, seed=0):
    n = min(len(batch), len(profile.ref))
    ref_sub = profile.ref.sample(n=n, random_state=seed)
    per = {f: bias_corrected_cramers_v(_table(ref_sub[f].values, batch[f].values))
           for f in profile.categorical}
    return aggregate(per, "chi2"), per
