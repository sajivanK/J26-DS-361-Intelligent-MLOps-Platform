"""MMD detector: biased RBF MMD^2 on all features together.

Features are encoded by the reference profile (standardised continuous +
one-hot categoricals), so categories are not treated as numbers.
The bandwidth and the reference subsample are fixed at onboarding, so scores
are comparable across batches.
"""
from ..profile import _rbf


def mmd_detector(profile, batch, seed=0):
    n = min(profile.kernel_n, len(batch))
    Y = profile.encode(batch.sample(n=n, random_state=seed))
    X, s = profile.ref_kernel, profile.mmd_sigma
    mmd2 = profile.ref_kxx + _rbf(Y, Y, s).mean() - 2.0 * _rbf(X, Y, s).mean()
    return {"mmd": float(max(0.0, mmd2))}
