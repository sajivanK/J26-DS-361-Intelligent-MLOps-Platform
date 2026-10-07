"""LSDD detector: Least-Squares Density Difference from alibi-detect.

Uses the same encoded fixed reference subsample as MMD. alibi-detect is a
hard requirement: if it is missing this raises (no silent 0.0).
"""
import sys
import numpy as np

LSDD_PERMUTATIONS = 50
_CACHE = {}


def _get_detector(profile):
    key = id(profile)
    if key not in _CACHE:
        # Colab ships TensorFlow + transformers versions that break
        # alibi-detect's optional TF imports; we only use the PyTorch backend.
        if "tensorflow" not in sys.modules:
            sys.modules["tensorflow"] = None
        from alibi_detect.cd import LSDDDrift
        _CACHE.clear()
        _CACHE[key] = LSDDDrift(profile.ref_kernel.astype(np.float32),
                                backend="pytorch", p_val=0.05,
                                n_permutations=LSDD_PERMUTATIONS)
    return _CACHE[key]


def lsdd_detector(profile, batch, seed=0):
    cd = _get_detector(profile)
    n = min(profile.kernel_n, len(batch))
    Y = profile.encode(batch.sample(n=n, random_state=seed)).astype(np.float32)
    out = cd.predict(Y)
    return {"lsdd": float(np.ravel(out["data"]["distance"])[0])}
