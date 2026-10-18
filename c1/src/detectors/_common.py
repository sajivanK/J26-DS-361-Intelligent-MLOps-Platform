"""Shared helpers for C1 detectors."""
import numpy as np


def aggregate(per_feature, prefix):
    """Mean and max of per-feature scores. Max catches one strongly drifted
    feature that a mean over many unchanged features would dilute."""
    vals = np.array(list(per_feature.values()), dtype=float)
    if vals.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    return {f"{prefix}_mean": float(vals.mean()), f"{prefix}_max": float(vals.max())}
