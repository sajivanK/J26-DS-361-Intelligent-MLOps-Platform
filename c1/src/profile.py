"""
Reference profile for C1 detectors.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Everything a detector needs to know about the reference data is computed
ONCE here (at onboarding time), then reused for every production batch:
  - which features are continuous / categorical (from feature_types, never guessed)
  - standardisation statistics for continuous features
  - the known categories of each categorical feature (for one-hot encoding)
  - PSI bin edges and reference bin proportions
  - a fixed reference subsample for the kernel detectors (MMD, LSDD)
    and the MMD kernel bandwidth (median heuristic on that subsample)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

PSI_BINS = 10
PSI_EPS = 1e-4          # floor for empty bins so log() stays finite
KERNEL_SUBSAMPLE = 1000  # rows per side for MMD / LSDD
PROFILE_SEED = 0


class ReferenceProfile:
    def __init__(self, reference_df, feature_types, target_col="target",
                 n_psi_bins=PSI_BINS, kernel_subsample=KERNEL_SUBSAMPLE,
                 seed=PROFILE_SEED):
        features = [c for c in reference_df.columns if c != target_col]
        unknown = [f for f in features if feature_types.get(f) not in
                   ("continuous", "categorical")]
        if unknown:
            raise ValueError(f"feature_types missing or invalid for: {unknown}")

        self.target_col = target_col
        self.features = features
        self.continuous = [f for f in features if feature_types[f] == "continuous"]
        self.categorical = [f for f in features if feature_types[f] == "categorical"]
        self.ref = reference_df[features].reset_index(drop=True)

        # Standardisation (continuous only) for the kernel detectors
        cont = self.ref[self.continuous].astype(float)
        self.mean = cont.mean().values
        self.std = cont.std(ddof=0).values + 1e-12

        # Known categories (sorted for a stable one-hot layout)
        self.categories = {f: np.sort(self.ref[f].unique()) for f in self.categorical}

        # PSI: reference-quantile interior edges for continuous features.
        # np.unique collapses repeated quantiles, so a point-mass feature
        # (e.g. capital_gain, mostly one value) keeps the pile in its own bin.
        self.psi_edges = {}
        self.psi_ref_props = {}
        qs = np.linspace(0, 1, n_psi_bins + 1)[1:-1]
        for f in self.continuous:
            edges = np.unique(np.quantile(self.ref[f].values.astype(float), qs))
            self.psi_edges[f] = edges
            self.psi_ref_props[f] = _bin_props(self.ref[f].values, edges)
        for f in self.categorical:
            self.psi_ref_props[f] = _cat_props(self.ref[f].values, self.categories[f])

        # Fixed reference subsample + kernel bandwidth for MMD / LSDD
        n = min(kernel_subsample, len(self.ref))
        self.kernel_n = kernel_subsample
        self.ref_kernel = self.encode(self.ref.sample(n=n, random_state=seed))
        d2 = euclidean_distances(self.ref_kernel, squared=True)
        self.mmd_sigma = float(np.sqrt(0.5 * np.median(d2[np.triu_indices(n, k=1)])))
        self.ref_kxx = float(_rbf(self.ref_kernel, self.ref_kernel, self.mmd_sigma).mean())

    def encode(self, df):
        """Standardised continuous columns + one-hot categoricals (reference
        categories only; a category never seen in the reference maps to all zeros)."""
        parts = []
        if self.continuous:
            parts.append((df[self.continuous].values.astype(float) - self.mean) / self.std)
        for f in self.categorical:
            vals = df[f].values
            parts.append((vals[:, None] == self.categories[f][None, :]).astype(float))
        return np.hstack(parts).astype(np.float64)


def _bin_props(values, interior_edges):
    # right=True: bin i holds edges[i-1] < x <= edges[i]
    idx = np.digitize(values.astype(float), interior_edges, right=True)
    counts = np.bincount(idx, minlength=len(interior_edges) + 1)
    return counts / counts.sum()


def _cat_props(values, categories):
    # last slot = any category not seen in the reference
    pos = {c: i for i, c in enumerate(categories)}
    idx = np.array([pos.get(v, len(categories)) for v in values])
    counts = np.bincount(idx, minlength=len(categories) + 1)
    return counts / counts.sum()


def _rbf(X, Y, sigma):
    return np.exp(-euclidean_distances(X, Y, squared=True) / (2.0 * sigma ** 2))
