"""
Baselines for Novelty 1: the EXISTING approaches C2 must beat.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Baseline A - Drift-only ranking (proposal Section 3.3, Baseline 1)
    Rank features by how much their DATA distribution changed, the way common
    drift tools (e.g. Evidently AI) do it: KS test for continuous features,
    chi-square test for categorical features, ranked by -log10(p-value) so
    both test types are on the same scale. The model is never consulted.

Baseline B - Snapshot SHAP ranking (proposal Section 3.3, Baseline 2)
    Rank features by the change in mean |SHAP| between the reference data and
    ONE production batch. This is the single-point comparison used by Arize AI /
    Fiddler AI and by the compute_temporal_shap() in the C2 research guide.
    It has no notion of noise and no memory of previous cycles.

Baseline 4 - Subgroup-only ranking (proposal Section 3.3, Baseline 4)
    Rank features by how strongly the model's errors concentrate in one of the
    feature's value slices, using the SliceLine slice score (Sagadeeva & Boehm,
    2021) on the current batch. Finds WHERE the model fails now, with no
    explanation or drift evidence.

(Baseline 3, performance-only, is the P signal of the Drift Impact Score used
on its own; it lives in c2/src/drift_impact.py and is ranked in the Novelty 2
evaluation.)

All return a dict {feature: score} sorted from most to least suspicious.
"""
import numpy as np
import shap
from scipy.stats import chi2_contingency, ks_2samp

from c2.src.config import TARGET_COL

_P_FLOOR = 1e-300   # avoid log10(0)


def _features(df):
    return [c for c in df.columns if c != TARGET_COL]


def _sorted(scores):
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ── BASELINE A: DRIFT-ONLY ─────────────────────────────────────────────────
def drift_only_ranking(ref_df, cur_df, feature_types):
    """
    Score = -log10(p) of a two-sample test per feature.
    Higher score = stronger evidence the feature's distribution changed.
    """
    scores, pvalues = {}, {}
    for f in _features(ref_df):
        ref, cur = ref_df[f].values, cur_df[f].values
        if feature_types.get(f) == "categorical":
            cats = np.union1d(ref, cur)
            table = np.array([[np.sum(ref == c) for c in cats],
                              [np.sum(cur == c) for c in cats]])
            table = table[:, table.sum(axis=0) > 0]
            p = 1.0 if table.shape[1] < 2 else chi2_contingency(table)[1]
        else:
            p = ks_2samp(ref, cur).pvalue
        pvalues[f] = float(p)
        scores[f] = float(-np.log10(max(p, _P_FLOOR)))
    return {"ranking": _sorted(scores), "pvalues": pvalues}


# ── BASELINE B: SNAPSHOT SHAP ──────────────────────────────────────────────
class SnapshotSHAP:
    """
    Single-point SHAP comparison. The reference mean |SHAP| is computed once
    (as a production tool would cache it); each call compares one batch to it.
    """

    def __init__(self, model, ref_df, sample_size=1000, seed=42):
        self.explainer = shap.TreeExplainer(model)
        self.features = _features(ref_df)
        X_ref = ref_df[self.features].sample(
            n=min(sample_size, len(ref_df)), random_state=seed)
        self.ref_mean_abs = np.abs(self.explainer.shap_values(X_ref)).mean(axis=0)
        self.sample_size = sample_size
        self.seed = seed

    def mean_abs_shap(self, df):
        X = df[self.features]
        if len(X) > self.sample_size:
            X = X.sample(n=self.sample_size, random_state=self.seed)
        return np.abs(self.explainer.shap_values(X)).mean(axis=0)

    def rank(self, cur_df):
        change = np.abs(self.mean_abs_shap(cur_df) - self.ref_mean_abs)
        return {"ranking": _sorted(dict(zip(self.features, change.tolist())))}


# ── BASELINE 4: SUBGROUP-ONLY (SLICELINE STYLE) ───────────────────────────
def subgroup_only_ranking(cur_df, model, feature_types, ref_df=None,
                          n_bins=10, alpha=0.95, min_support=0.05):
    """
    SliceLine score for single-feature slices (one category, or one quantile
    bin of a continuous feature):
        score(s) = alpha * (err_s / err_all - 1) - (1 - alpha) * (n / |s| - 1)
    A feature's score is its best slice. Slices smaller than min_support of the
    batch are ignored (SliceLine's minimum support constraint).
    """
    feats = _features(cur_df)
    X = cur_df[feats]
    err = (model.predict(X) != cur_df[TARGET_COL].values).astype(float)
    n, err_all = len(err), max(err.mean(), 1e-9)
    edge_src = ref_df if ref_df is not None else cur_df
    scores, best = {}, {}
    for f in feats:
        x = X[f].values
        if feature_types.get(f) == "categorical":
            groups = x
        else:
            edges = np.unique(np.quantile(edge_src[f].values,
                                          np.linspace(0, 1, n_bins + 1)))
            groups = np.clip(np.searchsorted(edges, x, side="right") - 1,
                             0, len(edges) - 2)
        top, top_g = -np.inf, None
        for g in np.unique(groups):
            m = groups == g
            size = m.sum()
            if size < min_support * n:
                continue
            sc = alpha * (err[m].mean() / err_all - 1) - (1 - alpha) * (n / size - 1)
            if sc > top:
                top, top_g = sc, g
        scores[f] = float(top) if np.isfinite(top) else -1.0
        best[f] = top_g
    return {"ranking": _sorted(scores), "best_slice": best}


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.evaluation.baselines
    from c2.src.scenarios import artifacts, make_scenario

    a = artifacts()
    snap = SnapshotSHAP(a["model"], a["ref_df"])
    cases = [
        ("marginal_single", ["education_num"], 0.6),
        ("marginal_single", ["marital_status"], 0.6),
        ("marginal_single", ["native_country"], 0.6),
        ("interaction", ["education_num", "occupation"], 0.8),
        ("none", [], 0.0),
    ]
    print(f"{'scenario':<40} {'Baseline A top-3':<45} {'Baseline B top-3':<50} Baseline 4 top-3")
    for dtype, feats, mag in cases:
        s = make_scenario(dtype, feats, mag)
        A = list(drift_only_ranking(a["ref_df"], s["drifted_df"],
                                    a["feature_types"])["ranking"])[:3]
        B = list(snap.rank(s["drifted_df"])["ranking"])[:3]
        G = list(subgroup_only_ranking(s["drifted_df"], a["model"], a["feature_types"],
                                       ref_df=a["ref_df"])["ranking"])[:3]
        name = f"{dtype} {'+'.join(feats) or '-'} m={mag}"
        print(f"{name:<40} {str(A):<45} {str(B):<50} {G}")
    print("BASELINES OK")