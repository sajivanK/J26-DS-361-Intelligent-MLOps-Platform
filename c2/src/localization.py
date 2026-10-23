"""
Novelty 3: Hierarchical Localization.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Narrows the diagnosis from a feature name to a queryable subgroup rule that C3
can use to target synthetic repair data:

    Level 1  feature      root cause from the Drift Impact Score (Novelty 2)
    Level 2  value range  the range (or categories) of that feature where the
                          drift-harmed rows sit
    Level 3  subgroup     one extra condition on another feature, kept only if
                          it makes the rule clearly more precise

What existing approaches do:
    SliceLine / SHIFT search for slices where the model's error is high NOW,
    with no link to what drifted, so they keep finding the model's usual weak
    spots. The C2 research guide's draft takes the worst-accuracy quartile of the
    root feature and adds a median split of the second feature, again judged on
    raw errors.

What this module adds: rows are targeted by DRIFT HARM, not raw error. A row is
drift-harmed when the model is wrong on it now but becomes right once the root
cause feature is restored to reference values (averaged over several restore
draws). Each level then picks the rule that best matches the harmed rows (F1),
and a level is only kept if it improves the match by at least `min_gain`.

Output fields map onto the C2 -> C3 contract:
Privacy (NFR4): every candidate slice must contain at least MIN_GROUP_SIZE rows
(and at least min_support of the batch), so no reported subgroup is small enough
to risk re-identifying individuals.
    affected_slice_rule      pandas query string, e.g. "capital_gain > 1.21"
    affected_value_ranges    {feature: [low, high]} or {feature: [categories]}
    slice_support            fraction of the batch matched by the rule
"""
import numpy as np

from c2.src.config import TARGET_COL

_INF = float("inf")
MIN_GROUP_SIZE = 20   # NFR4 privacy: no slice smaller than this is scored or reported

def _fmt(v):
    # These numbers are executable C2-to-C3 predicates, not display labels.
    # Round-trip precision preserves the exact mask used to calculate support.
    return repr(float(v))


def _f1(mask, h):
    tp = float(h[mask].sum())
    if tp <= 0:
        return 0.0, 0.0, 0.0
    prec = tp / mask.sum()
    rec = tp / h.sum()
    return 2 * prec * rec / (prec + rec), prec, rec


class HierarchicalLocalizer:

    def __init__(self, model, ref_df, feature_types, n_bins=10, n_restore=5,
                 min_support=0.02, min_gain=0.02, harm_threshold=None, seed=42):
        self.model = model
        self.ref_df = ref_df
        self.feature_types = feature_types
        self.features = [c for c in ref_df.columns if c != TARGET_COL]
        self.n_bins = n_bins
        self.n_restore = n_restore
        self.min_support = min_support
        self.min_gain = min_gain
        self.harm_threshold = harm_threshold
        self.seed = seed

    # ── which rows did the drift harm? ─────────────────────────────────────
    def drift_harm(self, cur_df, root):
        """Per-row probability that restoring `root` turns a wrong prediction right."""
        X = cur_df[self.features]
        y = cur_df[TARGET_COL].values
        wrong = self.model.predict(X) != y
        rng = np.random.default_rng(self.seed)
        fixed = np.zeros(len(X))
        for _ in range(self.n_restore):
            Xr = X.copy()
            Xr[root] = self.ref_df[root].sample(
                len(X), replace=True, random_state=int(rng.integers(1e9))).values
            fixed += wrong & (self.model.predict(Xr) == y)
        return fixed / self.n_restore

    # ── candidate conditions ───────────────────────────────────────────────
    def _range_candidates(self, x, f):
        """Level 2: contiguous value ranges (continuous) or category sets."""
        out = []
        if self.feature_types.get(f) == "categorical":
            for c in np.unique(x):
                out.append((x == c, f"{f} == {int(c)}", [int(c)]))
            return out
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        k = len(edges) - 1
        for i in range(k):
            for j in range(i, k):
                lo = -_INF if i == 0 else edges[i]
                hi = _INF if j == k - 1 else edges[j + 1]
                if lo == -_INF and hi == _INF:
                    continue
                mask = (x > lo) & (x <= hi)
                parts = []
                if lo != -_INF:
                    parts.append(f"{f} > {_fmt(lo)}")
                if hi != _INF:
                    parts.append(f"{f} <= {_fmt(hi)}")
                out.append((mask, " and ".join(parts), [lo, hi]))
        return out

    def _refine_candidates(self, X, base, root):
        """Level 3: one extra condition on another feature, inside the level-2 rows."""
        out = []
        for g in self.features:
            if g == root:
                continue
            x = X[g].values
            if self.feature_types.get(g) == "categorical":
                for c in np.unique(x[base]):
                    out.append((x == c, f"{g} == {int(c)}", g))
            else:
                for q in np.unique(np.quantile(x[base], [0.25, 0.5, 0.75])):
                    out.append((x <= q, f"{g} <= {_fmt(q)}", g))
                    out.append((x > q, f"{g} > {_fmt(q)}", g))
        return out

    # ── main entry point ───────────────────────────────────────────────────
    def localize(self, cur_df, root, ranked_features=None, harm=None):
        """
        harm: optional per-row harm values to localize against. Default is the
        estimated drift harm; the evaluation passes the true harm to build an
        oracle (the best rule of the same shape).
        """
        X = cur_df[self.features]
        n = len(X)
        h = self.drift_harm(cur_df, root) if harm is None else np.asarray(harm, float)
        if harm is None and self.harm_threshold is not None:
            h = (h >= self.harm_threshold).astype(float)
        min_rows = max(int(self.min_support * n), MIN_GROUP_SIZE)
        result = {"root_cause_feature": root,
                  "affected_features_ranked": list(ranked_features or [root])[:3],
                  "harmed_rows_estimate": round(float(h.sum()), 1)}

        if h.sum() < 3:
            # No concentrated drift harm: report the root feature only
            result.update({"affected_slice_rule": "", "affected_value_ranges": {},
                           "slice_support": 1.0, "levels_used": 1,
                           "est_precision": None, "est_recall": None,
                           "note": "no concentrated drift harm found"})
            return result

        # Level 2
        best = (0.0, None)
        for mask, rule, rng_ in self._range_candidates(X[root].values, root):
            if mask.sum() < min_rows:
                continue
            f1, p, r = _f1(mask, h)
            if f1 > best[0]:
                best = (f1, (mask, rule, rng_, p, r))
        if best[1] is None:
            result.update({"affected_slice_rule": "", "affected_value_ranges": {},
                           "slice_support": 1.0, "levels_used": 1,
                           "est_precision": None, "est_recall": None,
                           "note": "no value range met the support constraint"})
            return result
        f1_2, (mask2, rule2, range2, p2, r2) = best

        # Level 3
        best3 = (f1_2, None)
        for cond, rule, g in self._refine_candidates(X, mask2, root):
            mask = mask2 & cond
            if mask.sum() < min_rows:
                continue
            f1, p, r = _f1(mask, h)
            if f1 > best3[0] + self.min_gain:
                best3 = (f1, (mask, rule, g, p, r))

        if best3[1] is not None:
            mask, extra, g, p, r = best3[1]
            rule = f"{rule2} and {extra}"
            levels = 3
        else:
            mask, rule, p, r, levels = mask2, rule2, p2, r2, 2

        result.update({
            "affected_slice_rule": rule,
            "affected_value_ranges": {root: range2},
            "slice_support": round(float(mask.mean()), 4),
            "levels_used": levels,
            "level2_rule": rule2,
            "est_precision": round(p, 3),
            "est_recall": round(r, 3),
            "note": "",
        })
        return result


# ── BASELINES FOR NOVELTY 3 ────────────────────────────────────────────────
def guide_localization(cur_df, model, root, second, n_bins=4):
    """
    The C2 research guide's draft: worst-accuracy quantile bin of the root
    feature, plus the median split (or most common value) of the second feature.
    """
    X = cur_df.drop(columns=[TARGET_COL])
    y = cur_df[TARGET_COL].values
    correct = model.predict(X) == y
    bins = np.unique(np.quantile(X[root], np.linspace(0, 1, n_bins + 1)))
    if len(bins) < 3:
        return f"{root} == {_fmt(X[root].mode()[0])}"
    ids = np.clip(np.searchsorted(bins, X[root].values, side="right") - 1, 0, len(bins) - 2)
    accs = {b: correct[ids == b].mean() for b in np.unique(ids) if (ids == b).sum() >= 20}
    worst = min(accs, key=accs.get)
    rule = f"{root} > {_fmt(bins[worst])} and {root} <= {_fmt(bins[worst + 1])}"
    sv = X[second]
    if sv.nunique() <= 10:
        rule += f" and {second} == {_fmt(sv[ids == worst].mode()[0])}"
    else:
        rule += f" and {second} > {_fmt(sv.median())}"
    return rule


def sliceline_localization(cur_df, model, feature_types, n_bins=10, alpha=0.95,
                           min_support=0.02):
    """
    SliceLine-style search (Sagadeeva & Boehm, 2021) on the current batch,
    levels 1 and 2 (single slices, then pairs of the top single slices), scored
    by error concentration only. No root cause input.
    """
    X = cur_df.drop(columns=[TARGET_COL])
    err = (model.predict(X) != cur_df[TARGET_COL].values).astype(float)
    n, e_all = len(err), max(err.mean(), 1e-9)

    def score(mask):
        s = mask.sum()
        if s < max(min_support * n, 5):
            return -np.inf
        return alpha * (err[mask].mean() / e_all - 1) - (1 - alpha) * (n / s - 1)

    singles = []
    for f in X.columns:
        x = X[f].values
        if feature_types.get(f) == "categorical":
            for c in np.unique(x):
                singles.append((x == c, f"{f} == {int(c)}"))
        else:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                m = (x > lo) & (x <= hi) if i else (x <= hi)
                singles.append((m, f"{f} > {_fmt(lo)} and {f} <= {_fmt(hi)}"
                                if i else f"{f} <= {_fmt(hi)}"))
    scored = sorted(((score(m), m, r) for m, r in singles), key=lambda t: -t[0])
    best = scored[0]
    top = scored[:15]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if top[i][2].split()[0] == top[j][2].split()[0]:
                continue
            m = top[i][1] & top[j][1]
            sc = score(m)
            if sc > best[0]:
                best = (sc, m, f"{top[i][2]} and {top[j][2]}")
    return best[2]



def shift_region_localization(cur_df, ref_df, root, feature_types, n_bins=10,
                              min_support=0.02):
    """
    Label-free baseline: the value range (or category) of the root feature where the
    batch has gained the most share against the reference data ("where the values
    moved"). Uses no labels and no model. It answers "where did the data change",
    not "where does the change matter to the model".
    """
    x, r = cur_df[root].to_numpy(), ref_df[root].to_numpy()
    min_rows = max(int(min_support * len(x)), MIN_GROUP_SIZE)
    best = (-np.inf, "")
    if feature_types.get(root) == "categorical":
        for c in np.unique(x):
            m = x == c
            if m.sum() >= min_rows:
                gain = m.mean() - np.mean(r == c)
                if gain > best[0]:
                    best = (gain, f"{root} == {int(c)}")
        return best[1]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    k = len(edges) - 1
    for i in range(k):
        for j in range(i, k):
            lo = -_INF if i == 0 else edges[i]
            hi = _INF if j == k - 1 else edges[j + 1]
            if lo == -_INF and hi == _INF:
                continue
            m = (x > lo) & (x <= hi)
            if m.sum() < min_rows:
                continue
            gain = m.mean() - np.mean((r > lo) & (r <= hi))
            if gain > best[0]:
                parts = ([f"{root} > {_fmt(lo)}"] if lo != -_INF else []) + \
                        ([f"{root} <= {_fmt(hi)}"] if hi != _INF else [])
                best = (gain, " and ".join(parts))
    return best[1]

if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.localization
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from c2.src.scenarios import artifacts, make_scenario

    a = artifacts()
    loc = HierarchicalLocalizer(a["model"], a["ref_df"], a["feature_types"])
    for dtype, feats, mag, root in [
        ("marginal_single", ["capital_gain"], 0.8, "capital_gain"),
        ("marginal_single", ["marital_status"], 0.6, "marital_status"),
        ("marginal_single", ["age"], 0.8, "age"),
    ]:
        s = make_scenario(dtype, feats, mag, seed=22)
        out = loc.localize(s["drifted_df"], root)
        print(f"{root:<15} harmed~{out['harmed_rows_estimate']:<6} levels={out['levels_used']} "
              f"support={out['slice_support']:<7} rule: {out['affected_slice_rule']}")
    print("LOCALIZATION OK")
