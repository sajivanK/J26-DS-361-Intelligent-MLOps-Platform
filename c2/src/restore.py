"""
Restore operator: put one feature of a batch back to its normal (reference) distribution.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Used by the label-free Drift Impact Score (restore test) and by label-free
localization (rows at risk). No labels are used, only the reference data.

    Continuous feature   rank-preserving quantile mapping: each row keeps its rank
                         inside the batch and gets the reference value at that
                         rank. The row with the highest capital_gain still has a
                         high capital_gain afterwards, so each row gets a sensible
                         "what if this feature were normal" value. Ties (e.g. the
                         many zero capital gains) are broken at random (seeded),
                         so tied rows spread over the matching reference values.
    Categorical feature  move the fewest rows needed so the category shares match
                         the reference: rows in over-represented categories
                         (including categories never seen in the reference) are
                         reassigned at random (seeded) to under-represented ones.
                         Rows in categories that are not over-represented keep
                         their value.

The answer to "what would this batch look like if feature f had not drifted"
then has the reference marginal for f, with every other feature unchanged.
(Proposed by Sajivan K in the C2 label-free plan, 23 Oct 2026; earlier C2 code
resampled reference values at random, which loses each row's position.)

Usage:
    restored = restore(batch_df, "capital_gain", ref_df, feature_types, seed=0)
"""
import numpy as np


def _restore_continuous(x, ref_sorted, rng):
    n = len(x)
    order = np.lexsort((rng.random(n), x))          # sort by value, random tie-break
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    q = (ranks + 0.5) / n                             # mid-rank quantiles in (0, 1)
    idx = np.clip((q * len(ref_sorted)).astype(int), 0, len(ref_sorted) - 1)
    return ref_sorted[idx]


def _restore_categorical(x, ref_values, rng):
    n = len(x)
    cats, ref_counts = np.unique(ref_values, return_counts=True)
    target = ref_counts / ref_counts.sum() * n
    target_int = np.floor(target).astype(int)
    rest = n - target_int.sum()                       # largest-remainder rounding to n rows
    if rest > 0:
        target_int[np.argsort(-(target - target_int))[:rest]] += 1
    want = dict(zip(cats.tolist(), target_int.tolist()))

    out = x.copy()
    movable = []
    for c in np.unique(x):
        idx = np.flatnonzero(x == c)
        excess = len(idx) - want.get(c.item() if hasattr(c, "item") else c, 0)
        if excess > 0:                                # over-represented (or unseen): free rows
            movable.extend(rng.choice(idx, size=excess, replace=False).tolist())
    rng.shuffle(movable)
    counts = {c: int((x == c).sum()) for c in want}
    pos = 0
    for c, w in want.items():
        need = w - counts.get(c, 0)
        if need > 0:
            take = movable[pos:pos + need]
            out[take] = c
            pos += len(take)
    return out


def restore(df, feature, ref_df, feature_types, seed=0):
    """Copy of df with `feature` restored to the reference distribution."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    x = df[feature].to_numpy()
    if feature_types.get(feature) == "categorical":
        out[feature] = _restore_categorical(x, ref_df[feature].to_numpy(), rng)
    else:
        out[feature] = _restore_continuous(x.astype(float), np.sort(ref_df[feature].to_numpy(float)),
                                           rng).astype(df[feature].dtype, copy=False)
    return out


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.restore
    from scipy.stats import ks_2samp, spearmanr

    from c2.src.scenarios import artifacts, make_scenario
    a = artifacts()
    ref, types = a["ref_df"], a["feature_types"]
    for f, mag in [("capital_gain", 0.8), ("age", 0.6), ("marital_status", 0.6), ("native_country", 0.6)]:
        b = make_scenario("marginal_single", [f], mag, seed=5)["drifted_df"]
        r = restore(b, f, ref, types, seed=1)
        others_same = all((b[g] == r[g]).all() for g in b.columns if g != f)
        if types.get(f) == "categorical":
            ref_share = ref[f].value_counts(normalize=True)
            gap_before = (b[f].value_counts(normalize=True) - ref_share).abs().fillna(ref_share).sum() / 2
            gap_after = (r[f].value_counts(normalize=True) - ref_share).abs().fillna(ref_share).sum() / 2
            moved = float((b[f] != r[f]).mean())
            print(f"{f:<15} category gap {gap_before:.3f} -> {gap_after:.3f}   rows moved {moved:.1%}   "
                  f"other columns unchanged={others_same}")
        else:
            ks0, ks1 = ks_2samp(b[f], ref[f]).statistic, ks_2samp(r[f], ref[f]).statistic
            rho = spearmanr(b[f], r[f]).statistic
            print(f"{f:<15} KS vs reference {ks0:.3f} -> {ks1:.3f}   rank kept (Spearman) {rho:.3f}   "
                  f"other columns unchanged={others_same}")
    print("RESTORE OK")
