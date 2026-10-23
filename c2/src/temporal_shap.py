"""
Novelty 1: Temporal SHAP Monitoring.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

What existing approaches do (Baseline B, Arize/Fiddler style):
    compare mean |SHAP| of ONE production batch against the reference, once.
    Limitations (proposal, Lundberg et al. review): no way to tell a real shift
    from ordinary SHAP noise, no memory across monitoring cycles, and mean |SHAP|
    barely moves when only the relationship between features changes.

What this module adds:
    1. Two explanation-change channels per feature
         distribution channel - Wasserstein distance between the reference and
                                current distributions of the feature's SHAP values
         behaviour channel    - change in the feature's SHAP value FOR THE SAME
                                feature value (conditional on value bins); this moves
                                when the model's reasoning about a feature changes,
                                e.g. when its relationship with another feature
                                breaks (interaction drift), even if the feature's own
                                distribution is unchanged
    2. A noise band: both channels are converted to z-scores against a bootstrap
       null built by resampling clean reference SHAP values, so a change is
       measured as "how many times larger than normal SHAP noise".
    3. Temporal persistence across monitoring cycles: evidence is accumulated
       with an exponentially weighted average, and an alarm requires the same
       feature to be significant in at least k of the last w cycles.
    4. Stratified sampling over the model's prediction space (probability
       deciles), so rare high-score regions are not missed when subsampling.

Main class: TemporalSHAPMonitor
    monitor = TemporalSHAPMonitor(model, ref_df, feature_types)
    for batch in production_batches:          # one call per monitoring cycle
        result = monitor.update(batch)
    result["ranking"], result["alarm"], result["diagnosis_stability"], ...
"""
import numpy as np
import shap
from scipy.stats import wasserstein_distance

from c2.src.config import TARGET_COL


def stratified_sample(df, model, n, seed=0, n_strata=10):
    """Sample n rows, stratified by deciles of the model's predicted probability."""
    if len(df) <= n:
        return df
    proba = model.predict_proba(df)[:, 1]
    strata = np.clip((proba * n_strata).astype(int), 0, n_strata - 1)
    rng = np.random.default_rng(seed)
    picks = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        k = int(round(n * len(idx) / len(df)))
        if k > 0:
            picks.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    picks = np.concatenate(picks)
    return df.iloc[np.sort(picks)]


class TemporalSHAPMonitor:

    def __init__(self, model, ref_df, feature_types, pool_size=5000,
                 sample_size=1000, n_boot=200, n_bins=10, alpha=0.5,
                 persist_k=2, window=3, z_thresh=None, fwer=0.01,
                 stratified=True, seed=42):
        self.model = model
        self.feature_types = feature_types
        self.features = [c for c in ref_df.columns if c != TARGET_COL]
        self.sample_size = sample_size
        self.alpha = alpha
        self.persist_k = persist_k
        self.window = window
        self.stratified = stratified
        self.seed = seed
        self.explainer = shap.TreeExplainer(model)

        # Reference pool: clean SHAP values the model produces normally
        X_ref = ref_df[self.features]
        pool = (stratified_sample(X_ref, model, pool_size, seed)
                if stratified else X_ref.sample(min(pool_size, len(X_ref)),
                                                random_state=seed))
        self.X_ref = pool.values
        self.S_ref = self._shap(pool)

        # Value bins per feature (categories, or reference quantile bins)
        self.edges = {}
        for j, f in enumerate(self.features):
            if feature_types.get(f) == "categorical":
                self.edges[j] = None
            else:
                q = np.quantile(self.X_ref[:, j], np.linspace(0, 1, n_bins + 1))
                self.edges[j] = np.unique(q)
        self.ref_cond = {}
        for j in range(len(self.features)):
            b = self._bins(self.X_ref[:, j], j)
            self.ref_cond[j] = {v: self.S_ref[b == v, j].mean() for v in np.unique(b)}

        # Noise band: bootstrap clean batches of the monitoring sample size
        rng = np.random.default_rng(seed)
        n = min(sample_size, len(self.S_ref) // 2)
        boot_d, boot_b = [], []
        for _ in range(n_boot):
            idx = rng.choice(len(self.S_ref), size=n, replace=False)
            d, b = self._channels(self.X_ref[idx], self.S_ref[idx])
            boot_d.append(d)
            boot_b.append(b)
        boot_d, boot_b = np.array(boot_d), np.array(boot_b)
        self.mu_d, self.sd_d = boot_d.mean(0), boot_d.std(0) + 1e-12
        self.mu_b, self.sd_b = boot_b.mean(0), boot_b.std(0) + 1e-12

        # Alarm threshold controls the family-wise false alarm rate: the
        # (1 - fwer) quantile of the largest clean z over all features/channels
        if z_thresh is None:
            zmax = np.maximum((boot_d - self.mu_d) / self.sd_d,
                              (boot_b - self.mu_b) / self.sd_b).max(axis=1)
            z_thresh = float(np.quantile(zmax, 1 - fwer))
        self.z_thresh = z_thresh
        self.reset()

    # ── internals ──────────────────────────────────────────────────────────
    def _shap(self, X):
        return np.asarray(self.explainer.shap_values(X))

    def _bins(self, x, j):
        if self.edges[j] is None:
            return x.astype(int)
        e = self.edges[j]
        return np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)

    def _channels(self, X, S):
        """Raw distribution-channel and behaviour-channel statistics per feature."""
        dist = np.array([wasserstein_distance(self.S_ref[:, j], S[:, j])
                         for j in range(len(self.features))])
        behav = np.zeros(len(self.features))
        for j in range(len(self.features)):
            b = self._bins(X[:, j], j)
            total, weight = 0.0, 0
            for v in np.unique(b):
                if v not in self.ref_cond[j]:
                    continue
                m = b == v
                if m.sum() < 10:
                    continue
                total += m.sum() * abs(S[m, j].mean() - self.ref_cond[j][v])
                weight += m.sum()
            behav[j] = total / max(weight, 1)
        return dist, behav

    def _sample(self, df):
        X = df[self.features]
        if len(X) <= self.sample_size:
            return X
        if self.stratified:
            return stratified_sample(X, self.model, self.sample_size, self.seed)
        return X.sample(self.sample_size, random_state=self.seed)

    # ── single cycle (noise band only, no temporal memory) ─────────────────
    def score_batch(self, cur_df):
        """Noise-normalised SHAP change for ONE batch. Used alone as an ablation."""
        X = self._sample(cur_df)
        S = self._shap(X)
        d, b = self._channels(X.values, S)
        z_d = (d - self.mu_d) / self.sd_d
        z_b = (b - self.mu_b) / self.sd_b
        z = np.maximum(z_d, z_b)
        mean_abs_ref = np.abs(self.S_ref).mean(0)
        mean_abs_cur = np.abs(S).mean(0)
        return {
            "z": dict(zip(self.features, z.tolist())),
            "z_distribution": dict(zip(self.features, z_d.tolist())),
            "z_behaviour": dict(zip(self.features, z_b.tolist())),
            "ranking": _sorted(dict(zip(self.features, z.tolist()))),
            "significant": [f for f, v in zip(self.features, z) if v > self.z_thresh],
            "mean_abs_shap_ref": dict(zip(self.features, mean_abs_ref.tolist())),
            "mean_abs_shap_current": dict(zip(self.features, mean_abs_cur.tolist())),
        }

    # ── temporal monitoring ────────────────────────────────────────────────
    def reset(self):
        self.history = []          # one score_batch() result per cycle
        self.ewma = None
        self.top1_history = []

    def update(self, cur_df):
        """Process the next monitoring cycle and return the temporal diagnosis."""
        cycle = self.score_batch(cur_df)
        z = np.array([cycle["z"][f] for f in self.features])
        self.ewma = z if self.ewma is None else self.alpha * z + (1 - self.alpha) * self.ewma
        self.history.append(cycle)

        recent = self.history[-self.window:]
        hits = {f: sum(f in c["significant"] for c in recent) for f in self.features}
        persistent = [f for f in self.features if hits[f] >= self.persist_k]
        ranking = _sorted(dict(zip(self.features, self.ewma.tolist())))
        top1 = next(iter(ranking))
        self.top1_history.append(top1)
        last = self.top1_history[-self.window:]
        stability = sum(t == top1 for t in last) / len(last)

        # Pattern hint for the top feature: which channel carries the evidence
        zd, zb = cycle["z_distribution"][top1], cycle["z_behaviour"][top1]
        pattern = "behaviour_shift" if zb > 1.5 * max(zd, 0) else "distribution_shift"

        return {
            "cycle": len(self.history) - 1,
            "ranking": ranking,                    # accumulated evidence (EWMA z)
            "cycle_z": cycle["z"],
            "z_distribution": cycle["z_distribution"],
            "z_behaviour": cycle["z_behaviour"],
            "significant_this_cycle": cycle["significant"],
            "persistent_features": persistent,
            "alarm": len(persistent) > 0,
            "root_cause_feature": top1,
            "diagnosis_stability": round(stability, 3),
            "shift_pattern": pattern,
            "z_thresh": self.z_thresh,
            "mean_abs_shap_ref": cycle["mean_abs_shap_ref"],
            "mean_abs_shap_current": cycle["mean_abs_shap_current"],
        }


def _sorted(scores):
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.temporal_shap
    import time
    from c2.src.scenarios import artifacts, make_temporal_stream

    a = artifacts()
    t0 = time.time()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    print(f"monitor built in {time.time() - t0:.1f}s, alarm threshold z > {mon.z_thresh:.2f}")

    for dtype, feats, mag, label in [
        ("marginal_single", ["marital_status"], 0.4, "marginal marital_status"),
        ("interaction", ["education_num", "occupation"], 0.8, "interaction edu x occ"),
        ("none", [], 0.0, "no drift"),
    ]:
        mon.reset()
        print(f"\n{label} (drift starts at cycle 2)")
        for s in make_temporal_stream(dtype, feats, mag, n_cycles=5, onset=2, seed=7):
            r = mon.update(s["drifted_df"])
            top3 = [f"{f}:{v:.1f}" for f, v in list(r["ranking"].items())[:3]]
            print(f"  cycle {r['cycle']}: alarm={str(r['alarm']):<5} top3={top3} "
                  f"stability={r['diagnosis_stability']} pattern={r['shift_pattern']}")
    print("\nTEMPORAL SHAP OK")