"""
Novelty 2: Drift Impact Score (DIS).
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Question it answers: not just "which feature changed" but "which feature change
actually HURTS the model". It fuses three evidence sources PER FEATURE:

    E  explanation evidence  Novelty 1 temporal SHAP z-score (changes in the
                             model's reasoning, including interaction drift)
    D  drift evidence        -log10 p of a KS / chi-square test (how strongly the
                             feature's data distribution changed)
    P  performance evidence  restore test: put the feature back to reference
                             values and measure how much the log-loss recovers,
                             compared with how much it recovers on CLEAN batches
                             (restoring a feature always costs some loss through
                             normal permutation importance, so the raw gain must
                             be judged against that feature's own clean baseline)

Fix over the C2 research guide's draft: there, the performance and detector
signals were one global number added equally to every feature, so they never
changed the ranking and the "fusion" reduced to SHAP alone. Here all three
signals are per feature.

Every signal is first expressed as evidence against noise (z-score or -log10 p)
and then mapped to [0, 1) with s(x) = x / (x + c), where c is that signal's own
significance level, so s = 0.5 exactly where the signal becomes significant.
Scores are therefore comparable across batches, not just within one.

    DIS_f = wE * E_f + wD * D_f + wP * P_f         (default weights 0.4 / 0.2 / 0.4)

Malignant vs benign (team definition: accuracy drop > 2pp): flagged malignant
when the top feature's performance evidence is significant (P_f >= 0.5) AND its
estimated harm exceeds 2pp. Estimated harm = accuracy recovered by restoring the
feature, minus the recovery seen on clean batches. A feature can drift strongly
(high D) yet be benign (low P).

Usage:
    dis = DriftImpactScorer(model, ref_df, feature_types, z_thresh=monitor.z_thresh)
    dis.calibrate_performance(clean_warmup_batches)   # first clean cycles
    out = dis.score(current_batch, monitor_result)
"""
import numpy as np
import pandas as pd

from c2.evaluation.baselines import drift_only_ranking
from c2.src.config import TARGET_COL

DEFAULT_WEIGHTS = (0.4, 0.2, 0.4)   # (explanation, drift, performance)
P_Z_SIGNIFICANT = 3.0               # performance z treated as significant
MALIGNANT_PP = 2.0                  # team definition (inject_drift): drop > 2pp
_EPS = 1e-9


def _log_loss(y, p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _sat(x, c):
    x = max(float(x), 0.0)
    return x / (x + c)


class DriftImpactScorer:

    def __init__(self, model, ref_df, feature_types, z_thresh,
                 weights=DEFAULT_WEIGHTS, n_restore=2, seed=42):
        self.model = model
        self.ref_df = ref_df
        self.feature_types = feature_types
        self.features = [c for c in ref_df.columns if c != TARGET_COL]
        self.weights = weights
        self.n_restore = n_restore
        self.seed = seed
        self.c_E = float(z_thresh)                                 # Novelty 1 alarm level
        self.c_D = float(-np.log10(0.05 / len(self.features)))    # Bonferroni 5%
        self.c_P = P_Z_SIGNIFICANT
        self.mu_P = None
        self.sd_P = None

    # ── performance evidence ───────────────────────────────────────────────
    def restore_gains(self, df, seed=0):
        """
        Log-loss and accuracy (pp) recovered when each feature is restored to
        reference values. Returns (loss_gains, accuracy_gains).
        """
        X = df[self.features]
        y = df[TARGET_COL].values
        p0 = self.model.predict_proba(X)[:, 1]
        base_loss, base_acc = _log_loss(y, p0), np.mean((p0 >= 0.5) == y)
        rng = np.random.default_rng(seed)
        loss_gains, acc_gains = {}, {}
        for f in self.features:
            losses, accs = [], []
            for _ in range(self.n_restore):
                Xr = X.copy()
                Xr[f] = self.ref_df[f].sample(
                    len(X), replace=True, random_state=int(rng.integers(1e9))).values
                p = self.model.predict_proba(Xr)[:, 1]
                losses.append(_log_loss(y, p))
                accs.append(np.mean((p >= 0.5) == y))
            loss_gains[f] = base_loss - float(np.mean(losses))
            acc_gains[f] = 100 * (float(np.mean(accs)) - base_acc)
        return loss_gains, acc_gains

    def calibrate_performance(self, clean_batches, n_boot=15, batch_size=1000):
        """
        Learn each feature's normal restore gain (mean, sd) from clean production
        batches, e.g. the warm-up cycles before any drift. Bootstrap batches of the
        monitoring size are drawn from the pooled clean rows.
        """
        pool = pd.concat(clean_batches, ignore_index=True)
        rng = np.random.default_rng(self.seed)
        rows, accs = [], []
        for i in range(n_boot):
            b = pool.sample(min(batch_size, len(pool)), replace=True,
                            random_state=int(rng.integers(1e9)))
            g, a = self.restore_gains(b, seed=i)
            rows.append([g[f] for f in self.features])
            accs.append([a[f] for f in self.features])
        rows, accs = np.array(rows), np.array(accs)
        self.mu_P = dict(zip(self.features, rows.mean(0)))
        self.sd_P = dict(zip(self.features, np.maximum(rows.std(0), 1e-3)))
        self.mu_acc = dict(zip(self.features, accs.mean(0)))
        return self

    # ── signals and fusion ─────────────────────────────────────────────────
    def signals(self, cur_df, shap_result):
        """Evidence (E, D, P) per feature, raw and normalised to [0, 1)."""
        if self.mu_P is None:
            raise RuntimeError("call calibrate_performance(clean_batches) first")
        z = shap_result["ranking"]                       # Novelty 1 (EWMA z)
        drift = drift_only_ranking(self.ref_df, cur_df, self.feature_types)
        gains, accs = self.restore_gains(cur_df, seed=self.seed)
        raw = {}
        for f in self.features:
            raw[f] = {"E": float(z[f]),
                      "D": float(drift["ranking"][f]),
                      "P": float((gains[f] - self.mu_P[f]) / self.sd_P[f]),
                      "harm_pp": float(accs[f] - self.mu_acc[f])}
        norm = {f: {"E": _sat(r["E"], self.c_E),
                    "D": _sat(r["D"], self.c_D),
                    "P": _sat(r["P"], self.c_P)} for f, r in raw.items()}
        return raw, norm

    def fuse(self, norm, weights=None):
        wE, wD, wP = weights or self.weights
        dis = {f: wE * s["E"] + wD * s["D"] + wP * s["P"] for f, s in norm.items()}
        return dict(sorted(dis.items(), key=lambda kv: kv[1], reverse=True))

    def score(self, cur_df, shap_result, weights=None):
        raw, norm = self.signals(cur_df, shap_result)
        dis = self.fuse(norm, weights)
        top = next(iter(dis))
        harm = raw[top]["harm_pp"]
        return {
            "drift_impact_scores": {f: round(v, 4) for f, v in dis.items()},
            "root_cause_feature": top,
            "top_score": round(dis[top], 4),
            "estimated_harm_pp": round(harm, 2),
            "is_malignant": bool(norm[top]["P"] >= 0.5 and harm > MALIGNANT_PP),
            "signals_raw": raw,
            "signals_normalised": norm,
            "weights": weights or self.weights,
        }


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.drift_impact
    import time
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from c2.src.scenarios import artifacts, make_temporal_stream
    from c2.src.temporal_shap import TemporalSHAPMonitor

    a = artifacts()
    t0 = time.time()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    dis = DriftImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)
    print(f"built in {time.time() - t0:.1f}s")

    for dtype, feats, mag in [
        ("marginal_single", ["capital_gain"], 0.8),
        ("marginal_single", ["marital_status"], 0.4),
        ("marginal_single", ["native_country"], 0.6),
        ("interaction", ["age", "marital_status"], 0.8),
        ("none", [], 0.0),
    ]:
        mon.reset()
        stream = make_temporal_stream(dtype, feats, mag, seed=22)
        for s in stream:
            r = mon.update(s["drifted_df"])
        dis.calibrate_performance([s["drifted_df"] for s in stream[:2]])
        out = dis.score(stream[-1]["drifted_df"], r)
        top3 = [f"{f}:{v:.2f}" for f, v in list(out["drift_impact_scores"].items())[:3]]
        harm = stream[-1]["ground_truth"]["actual_accuracy_drop"]
        print(f"{dtype:<16} {'+'.join(feats) or '-':<20} true harm={harm:5.1f}pp | "
              f"est harm={out['estimated_harm_pp']:5.1f}pp malignant={out['is_malignant']!s:<5} "
              f"top3={top3}")
    print("DRIFT IMPACT OK")