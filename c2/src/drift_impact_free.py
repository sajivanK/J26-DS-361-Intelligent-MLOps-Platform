"""
Novelty 2, label-free: Drift Impact Score without production labels.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

In production C2 never sees the true answers of a batch. It only has
  - the deployed model (its predictions on any rows),
  - the reference data the model was trained on (what "normal" looks like),
  - C1's alert (contract v1.1).
So every signal here uses inputs and the model's predictions only; the label
column, if a batch has one, is never read.

Per feature f, four evidence sources:

    E   explanation evidence   Novelty 1 temporal SHAP z (unchanged)
    D   drift evidence         -log10 p of KS / chi-square (unchanged)
    DI  impact (restore test   put f back to its normal distribution with the
        without labels)        rank-preserving restore operator (c2/src/restore.py)
                               and ask how much changes. The operator leaves an
                               undrifted feature almost unchanged, so the result
                               is read on an ABSOLUTE scale (above its small
                               clean-batch mean), not as a z-score: z-scores made
                               tiny effects on features with very low clean noise
                               look large (pilot, 23 Oct).
          harm drop            with a label-free harm predictor (C1's predict_harm,
          (when available)     the plan agreed with C1 on 23 Oct 2026): how much
                               PREDICTED HARM disappears, in pp,
                                   harm(batch) - harm(batch with f restored);
                               significant at the team's 2 pp harm line.
          decision change      without a harm predictor: the share of the model's
          (default)            decisions that flip when f is restored. Restoring f
                               can change accuracy by at most this share, so 2% of
                               decisions is the most that could reach the 2 pp
                               harm line: that is its significance level.
                               (Pilot history: "how far predictions move" with
                               random reference resampling ranked important
                               features first whether or not they drifted, 50%
                               top-1 alone; "how much of the output gap closes"
                               as a z-score blamed a harmless drifted feature in
                               harmful+harmless pairs. Both replaced.)
    PS  prediction-shift       inside each value slice of f (category or
        concentration          reference decile), how far the model's mean
                               predicted probability moved against the same slice
                               in the clean warm-up cycles, as a two-sample z;
                               the largest slice counts. It catches interaction
                               drift: the partner feature's slices look normal,
                               but the model treats them differently.

Each signal becomes evidence against noise and is mapped to [0, 1) with
s(x) = x / (x + c), c = the signal's significance level (s = 0.5 exactly where
it becomes significant), then fused:

    DIS_f = wE E_f + wD D_f + wDI DI_f + wPS PS_f      (default 0.35/0.15/0.35/0.15,
                                                        the same weights as the
                                                        labelled score, fixed
                                                        before evaluation)

How much accuracy the drift costs cannot be measured without labels: it is
estimated by the harm predictor (C1 when connected), batch level and per feature
(the harm drop of the root feature, minus its clean-batch baseline).

Rows at risk (for label-free localization, Novelty 3): for every row, how far the
model's predicted probability moves when the root feature is restored.

Usage:
    sc = LabelFreeImpactScorer(model, ref_df, feature_types, z_thresh=monitor.z_thresh,
                               harm_fn=c1_predict_harm or doc_harm(model, ref_df))
    sc.calibrate(clean_warmup_batches)            # unlabelled clean batches
    out = sc.score(current_batch, monitor_result, c1_output)
"""
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance

from c2.evaluation.baselines import drift_only_ranking
from c2.src.config import TARGET_COL
from c2.src.localization import MIN_GROUP_SIZE
from c2.src.restore import restore

DEFAULT_WEIGHTS = (0.35, 0.15, 0.35, 0.15)   # (explanation, drift, decision impact, pred. shift)
DI_FLIP_SIGNIFICANT = 0.02                    # 2% of decisions ~ the 2 pp harm line
DI_HARM_SIGNIFICANT = 2.0                     # pp, the team's harm line
PS_Z_SIGNIFICANT = 3.0
MALIGNANT_PP = 2.0                            # team definition, applied to C1's estimate
_EPS = 1e-9


def _sat(x, c):
    x = max(float(x), 0.0)
    return x / (x + c)


def doc_harm(model, ref_df, n=5000, seed=0):
    """Label-free harm predictor (DoC, difference of confidences): how much the model's
    average confidence fell against normal data, in percentage points. A stand-in for
    C1's predict_harm until C1's engine is connected. Known weakness: it misses drift
    that makes the model confidently wrong."""
    feats = [c for c in ref_df.columns if c != TARGET_COL]
    p_ref = model.predict_proba(ref_df[feats].sample(min(n, len(ref_df)), random_state=seed))[:, 1]
    conf_ref = float(np.maximum(p_ref, 1 - p_ref).mean())

    def harm(df):
        p = model.predict_proba(df[feats])[:, 1]
        return 100 * (conf_ref - float(np.maximum(p, 1 - p).mean()))
    harm.name = "DoC"
    return harm


class LabelFreeImpactScorer:

    def __init__(self, model, ref_df, feature_types, z_thresh,
                 weights=DEFAULT_WEIGHTS, n_restore=2, harm_fn=None, seed=42):
        self.model = model
        self.ref_df = ref_df
        self.feature_types = feature_types
        self.features = [c for c in ref_df.columns if c != TARGET_COL]
        self.weights = weights
        self.n_restore = n_restore
        self.seed = seed
        self.c = {"E": float(z_thresh),
                  "D": float(-np.log10(0.05 / len(self.features))),   # Bonferroni 5%
                  "DI": DI_HARM_SIGNIFICANT if harm_fn else DI_FLIP_SIGNIFICANT,
                  "PS": PS_Z_SIGNIFICANT}
        self.harm_fn = harm_fn                       # batch -> predicted harm (pp), no labels
        self.harm_name = getattr(harm_fn, "name", "harm predictor") if harm_fn else None
        self.base = None                             # clean-batch baselines of the restore test
        self.warm_slices = None
        ref_sample = ref_df.sample(min(5000, len(ref_df)), random_state=seed)
        self.p_ref = self._proba(self._X(ref_sample))          # the model's normal outputs
        self.edges = {}
        for f in self.features:
            if feature_types.get(f) != "categorical":
                self.edges[f] = np.unique(np.quantile(ref_df[f], np.linspace(0, 1, 11)))

    # ── helpers ─────────────────────────────────────────────────────────────
    def _X(self, df):
        return df[self.features]                       # the label column is never used

    def _proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def _slices(self, df, f):
        x = df[f].values
        if f not in self.edges:
            return x.astype(int)
        e = self.edges[f]
        return np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)

    # ── decision impact (restore test on predictions) ───────────────────────
    def restore_test(self, df, seed=0, features=None):
        """Per feature, with the feature restored to its normal distribution:
        gap_closed  how much of the gap between the batch's outputs and the model's
                    normal outputs closes
        harm_drop   how much predicted harm disappears (None without a harm predictor)
        flip_share  share of decisions that flip"""
        X = self._X(df)
        p0 = self._proba(X)
        gap0 = wasserstein_distance(p0, self.p_ref)
        h0 = self.harm_fn(X) if self.harm_fn else None
        out = {}
        for k, f in enumerate(features or self.features):
            c, hd, fl = [], [], []
            for r in range(self.n_restore):
                Xr = restore(X, f, self.ref_df, self.feature_types, seed=seed * 1000 + k * 10 + r)
                p = self._proba(Xr)
                c.append(gap0 - wasserstein_distance(p, self.p_ref))
                fl.append(np.mean((p >= 0.5) != (p0 >= 0.5)))
                if self.harm_fn:
                    hd.append(h0 - self.harm_fn(Xr))
            out[f] = {"gap_closed": float(np.mean(c)), "flip_share": float(np.mean(fl)),
                      "harm_drop": float(np.mean(hd)) if hd else None}
        return out

    def rows_at_risk(self, df, root, seed=0):
        """Per row: how far the model's predicted probability moves when the root
        feature is restored (averaged over restore draws), and whether its decision flips."""
        X = self._X(df)
        p0 = self._proba(X)
        moves, flips = np.zeros(len(X)), np.zeros(len(X))
        for r in range(self.n_restore):
            p = self._proba(restore(X, root, self.ref_df, self.feature_types, seed=seed * 1000 + r))
            moves += np.abs(p - p0)
            flips += (p >= 0.5) != (p0 >= 0.5)
        return moves / self.n_restore, flips / self.n_restore

    # ── prediction-shift concentration ─────────────────────────────────────
    def _slice_stats(self, df):
        """{feature: {slice: (n, mean p, var p)}} of the model's predictions."""
        p = self._proba(self._X(df))
        out = {}
        for f in self.features:
            s = self._slices(df, f)
            out[f] = {int(v): (int((s == v).sum()), float(p[s == v].mean()),
                               float(p[s == v].var()))
                      for v in np.unique(s)}
        return out

    def prediction_shift_z(self, cur_df):
        cur = self._slice_stats(cur_df)
        min_n = max(int(0.02 * len(cur_df)), MIN_GROUP_SIZE)
        z = {}
        for f in self.features:
            best = 0.0
            for v, (n1, m1, v1) in cur[f].items():
                n0, m0, v0 = self.warm_slices[f].get(v, (0, 0.0, 0.0))
                if n1 < min_n or n0 < 10:
                    continue
                se = np.sqrt(max(v1, 1e-6) / n1 + max(v0, 1e-6) / n0)
                best = max(best, abs(m1 - m0) / se)
            z[f] = float(best)
        return z

    # ── calibration on clean, unlabelled batches ────────────────────────────
    def calibrate(self, clean_batches, n_boot=15, batch_size=1000):
        """Normal restore-test results per feature (mean, sd) and normal per-slice
        predictions, from the first clean production cycles (no labels needed)."""
        pool = pd.concat([self._X(b) for b in clean_batches], ignore_index=True)
        rng = np.random.default_rng(self.seed)
        runs = []
        for i in range(n_boot):
            b = pool.sample(min(batch_size, len(pool)), replace=True,
                            random_state=int(rng.integers(1e9)))
            runs.append(self.restore_test(b, seed=i + 1))
        self.base = {}
        for key, floor in (("gap_closed", 1e-4), ("harm_drop", 0.05), ("flip_share", 1e-4)):
            if runs[0][self.features[0]][key] is None:
                continue
            m = np.array([[r[f][key] for f in self.features] for r in runs])
            self.base[key] = (dict(zip(self.features, m.mean(0))),
                              dict(zip(self.features, np.maximum(m.std(0), floor))))
        self.warm_slices = self._slice_stats(pool)
        return self

    def _z(self, key, f, value):
        mu, sd = self.base[key]
        return float((value - mu[f]) / sd[f])

    # ── signals and fusion ─────────────────────────────────────────────────
    def signals(self, cur_df, shap_result):
        if self.base is None:
            raise RuntimeError("call calibrate(clean_batches) first")
        z = shap_result["ranking"]                              # Novelty 1 (EWMA z)
        drift = drift_only_ranking(self.ref_df, self._X(cur_df), self.feature_types)
        rt = self.restore_test(cur_df, seed=self.seed)
        ps = self.prediction_shift_z(cur_df)
        raw = {}
        for f in self.features:
            flip_excess = rt[f]["flip_share"] - self.base["flip_share"][0][f]
            harm_excess = (rt[f]["harm_drop"] - self.base["harm_drop"][0][f]
                           if rt[f]["harm_drop"] is not None else None)
            raw[f] = {"E": float(z[f]), "D": float(drift["ranking"][f]),
                      "DI": harm_excess if harm_excess is not None else flip_excess,
                      "PS": ps[f], "decision_change": flip_excess, "harm_excess": harm_excess,
                      "gap_closed": rt[f]["gap_closed"], "harm_drop": rt[f]["harm_drop"],
                      "flip_share": rt[f]["flip_share"]}
        norm = {f: {k: _sat(r[k], self.c[k]) for k in ("E", "D", "DI", "PS")}
                for f, r in raw.items()}
        return raw, norm

    def fuse(self, norm, weights=None):
        wE, wD, wDI, wPS = tuple(weights or self.weights)
        dis = {f: wE * s["E"] + wD * s["D"] + wDI * s["DI"] + wPS * s["PS"]
               for f, s in norm.items()}
        return dict(sorted(dis.items(), key=lambda kv: kv[1], reverse=True))

    def score(self, cur_df, shap_result, c1_output=None, weights=None):
        raw, norm = self.signals(cur_df, shap_result)
        dis = self.fuse(norm, weights)
        top = next(iter(dis))
        flips_total = sum(max(r["flip_share"], 0.0) for r in raw.values()) or _EPS
        pred = (c1_output or {}).get("predicted_harm")
        c1_pp = None if pred is None else round(100 * float(pred), 2)
        est = None
        if raw[top]["harm_drop"] is not None:            # root's predicted harm, above its noise
            est = round(raw[top]["harm_drop"] - self.base["harm_drop"][0][top], 2)
        harm_for_rule = est if est is not None else c1_pp
        return {
            "drift_impact_scores": {f: round(v, 4) for f, v in dis.items()},
            "root_cause_feature": top,
            "top_score": round(dis[top], 4),
            "decision_flip_share": round(raw[top]["flip_share"], 4),
            "decision_flip_part": round(raw[top]["flip_share"] / flips_total, 4),
            "estimated_harm_pp": est,                     # root feature, from the harm predictor
            "harm_predictor": self.harm_name,
            "c1_predicted_harm_pp": c1_pp,                # C1's batch-level label-free estimate
            "is_malignant": None if harm_for_rule is None else bool(harm_for_rule > MALIGNANT_PP),
            "impact_signal": "harm drop" if self.harm_fn else "decision change",
            "signals_raw": raw,
            "signals_normalised": norm,
            "weights": tuple(weights or self.weights),
            "labels_used": False,
        }


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.drift_impact_free
    import time
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from c2.src.scenarios import artifacts, make_temporal_stream
    from c2.src.temporal_shap import TemporalSHAPMonitor

    a = artifacts()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    scorers = {"decision change": LabelFreeImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh),
               "harm drop (DoC)": LabelFreeImpactScorer(a["model"], a["ref_df"], a["feature_types"],
                                                        mon.z_thresh, harm_fn=doc_harm(a["model"], a["ref_df"]))}
    for dtype, feats, mag in [("marginal_single", ["capital_gain"], 0.8),
                              ("marginal_single", ["native_country"], 0.6),
                              ("marginal_multi", ["capital_gain", "native_country"], 0.6),
                              ("interaction", ["age", "marital_status"], 0.8),
                              ("none", [], 0.0)]:
        mon.reset()
        stream = make_temporal_stream(dtype, feats, mag, seed=22)
        unlabelled = [s["drifted_df"].drop(columns=[TARGET_COL]) for s in stream]
        for b in unlabelled:
            r = mon.update(b)
        true = stream[-1]["ground_truth"]["actual_accuracy_drop"]
        for name, sc in scorers.items():
            t0 = time.time()
            sc.calibrate(unlabelled[:2])
            out = sc.score(unlabelled[-1], r)
            top3 = [f"{f}:{v:.2f}" for f, v in list(out["drift_impact_scores"].items())[:3]]
            est = out["estimated_harm_pp"]
            print(f"{dtype:<16} {'+'.join(feats) or '-':<28} {name:<16} top3={top3} "
                  f"est harm={'-' if est is None else f'{est:.1f}'} (true {true:.1f}) "
                  f"flips={out['decision_flip_share']:.1%} ({time.time() - t0:.1f}s)")
    print("LABEL-FREE DRIFT IMPACT OK")
