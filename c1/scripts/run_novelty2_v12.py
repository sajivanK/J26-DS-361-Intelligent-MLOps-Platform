"""
Novelty 2 on benchmark v1.2 (UCI Adult): split conformal bounds on N1's predicted accuracy drop.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Point predictor: N1 full config (9 detector signals + 3 confidence + 3 positive-class stats
+ batch_size), GBR with the N1 hyperparameters.
Methods (alpha = 0.05, target 95% coverage):
  plain          score |y - yhat|,               interval yhat +/- q
  bs_normalised  score |y - yhat| / sigma(bs),   interval yhat +/- q * sigma(bs)
                 sigma(bs) = sampling sd of batch accuracy at batch size bs (finite-population
                 corrected, p = fixed baseline accuracy, N = test pool). Batch size is known at
                 inference, so this is a legitimate normaliser.
Protocol: S group-aware splits (StratifiedGroupKFold by drift_type, 5 folds; fold 0 = test,
fold 1 = calibration, rest = train). Groups = drift conditions (all seeds), so test conditions
are never seen in training or calibration. The guarantee is MARGINAL over the condition mix.
Shift test: leave-one-feature-out (calibrate without any condition touching the feature,
test on those conditions). Exchangeability is violated there by design.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold

DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
FEATS = DET_V12 + CONF + POS + ["batch_size"]
TARGETS = {"fixed": "actual_accuracy_drop", "paired": "harm_paired"}
ALPHA = 0.05
METHODS = ["plain", "bs_normalised"]


def make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def conformal_q(scores, alpha=ALPHA):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def intervals(y_cal, p_cal, s_cal, p_te, s_te):
    qa = conformal_q(np.abs(y_cal - p_cal))
    qb = conformal_q(np.abs(y_cal - p_cal) / s_cal)
    return {"plain": (p_te - qa, p_te + qa), "bs_normalised": (p_te - qb * s_te, p_te + qb * s_te)}


def main():
    ap = argparse.ArgumentParser()
    for k in ["csv", "feature_json", "shared_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--lofo_seeds", type=int, default=5)
    ap.add_argument("--test_rows", type=int, default=6513)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                           df["condition_id"].str.rsplit("_", n=1).str[0])
    n = len(df)
    X = df[FEATS].values.astype(float)
    p, N = float(df["fixed_baseline_accuracy"].iloc[0]), a.test_rows
    bs = df["batch_size"].values.astype(float)
    sig = 100 * np.sqrt(p * (1 - p) / bs * (N - bs) / (N - 1))
    types = sorted(df["drift_type"].unique())
    sizes = sorted(df["batch_size"].unique())
    dt = df["drift_type"].values
    res = {"alpha": ALPHA, "splits": a.splits, "sigma_by_batch_pp": {int(b): float(sig[bs == b][0]) for b in sizes},
           "targets": {}, "lofo": {}}
    t0 = time.time()

    for t, tcol in TARGETS.items():
        y = df[tcol].values
        recs = []
        for s in range(a.splits):
            folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s).split(df, dt, df["group"]))
            te, ca = folds[0][1], folds[1][1]
            tr = np.setdiff1d(np.arange(n), np.concatenate([te, ca]))
            m = make_model(s).fit(X[tr], y[tr])
            for meth, (lo, hi) in intervals(y[ca], m.predict(X[ca]), sig[ca], m.predict(X[te]), sig[te]).items():
                cov = (y[te] >= lo) & (y[te] <= hi)
                r = {"split": s, "method": meth, "coverage": cov.mean(), "width": (hi - lo).mean(),
                     "coverage_lower_clipped_at_0": ((y[te] >= np.maximum(lo, 0)) & (y[te] <= hi)).mean()}
                for d_ in types:
                    mk = dt[te] == d_
                    r[f"cov_{d_}"] = cov[mk].mean() if mk.any() else np.nan
                for b in sizes:
                    mk = bs[te] == b
                    r[f"cov_bs{int(b)}"] = cov[mk].mean() if mk.any() else np.nan
                    r[f"width_bs{int(b)}"] = (hi - lo)[mk].mean() if mk.any() else np.nan
                recs.append(r)
            if (s + 1) % 25 == 0:
                print(f"  [{t}] split {s + 1}/{a.splits} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        R = pd.DataFrame(recs)
        g = R.groupby("method")
        print("\n" + "=" * 80 + f"\nTARGET {t} ({tcol}) | alpha {ALPHA} | {a.splits} group-aware splits\n" + "=" * 80)
        rows = []
        for meth in METHODS:
            x = R[R["method"] == meth]
            rows.append([meth, f"{x['coverage'].mean():.3f} +/- {x['coverage'].std():.3f}",
                         f"{(x['coverage'] >= 1 - ALPHA).mean():.0%}",
                         f"{x['coverage_lower_clipped_at_0'].mean():.3f}", f"{x['width'].mean():.2f}"])
        print(pd.DataFrame(rows, columns=["method", "coverage", "splits >= 95%", "cov if lower clipped at 0",
                                          "mean width (pp)"]).to_string(index=False))
        print("\nCoverage by drift type (mean over splits):")
        print(g[[f"cov_{d_}" for d_ in types]].mean().T.round(3).to_string())
        print("\nCoverage and width by batch size (mean over splits):")
        bt = pd.DataFrame({meth: {**{f"cov_{int(b)}": R[R.method == meth][f"cov_bs{int(b)}"].mean() for b in sizes},
                                  **{f"width_{int(b)}": R[R.method == meth][f"width_bs{int(b)}"].mean() for b in sizes}}
                           for meth in METHODS})
        print(bt.round(3).to_string())
        res["targets"][t] = {meth: {k: [float(R[R.method == meth][k].mean()), float(R[R.method == meth][k].std())]
                                    for k in R.columns if k not in ("split", "method")} for meth in METHODS}

    # ── Leave-one-feature-out shift test ─────────────────────────────────
    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    imp = json.load(open(a.feature_json))
    hi_f = imp["feature_groups"]["high_importance"]["features"][:5]
    lo_f = imp["feature_groups"]["low_importance"]["features"][:5]
    touched = {}
    for c in bl.build_condition_list(hi_f, lo_f, imp["feature_types"], df["dataset"].iloc[0]):
        fs = set(c["features"])
        for x_, z_ in (c.get("interaction_pairs") or []):
            fs |= {x_, z_}
        touched[c["condition_id"]] = fs
    tf = df["condition_id"].map(touched)
    for t, tcol in TARGETS.items():
        y = df[tcol].values
        res["lofo"][t], rows = {}, []
        for f in hi_f + lo_f:
            test = tf.apply(lambda s_: f in s_).values
            rest = np.where(~test)[0]
            covs = {meth: [] for meth in METHODS}
            wids = {meth: [] for meth in METHODS}
            for s in range(a.lofo_seeds):
                sub = df.iloc[rest]
                fo = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
                          .split(sub, sub["drift_type"], sub["group"]))
                ca = rest[fo[0][1]]
                tr = np.setdiff1d(rest, ca)
                m = make_model(s).fit(X[tr], y[tr])
                te = np.where(test)[0]
                for meth, (lo, hi) in intervals(y[ca], m.predict(X[ca]), sig[ca], m.predict(X[te]), sig[te]).items():
                    covs[meth].append(((y[te] >= lo) & (y[te] <= hi)).mean())
                    wids[meth].append((hi - lo).mean())
            res["lofo"][t][f] = {"n": int(test.sum()), "true_mean": float(y[test].mean()),
                                 **{f"coverage_{m_}": float(np.mean(covs[m_])) for m_ in METHODS},
                                 **{f"width_{m_}": float(np.mean(wids[m_])) for m_ in METHODS}}
            rows.append([f, int(test.sum()), f"{y[test].mean():.2f}"] +
                        [f"{np.mean(covs[m_]):.3f}" for m_ in METHODS] + [f"{np.mean(wids[m_]):.2f}" for m_ in METHODS])
        print("\n" + "=" * 80 + f"\nLEAVE-ONE-FEATURE-OUT ({t}): calibrate without the feature, test on it\n" + "=" * 80)
        print(pd.DataFrame(rows, columns=["held_out", "n", "true_mean", "cov_plain", "cov_bsnorm",
                                          "width_plain", "width_bsnorm"]).to_string(index=False))

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "novelty2_results_adult_v12.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved: {out}   Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
