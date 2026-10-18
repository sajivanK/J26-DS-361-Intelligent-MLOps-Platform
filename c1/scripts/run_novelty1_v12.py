"""
Novelty 1 on benchmark v1.2 (UCI Adult).
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Question: with fairly built detectors (v1.2), do detector signals add value on top
of the model's own prediction statistics when predicting accuracy drop?

Protocol (handoff section 9):
  groups: one per drift condition (all seeds together); each no-drift row is its own group
  StratifiedGroupKFold by drift_type, 5 folds x 10 repeats, mean +/- sd over repeats
  GBR(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=5)
  batch_size is an input for every learned config except all_v12_no_batchsize
  targets: fixed (actual_accuracy_drop, primary) and paired (harm_paired)
  no p-values from repeated CV (repeats are not independent)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

DET_V11 = ["v11_ks_score", "v11_chi2_score", "v11_psi_score", "v11_mmd_score", "v11_lsdd_score", "v11_clf_auc"]
DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]

CONFIGS = {  # name: (input columns, add batch_size)
    "constant_mean":        (None, False),
    "ks_only_v11":          (["v11_ks_score"], True),
    "detectors_v11":        (DET_V11, True),
    "detectors_v12":        (DET_V12, True),
    "posprob_3":            (POS, True),
    "conf_3":               (CONF, True),
    "conf_plus_posprob":    (CONF + POS, True),
    "all_v11":              (DET_V11 + POS, True),
    "all_v12":              (DET_V12 + CONF, True),
    "all_v12_no_batchsize": (DET_V12 + CONF, False),
    "everything":           (DET_V12 + CONF + POS, True),
}
TARGETS = {"fixed": "actual_accuracy_drop", "paired": "harm_paired"}
KEY = ["constant_mean", "detectors_v12", "conf_3", "all_v12", "everything"]
PAIRS = [
    ("all_v12", "conf_3", "N1 claim: v1.2 detectors + confidence vs confidence alone"),
    ("everything", "conf_plus_posprob", "detectors on top of ALL prediction stats"),
    ("detectors_v12", "detectors_v11", "fair detectors vs old detectors"),
    ("conf_3", "posprob_3", "max-prob confidence vs positive-class stats"),
    ("all_v12", "all_v11", "new full model vs old full model"),
]
MAL = 2.0  # malignant threshold (pp)


def make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def X_of(df, cols, bs):
    return df[cols + (["batch_size"] if bs else [])].values.astype(float)


def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)),
            "mal_f1": float(f1_score(y > MAL, p > MAL, zero_division=0))}


def touched_features(shared_dir, imp, dataset):
    """All features each condition touches (marginal AND interaction-pair features)."""
    sys.path.insert(0, os.path.abspath(shared_dir))
    import benchmark_loop as bl
    hi = imp["feature_groups"]["high_importance"]["features"]
    lo = imp["feature_groups"]["low_importance"]["features"]
    out = {}
    for c in bl.build_condition_list(hi, lo, imp["feature_types"], dataset):
        fs = set(c["features"])
        for a, b in (c.get("interaction_pairs") or []):
            fs |= {a, b}
        out[c["condition_id"]] = fs
    return out, hi[:5] + lo[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--feature_json", required=True)
    ap.add_argument("--shared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test_rows", type=int, default=6513)
    a = ap.parse_args()
    R, K = a.repeats, a.folds

    df = pd.read_csv(a.csv)
    need = sorted({c for cols, _ in CONFIGS.values() if cols for c in cols} | {"batch_size"} | set(TARGETS.values()))
    assert len(df) == 1600, f"expected 1600 rows, got {len(df)}"
    assert (df["benchmark_version"].astype(str) == "1.2").all(), "not a v1.2 benchmark"
    assert df[need].isna().sum().sum() == 0, "NaN in model inputs"
    df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                           df["condition_id"].str.rsplit("_", n=1).str[0])
    n = len(df)
    print(f"Rows {n} | groups {df['group'].nunique()} (expected 480) | repeats {R} x folds {K}")
    print("Mean fixed drop by drift type (pp):", df.groupby("drift_type")["actual_accuracy_drop"].mean().round(2).to_dict())

    p = float(df["fixed_baseline_accuracy"].iloc[0]); N = a.test_rows
    floor = {int(bs): float(0.798 * 100 * np.sqrt(p * (1 - p) / bs * (N - bs) / (N - 1)))
             for bs in sorted(df["batch_size"].unique())}

    # ── Repeated grouped CV ──────────────────────────────────────────────
    oof = {t: {c: np.zeros((R, n)) for c in CONFIGS} for t in TARGETS}
    t0 = time.time()
    for r in range(R):
        folds = list(StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=r)
                     .split(df, df["drift_type"], df["group"]))
        for t, tcol in TARGETS.items():
            y = df[tcol].values
            for c, (cols, bs) in CONFIGS.items():
                X = None if cols is None else X_of(df, cols, bs)
                for k, (tr, te) in enumerate(folds):
                    if X is None:
                        oof[t][c][r, te] = y[tr].mean()
                    else:
                        oof[t][c][r, te] = make_model(1000 * r + k).fit(X[tr], y[tr]).predict(X[te])
        print(f"  repeat {r + 1}/{R} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = {"protocol": {"repeats": R, "folds": K, "groups": int(df["group"].nunique()),
                        "malignant_threshold_pp": MAL, "benchmark": os.path.basename(a.csv)},
           "noise_floor_fixed_mae_pp": floor, "summary": {}, "by_type": {}, "by_batch": {},
           "pairs": {}, "lofo": {}, "importance": {}}

    for t, tcol in TARGETS.items():
        y = df[tcol].values
        print("\n" + "=" * 78 + f"\nTARGET: {t} ({tcol})   mean +/- sd over {R} repeats\n" + "=" * 78)
        rows, res["summary"][t] = [], {}
        for c in CONFIGS:
            per = [metrics(y, oof[t][c][r]) for r in range(R)]
            s = {m: [float(np.mean([d[m] for d in per])), float(np.std([d[m] for d in per]))] for m in per[0]}
            res["summary"][t][c] = s
            rows.append([c, f"{s['mae'][0]:.3f} +/- {s['mae'][1]:.3f}", f"{s['rmse'][0]:.3f}",
                         f"{s['r2'][0]:.3f}", f"{s['mal_f1'][0]:.3f}"])
        print(pd.DataFrame(rows, columns=["config", "MAE (pp)", "RMSE", "R2", "mal_F1"]).to_string(index=False))

        res["by_type"][t] = {c: {dt: float(np.mean([mean_absolute_error(y[m], oof[t][c][r][m]) for r in range(R)]))
                                 for dt in sorted(df["drift_type"].unique())
                                 for m in [(df["drift_type"] == dt).values]} for c in KEY}
        print("\nMAE by drift type:")
        print(pd.DataFrame(res["by_type"][t]).round(3).to_string())

        res["by_batch"][t] = {c: {int(bs): float(np.mean([mean_absolute_error(y[m], oof[t][c][r][m]) for r in range(R)]))
                                  for bs in sorted(df["batch_size"].unique())
                                  for m in [(df["batch_size"] == bs).values]} for c in KEY}
        tb = pd.DataFrame(res["by_batch"][t]).round(3)
        if t == "fixed":
            tb["noise_floor"] = pd.Series(floor).round(3)
        print("\nMAE by batch size" + (" (noise_floor = MAE expected from sampling noise alone):" if t == "fixed" else ":"))
        print(tb.to_string())

        print("\nPaired comparisons (A minus B; negative = A better):")
        res["pairs"][t] = {}
        for A, B, why in PAIRS:
            d = np.array([res_mae(y, oof[t][A][r]) - res_mae(y, oof[t][B][r]) for r in range(R)])
            eA = np.abs(y - oof[t][A]).mean(0); eB = np.abs(y - oof[t][B]).mean(0)
            res["pairs"][t][f"{A}__vs__{B}"] = {"why": why, "mae_diff_mean": float(d.mean()),
                                               "mae_diff_sd": float(d.std()), "repeats_A_better": int((d < 0).sum()),
                                               "share_conditions_A_better": float((eA < eB).mean())}
            print(f"  {A} vs {B}: {d.mean():+.3f} +/- {d.std():.3f} pp | A better in {(d < 0).sum()}/{R} repeats"
                  f" | A better on {(eA < eB).mean():.0%} of conditions   [{why}]")

    # ── Leave-one-feature-out stress test (all 10 features) ──────────────
    imp = json.load(open(a.feature_json))
    touched, feats = touched_features(a.shared_dir, imp, df["dataset"].iloc[0])
    tf = df["condition_id"].map(touched)
    assert tf.notna().all(), "condition_id not found in condition list"
    for t, tcol in TARGETS.items():
        y = df[tcol].values
        res["lofo"][t], rows = {}, []
        for f in feats:
            test = tf.apply(lambda s: f in s).values
            tr = ~test
            res["lofo"][t][f] = {"n": int(test.sum()), "true_mean": float(y[test].mean())}
            row = [f, int(test.sum()), f"{y[test].mean():.2f}"]
            for c in KEY:
                cols, bs = CONFIGS[c]
                if cols is None:
                    pr = np.full(test.sum(), y[tr].mean())
                else:
                    X = X_of(df, cols, bs)
                    pr = make_model(0).fit(X[tr], y[tr]).predict(X[test])
                res["lofo"][t][f][c] = {"mae": float(mean_absolute_error(y[test], pr)), "pred_mean": float(pr.mean())}
                row.append(f"{mean_absolute_error(y[test], pr):.2f}")
            rows.append(row)
        print("\n" + "=" * 78 + f"\nLEAVE-ONE-FEATURE-OUT ({t}): train WITHOUT any condition touching the feature,"
              f"\ntest on those conditions. Cells = MAE (pp).\n" + "=" * 78)
        print(pd.DataFrame(rows, columns=["held_out", "n", "true_mean"] + KEY).to_string(index=False))

    # ── Importances (full fit, fixed target; impurity-based, rough guide only) ──
    y = df[TARGETS["fixed"]].values
    for c in ["all_v12", "everything"]:
        cols, bs = CONFIGS[c]
        m = make_model(0).fit(X_of(df, cols, bs), y)
        names = cols + (["batch_size"] if bs else [])
        res["importance"][c] = dict(sorted(zip(names, map(float, m.feature_importances_)), key=lambda x: -x[1]))
    print("\nGBR importances (all_v12, fixed target):")
    for k, v in res["importance"]["all_v12"].items():
        print(f"  {k:18s} {v:.3f}")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "novelty1_results_adult_v12.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved: {out}   Total time {(time.time() - t0) / 60:.1f} min")


def res_mae(y, p):
    return float(mean_absolute_error(y, p))


if __name__ == "__main__":
    main()
