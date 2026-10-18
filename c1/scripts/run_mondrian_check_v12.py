"""
Mondrian check for Novelty 2: does calibrating the conformal interval SEPARATELY per
predicted drift extent fix the under-coverage of marginal_multi?
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

WHY
  N2 (bs_normalised split conformal) reaches 95% coverage overall, but marginal_multi is
  under-covered on both datasets (UCI 0.89 / 0.87, IEEE 0.92 / 0.87). Split conformal only
  guarantees coverage MARGINALLY over the condition mix. A Mondrian (class-conditional)
  conformal predictor computes one quantile per group, which gives coverage per group.

WHICH GROUPS (must be known at inference time)
  The true drift type is NOT known when detect() runs, so it cannot be used. The engine does
  know N3's marginal_extent (none / single / multi). Groups = N3's out-of-fold predicted
  extent (novelty3_final_oof_*.csv). An "oracle" version that uses the TRUE drift type is
  also reported, clearly labelled, only as an upper reference.

METHODS (alpha = 0.05; all use the N2 batch-size normaliser sigma(bs))
  bs_normalised    one quantile for all rows (current engine N2)          [baseline]
  mondrian_extent  one quantile per predicted extent (none/single/multi)  [candidate]
  oracle_type      one quantile per TRUE drift type                       [reference only]
  A group with fewer than MIN_GROUP calibration rows falls back to the global quantile.

PROTOCOL
  Identical to run_novelty2*_v12.py: 100 group-aware splits (StratifiedGroupKFold by drift
  type, 5 folds; fold 0 test, fold 1 calibration, rest train), same GBR point predictor,
  same features, same sigma. Only the calibration step differs.

USAGE (Colab, repo root)
  UCI:  python c1/scripts/run_mondrian_check_v12.py --csv .../benchmark_uci_adult_v12_lsddseeded.csv \
            --oof .../lsdd_seeded/novelty3_final_oof_adult_v12.csv --out_json .../mondrian_adult_v12.json
  IEEE: python c1/scripts/run_mondrian_check_v12.py --csv .../benchmark_ieee_cis_v12.csv \
            --oof .../novelty3_final_oof_ieeecis_v12.csv --config .../benchmark_config_ieeecis_v12.json \
            --out_json .../mondrian_ieeecis_v12.json
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold

DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
FEATS = DET_V12 + CONF + POS + ["batch_size"]
ALPHA = 0.05
MIN_GROUP = 19          # ceil((n+1)(1-alpha)) <= n needs n >= 19 at alpha = 0.05
METHODS = ["bs_normalised", "mondrian_extent", "oracle_type"]
EXT_ORDER = ["none", "single", "multi"]


def make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def conformal_q(scores, alpha=ALPHA):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def group_q(scores, groups_cal, groups_te):
    """Per-group quantile for each test row; global quantile when the group is too small."""
    q_all = conformal_q(scores)
    qs = {g: (conformal_q(scores[groups_cal == g]) if (groups_cal == g).sum() >= MIN_GROUP else q_all)
          for g in np.unique(groups_te)}
    fallback = sum((groups_cal == g).sum() < MIN_GROUP for g in np.unique(groups_te))
    return np.array([qs[g] for g in groups_te]), qs, fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--oof", required=True, help="novelty3_final_oof_*.csv (pred_extent per condition)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--config", default=None, help="IEEE-CIS only: benchmark_config_ieeecis_v12.json")
    ap.add_argument("--test_rows", type=int, default=6513, help="UCI only: test pool size for sigma")
    ap.add_argument("--splits", type=int, default=100)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    oof = pd.read_csv(a.oof)[["condition_id", "pred_extent", "true_extent"]]
    df = df.merge(oof, on="condition_id", how="left", validate="one_to_one")
    assert df["pred_extent"].notna().all(), "OOF file does not cover every condition"
    df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                           df["condition_id"].str.rsplit("_", n=1).str[0])
    auc = "harm_metric" in df.columns and (df["harm_metric"] == "auc_drop").all()
    if auc:
        assert a.config, "AUC benchmark: pass --config (measured clean-batch AUC sd)"
        targets = {"fixed": "actual_auc_drop", "paired": "harm_auc_paired"}
        sd = {int(k): v["auc_sd_pp"] for k, v in json.load(open(a.config))["clean_batch_auc_noise"].items()}
        sig = np.array([sd[int(b)] for b in df["batch_size"]])
    else:
        targets = {"fixed": "actual_accuracy_drop", "paired": "harm_paired"}
        p, N = float(df["fixed_baseline_accuracy"].iloc[0]), a.test_rows
        bs_ = df["batch_size"].values.astype(float)
        sig = 100 * np.sqrt(p * (1 - p) / bs_ * (N - bs_) / (N - 1))
    ds = df["dataset"].iloc[0]
    n, X = len(df), df[FEATS].values.astype(float)
    dt, pe, bs = df["drift_type"].values, df["pred_extent"].values, df["batch_size"].values
    types, sizes = sorted(np.unique(dt)), sorted(np.unique(bs))
    acc = (df["pred_extent"] == df["true_extent"]).mean()
    print(f"{ds} | harm metric {'AUC' if auc else 'accuracy'} drop | N3 OOF extent accuracy {acc:.3f}")
    print("Predicted extent counts:", df["pred_extent"].value_counts().reindex(EXT_ORDER).to_dict())
    print("Predicted extent by true drift type (rows = true):")
    print(pd.crosstab(df["drift_type"], df["pred_extent"]).reindex(columns=EXT_ORDER, fill_value=0).to_string())

    res = {"dataset": ds, "alpha": ALPHA, "splits": a.splits, "min_group": MIN_GROUP,
           "n3_oof_extent_accuracy": float(acc), "targets": {}}
    t0 = time.time()
    for t, tcol in targets.items():
        y, recs, qrec, fb = df[tcol].values, [], [], 0
        for s in range(a.splits):
            folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s).split(df, dt, df["group"]))
            te, ca = folds[0][1], folds[1][1]
            tr = np.setdiff1d(np.arange(n), np.concatenate([te, ca]))
            m = make_model(s).fit(X[tr], y[tr])
            pc, pt = m.predict(X[ca]), m.predict(X[te])
            sc = np.abs(y[ca] - pc) / sig[ca]
            q_glob = conformal_q(sc)
            q_ext, qs_ext, f1 = group_q(sc, pe[ca], pe[te])
            q_typ, _, f2 = group_q(sc, dt[ca], dt[te])
            fb += f1
            qrec.append({"global": q_glob, **{f"ext_{g}": qs_ext.get(g, np.nan) for g in EXT_ORDER}})
            half = {"bs_normalised": q_glob * sig[te], "mondrian_extent": q_ext * sig[te],
                    "oracle_type": q_typ * sig[te]}
            for meth, h in half.items():
                cov = (y[te] >= pt - h) & (y[te] <= pt + h)
                r = {"split": s, "method": meth, "coverage": cov.mean(), "width": (2 * h).mean()}
                for d_ in types:
                    mk = dt[te] == d_
                    r[f"cov_{d_}"], r[f"width_{d_}"] = cov[mk].mean(), (2 * h)[mk].mean()
                for g in EXT_ORDER:
                    mk = pe[te] == g
                    r[f"cov_pred_{g}"] = cov[mk].mean() if mk.any() else np.nan
                for b in sizes:
                    mk = bs[te] == b
                    r[f"cov_bs{int(b)}"] = cov[mk].mean()
                recs.append(r)
            if (s + 1) % 25 == 0:
                print(f"  [{t}] split {s + 1}/{a.splits} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        R, Q = pd.DataFrame(recs), pd.DataFrame(qrec)
        print("\n" + "=" * 88 + f"\nTARGET {t} ({tcol}) | alpha {ALPHA} | {a.splits} group-aware splits\n" + "=" * 88)
        rows = []
        for meth in METHODS:
            x = R[R.method == meth]
            rows.append([meth, f"{x.coverage.mean():.3f} +/- {x.coverage.std():.3f}",
                         f"{(x.coverage >= 1 - ALPHA).mean():.0%}", f"{x.width.mean():.2f}",
                         f"{x[[f'cov_{d_}' for d_ in types]].mean().min():.3f}"])
        print(pd.DataFrame(rows, columns=["method", "coverage", "splits >= 95%", "mean width (pp)",
                                          "worst drift-type coverage"]).to_string(index=False))
        print("\nCoverage by TRUE drift type:")
        print(R.groupby("method")[[f"cov_{d_}" for d_ in types]].mean().reindex(METHODS).T.round(3).to_string())
        print("\nMean width (pp) by TRUE drift type:")
        print(R.groupby("method")[[f"width_{d_}" for d_ in types]].mean().reindex(METHODS).T.round(2).to_string())
        print("\nCoverage by PREDICTED extent (the groups mondrian_extent calibrates on):")
        print(R.groupby("method")[[f"cov_pred_{g}" for g in EXT_ORDER]].mean().reindex(METHODS).T.round(3).to_string())
        print("\nCoverage by batch size:")
        print(R.groupby("method")[[f"cov_bs{int(b)}" for b in sizes]].mean().reindex(METHODS).T.round(3).to_string())
        print(f"\nNormalised quantile q (mean over splits): global {Q['global'].mean():.3f} | "
              + " | ".join(f"{g} {Q[f'ext_{g}'].mean():.3f}" for g in EXT_ORDER)
              + f"   (fallbacks to global: {fb} group-splits)")
        res["targets"][t] = {"summary": {meth: {k: [float(R[R.method == meth][k].mean()), float(R[R.method == meth][k].std())]
                                                for k in R.columns if k not in ("split", "method")} for meth in METHODS},
                             "q_mean": {k: float(Q[k].mean()) for k in Q.columns}, "fallbacks": int(fb)}

    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(res, open(a.out_json, "w"), indent=2)
    print(f"\nSaved: {a.out_json}   Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
