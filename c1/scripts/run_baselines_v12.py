"""
Novelty 1 vs label-free baselines (ATC, DoC, CBPE) on benchmark v1.2 (UCI Adult).
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Baseline definitions (read from the sources for this implementation):
  ATC  Garg et al., ICLR 2022 (arXiv 2201.04234), Eq. 1-2.
       Score s = max confidence (ATC-MC) or negative entropy sum p log p (ATC-NE).
       Threshold t on held-out source data: share of source points with s < t equals source error.
       Target accuracy = share of target points with s >= t.
  DoC  Guillory et al., ICCV 2021 (arXiv 2107.03315).
       DoC = mean max-prob on source minus mean max-prob on target.
       DoC-Feat: predicted drop = DoC. DoC-Reg: linear regression DoC -> drop fitted on synthetic
       shifts (here: the injection training folds, same CV folds as N1).
  CBPE NannyML documentation ("how it works", performance estimation), binary accuracy.
       Isotonic calibrator fitted on labelled reference data, used only if it lowers ECE
       (3 stratified splits). Expected accuracy = mean(1 - |y_pred - p_calibrated|).
       Own implementation of the documented algorithm (not the nannyml package).
Held-out labelled source data = test.csv (the data that defines the fixed baseline).

Evaluation: same grouped CV as N1 (5 folds x R repeats) for learned methods.
Paired test: Wilcoxon signed-rank on per-GROUP mean absolute error (one value per drift
condition group, errors averaged over repeats), N1 vs each baseline on identical conditions.
"""
import argparse, json, os, sys, time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
N1 = {"N1_pred_stats_only": CONF + POS, "N1_full": DET_V12 + CONF + POS}
BASELINES = ["constant_mean", "ATC_MC", "ATC_NE", "DoC_Feat", "DoC_Reg", "CBPE"]
TARGETS = {"fixed": "actual_accuracy_drop", "paired": "harm_paired"}
MAL = 2.0
EPS = 1e-12


def make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def neg_entropy(P):
    P = np.clip(P, EPS, 1.0)
    return (P * np.log(P)).sum(axis=1)


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


class CBPE:
    def __init__(self, p_ref, y_ref, seed=42):
        raw, cal = [], []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(p_ref, y_ref):
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_ref[tr], y_ref[tr])
            raw.append(ece(y_ref[te], p_ref[te]))
            cal.append(ece(y_ref[te], iso.predict(p_ref[te])))
        self.ece_raw, self.ece_cal = float(np.mean(raw)), float(np.mean(cal))
        self.use_cal = self.ece_cal < self.ece_raw
        self.iso = (IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_ref, y_ref)
                    if self.use_cal else None)

    def accuracy(self, p, yhat):
        pc = self.iso.predict(p) if self.use_cal else p
        return float(np.mean(1.0 - np.abs(yhat - pc)))


def main():
    ap = argparse.ArgumentParser()
    for k in ["shared_dir", "ref", "test", "model", "feature_json", "csv", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    R, tc = a.repeats, "target"

    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    from injection.inject_drift import inject_drift

    ref, test = pd.read_csv(a.ref), pd.read_csv(a.test)
    model = joblib.load(a.model)
    imp = json.load(open(a.feature_json))
    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    hi = imp["feature_groups"]["high_importance"]["features"]
    lo = imp["feature_groups"]["low_importance"]["features"]
    conds = {c["condition_id"]: c for c in
             bl.build_condition_list(hi, lo, imp["feature_types"], df["dataset"].iloc[0])}
    fixed = float(df["fixed_baseline_accuracy"].iloc[0])

    # ── Source (held-out labelled) statistics ────────────────────────────
    Xs, ys = test.drop(columns=[tc]), test[tc].values
    Ps, yhs = model.predict_proba(Xs), model.predict(Xs)
    acc_s = float((yhs == ys).mean())
    assert abs(round(acc_s, 4) - fixed) < 1.01e-4, f"source accuracy {acc_s} != fixed baseline {fixed}"
    s_mc, s_ne = Ps.max(1), neg_entropy(Ps)
    t_mc, t_ne = float(np.quantile(s_mc, 1 - acc_s)), float(np.quantile(s_ne, 1 - acc_s))
    ac_src = float(s_mc.mean())
    cbpe = CBPE(Ps[:, 1], ys)
    print(f"Source (test.csv): accuracy {acc_s:.4f} | mean max-prob {ac_src:.4f}")
    print(f"ATC thresholds: MC {t_mc:.4f}, NE {t_ne:.4f} | source ATC estimate MC "
          f"{np.mean(s_mc >= t_mc):.4f}, NE {np.mean(s_ne >= t_ne):.4f}")
    print(f"CBPE: ECE raw {cbpe.ece_raw:.4f}, calibrated {cbpe.ece_cal:.4f} -> calibrate = {cbpe.use_cal}; "
          f"source estimate {cbpe.accuracy(Ps[:, 1], yhs):.4f}")

    # ── Phase 1: baseline predictions per condition (batches recreated) ──
    os.makedirs(a.out_dir, exist_ok=True)
    pred_path = os.path.join(a.out_dir, "baseline_preds_adult_v12.csv")
    if os.path.exists(pred_path) and not a.recompute:
        preds = pd.read_csv(pred_path)
        print(f"Loaded existing baseline predictions: {pred_path}")
    else:
        rows, t0 = [], time.time()
        for i, (_, r) in enumerate(df.iterrows()):
            c = conds[r["condition_id"]]
            res = inject_drift(reference_df=ref, model=model, drift_type=c["drift_type"],
                               features=c["features"], magnitude=c["magnitude"],
                               feature_types=c.get("feature_types", {}),
                               interaction_pairs=c.get("interaction_pairs"),
                               batch_size=c["batch_size"], target_col=tc,
                               seed=int(r["batch_seed"]), source_df=test, paired=True)
            X = res["drifted_df"].drop(columns=[tc])
            P, yh = model.predict_proba(X), model.predict(X)
            sm, sn = P.max(1), neg_entropy(P)
            if (abs(round(res["ground_truth"]["drifted_accuracy"], 4) - r["drifted_accuracy"]) > 1.01e-4 or
                    abs(round(float(sm.mean()), 6) - r["conf_mean"]) > 1.01e-6):
                raise SystemExit(f"STOP: batch for {r['condition_id']} does not match the v1.2 CSV.")
            rows.append({"condition_id": r["condition_id"],
                         "ATC_MC": (fixed - np.mean(sm >= t_mc)) * 100,
                         "ATC_NE": (fixed - np.mean(sn >= t_ne)) * 100,
                         "doc": ac_src - sm.mean(),
                         "DoC_Feat": (ac_src - sm.mean()) * 100,
                         "CBPE": (fixed - cbpe.accuracy(P[:, 1], yh)) * 100})
            if (i + 1) % 200 == 0:
                print(f"  baselines {i + 1}/1600 ({(time.time() - t0) / 60:.1f} min)", flush=True)
        preds = pd.DataFrame(rows)
        preds.to_csv(pred_path, index=False)
        print(f"Batches reproduced for all 1600 conditions. Saved {pred_path}")

    # ── Phase 2: evaluation ──────────────────────────────────────────────
    d = df.merge(preds, on="condition_id", validate="one_to_one")
    d["group"] = np.where(d["drift_type"] == "none", d["condition_id"],
                          d["condition_id"].str.rsplit("_", n=1).str[0])
    touched = {}
    for cid, c in conds.items():
        fs = set(c["features"])
        for x, z in (c.get("interaction_pairs") or []):
            fs |= {x, z}
        touched[cid] = fs
    pm = d["condition_id"].map(touched).apply(lambda s: bool(s & {"capital_gain", "capital_loss"})).values
    n = len(d)
    Xn1 = {k: d[cols + ["batch_size"]].values.astype(float) for k, cols in N1.items()}
    Xdoc = d[["doc"]].values
    results = {"source": {"accuracy": acc_s, "atc_t_mc": t_mc, "atc_t_ne": t_ne, "ac_src": ac_src,
                          "cbpe_calibrated": bool(cbpe.use_cal)}, "targets": {}}
    t0 = time.time()
    for t, tcol in TARGETS.items():
        y = d[tcol].values
        P = {m: np.tile(d[m].values.astype(float), (R, 1)) for m in ["ATC_MC", "ATC_NE", "DoC_Feat", "CBPE"]}
        for m in ["constant_mean", "DoC_Reg"] + list(N1):
            P[m] = np.zeros((R, n))
        for r in range(R):
            folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r).split(d, d["drift_type"], d["group"])
            for k, (tr, te) in enumerate(folds):
                P["constant_mean"][r, te] = y[tr].mean()
                P["DoC_Reg"][r, te] = LinearRegression().fit(Xdoc[tr], y[tr]).predict(Xdoc[te])
                for m in N1:
                    P[m][r, te] = make_model(1000 * r + k).fit(Xn1[m][tr], y[tr]).predict(Xn1[m][te])
        print(f"  [{t}] CV done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        order = BASELINES + list(N1)
        results["targets"][t] = {}
        for label, idx in [("ALL conditions", np.ones(n, bool)),
                           ("WITHOUT capital_gain / capital_loss conditions", ~pm)]:
            res_t = {}
            print("\n" + "=" * 86 + f"\nTARGET {t} ({tcol}) | {label} | {idx.sum()} rows\n" + "=" * 86)
            rows = []
            for m in order:
                maes = [mean_absolute_error(y[idx], P[m][r][idx]) for r in range(R)]
                r2s = [r2_score(y[idx], P[m][r][idx]) for r in range(R)]
                f1s = [f1_score(y[idx] > MAL, P[m][r][idx] > MAL, zero_division=0) for r in range(R)]
                res_t[m] = {"mae": [float(np.mean(maes)), float(np.std(maes))],
                            "r2": float(np.mean(r2s)), "mal_f1": float(np.mean(f1s))}
                rows.append([m, f"{np.mean(maes):.3f} +/- {np.std(maes):.3f}", f"{np.mean(r2s):.3f}", f"{np.mean(f1s):.3f}"])
            print(pd.DataFrame(rows, columns=["method", "MAE (pp)", "R2", "mal_F1"]).to_string(index=False))

            E = {m: np.abs(y[idx] - P[m][:, idx]).mean(0) for m in order}
            g = d["group"].values[idx]
            G = {m: pd.Series(E[m]).groupby(g).mean() for m in order}
            print(f"\nPaired test vs N1_full: Wilcoxon signed-rank on {len(G['N1_full'])} condition groups")
            res_t["wilcoxon_vs_N1_full"] = {}
            for b in BASELINES + ["N1_pred_stats_only"]:
                diff = (G["N1_full"] - G[b]).values
                p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                res_t["wilcoxon_vs_N1_full"][b] = {"median_diff": float(np.median(diff)),
                                                   "share_groups_N1_better": float((diff < 0).mean()), "p_value": p}
                print(f"  N1_full vs {b:18s} median diff {np.median(diff):+.3f} pp | "
                      f"N1 better on {(diff < 0).mean():.0%} of groups | p = {p:.2e}")
            best_atc = min(res_t["ATC_MC"]["mae"][0], res_t["ATC_NE"]["mae"][0])
            imp_atc = (best_atc - res_t["N1_full"]["mae"][0]) / best_atc * 100
            res_t["improvement_over_best_ATC_pct"] = imp_atc
            print(f"\nImprovement of N1_full over best ATC: {imp_atc:.1f}% (proposal target: above 30%)")
            bt = {m: {dt: float(mean_absolute_error(y[idx & (d['drift_type'] == dt).values],
                                                    P[m].mean(0)[idx & (d['drift_type'] == dt).values]))
                      for dt in sorted(d["drift_type"].unique())} for m in order}
            res_t["mae_by_type"] = bt
            print("\nMAE by drift type (predictions averaged over repeats):")
            print(pd.DataFrame(bt).round(3).to_string())
            results["targets"][t][label] = res_t

    out = os.path.join(a.out_dir, "baselines_results_adult_v12.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
