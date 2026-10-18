"""
Novelty 1 vs label-free baselines (ATC, DoC, CBPE) on benchmark v1.2 (IEEE-CIS Fraud), AUC-drop target.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Same evaluation protocol as run_baselines_v12.py (UCI Adult): grouped CV (5 folds x R repeats, the
N1 folds), Wilcoxon signed-rank on per-condition-group mean absolute error, N1 vs each baseline on
identical conditions. Held-out labelled source data = test.csv (defines the fixed baseline).

ADAPTATION TO AUC (harm on IEEE-CIS is AUC drop; ATC and DoC were designed to estimate ACCURACY)
  CBPE      NannyML's CBPE also estimates ROC AUC: calibrated probabilities are used as expected
            labels, giving an expected ROC curve (expected TPR / FPR over thresholds) and its area.
            Isotonic calibrator on source data, used only if it lowers ECE (as on UCI).
            CBPE       predicted drop = (fixed source AUC - CBPE AUC estimate on batch) x 100   [UCI-style]
            CBPE_rel   predicted drop = (CBPE estimate on source - CBPE estimate on batch) x 100
                       (removes any constant bias of the estimator; the more charitable version)
            OWN IMPLEMENTATION of the documented idea, not the nannyml package.
  ATC_Reg   ATC (Garg et al. 2022) estimates batch ACCURACY. Its estimated accuracy change is mapped to
            AUC drop by a linear regression fitted on the training folds (same folds as N1).
            ATC-MC only: for a two-class model ATC-NE ranks rows identically (shown on UCI).
  DoC_Reg   DoC (Guillory et al. 2021) = mean max-prob on source minus on batch, mapped to AUC drop by
            linear regression on the training folds (the DoC-Reg variant, as on UCI).
  ATC raw and DoC-Feat are not reported: they are accuracy estimates in accuracy points and cannot be
  compared with an AUC target without a mapping.
Views: ALL conditions, and WITHOUT low-importance conditions (their harm is 0, which is easy for everyone).
"""
import argparse, json, os, sys, time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
N1 = {"N1_pred_stats_only": CONF + POS, "N1_full": DET_V12 + CONF + POS}
BASELINES = ["constant_mean", "ATC_Reg", "DoC_Reg", "CBPE", "CBPE_rel"]
TARGETS = {"fixed": "actual_auc_drop", "paired": "harm_auc_paired"}
MAL = 2.0
EPS = 1e-12


def make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def expected_auc(score, p):
    """Expected ROC AUC when each row is positive with probability p (CBPE idea).
    Rows ranked by the model score; expected TPR / FPR accumulate p and 1 - p; trapezoid area."""
    o = np.argsort(-score, kind="mergesort")
    s, pp = score[o], p[o]
    last = np.r_[np.diff(s) != 0, True]               # end of each tied-score block
    tp = np.cumsum(pp)[last]
    fp = np.cumsum(1.0 - pp)[last]
    tpr = np.r_[0.0, tp / max(tp[-1], EPS)]
    fpr = np.r_[0.0, fp / max(fp[-1], EPS)]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


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

    def auc(self, p):
        pc = self.iso.predict(p) if self.use_cal else p
        return expected_auc(p, pc)


def main():
    ap = argparse.ArgumentParser()
    for k in ["shared_dir", "ref", "test", "model", "feature_json", "config", "csv", "out_dir"]:
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
    cfg = json.load(open(a.config))
    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    assert (df["harm_metric"] == "auc_drop").all(), "not an AUC-drop benchmark"
    hi = imp["feature_groups"]["high_importance"]["features"]
    lo = imp["feature_groups"]["low_importance"]["features"]
    conds = {c["condition_id"]: c for c in
             bl.build_condition_list(hi, lo, imp["feature_types"], df["dataset"].iloc[0],
                                     batch_sizes=cfg["batch_sizes"])}
    fixed_auc = float(df["fixed_baseline_auc"].iloc[0])
    fixed_acc = float(df["fixed_baseline_accuracy"].iloc[0])

    # ── Source (held-out labelled) statistics ────────────────────────────
    Xs, ys = test.drop(columns=[tc]), test[tc].values
    Ps, yhs = model.predict_proba(Xs), model.predict(Xs)
    acc_s, auc_s = float((yhs == ys).mean()), float(roc_auc_score(ys, Ps[:, 1]))
    assert abs(round(auc_s, 4) - fixed_auc) < 1.01e-4, f"source AUC {auc_s} != fixed baseline {fixed_auc}"
    s_mc = Ps.max(1)
    t_mc = float(np.quantile(s_mc, 1 - acc_s))
    ac_src = float(s_mc.mean())
    # self-check: with the TRUE labels as p, the expected AUC must equal the real AUC
    chk = expected_auc(Ps[:, 1], ys.astype(float))
    assert abs(chk - auc_s) < 1e-9, f"expected_auc self-check failed: {chk} vs {auc_s}"
    print(f"expected_auc self-check passed ({chk:.6f} = sklearn {auc_s:.6f})")
    cbpe = CBPE(Ps[:, 1], ys)
    cbpe_src = cbpe.auc(Ps[:, 1])
    print(f"Source (test.csv): AUC {auc_s:.4f} | accuracy {acc_s:.4f} | mean max-prob {ac_src:.4f}")
    print(f"ATC threshold (MC) {t_mc:.4f} | source ATC accuracy estimate {np.mean(s_mc >= t_mc):.4f}")
    print(f"CBPE: ECE raw {cbpe.ece_raw:.4f}, calibrated {cbpe.ece_cal:.4f} -> calibrate = {cbpe.use_cal}; "
          f"source AUC estimate {cbpe_src:.4f} (true {auc_s:.4f}, bias {100 * (cbpe_src - auc_s):+.2f} pp)")

    # ── Phase 1: baseline signals per condition (batches recreated) ─────
    os.makedirs(a.out_dir, exist_ok=True)
    pred_path = os.path.join(a.out_dir, "baseline_preds_ieeecis_v12.csv")
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
            B = res["drifted_df"]
            X, yb = B.drop(columns=[tc]), B[tc].values
            P = model.predict_proba(X)
            sm = P.max(1)
            if (abs(round(float(roc_auc_score(yb, P[:, 1])), 6) - r["drifted_auc"]) > 1.01e-6 or
                    abs(round(float(sm.mean()), 6) - r["conf_mean"]) > 1.01e-6):
                raise SystemExit(f"STOP: batch for {r['condition_id']} does not match the benchmark CSV.")
            cb = cbpe.auc(P[:, 1])
            rows.append({"condition_id": r["condition_id"],
                         "atc_acc_change": (fixed_acc - np.mean(sm >= t_mc)) * 100,
                         "doc": ac_src - sm.mean(),
                         "CBPE": (fixed_auc - cb) * 100,
                         "CBPE_rel": (cbpe_src - cb) * 100})
            if (i + 1) % 200 == 0:
                print(f"  baselines {i + 1}/1600 ({(time.time() - t0) / 60:.1f} min)", flush=True)
        preds = pd.DataFrame(rows)
        preds.to_csv(pred_path, index=False)
        print(f"Batches reproduced for all 1600 conditions (AUC and confidence match the CSV). Saved {pred_path}")

    # ── Phase 2: evaluation ──────────────────────────────────────────────
    d = df.merge(preds, on="condition_id", validate="one_to_one")
    d["group"] = np.where(d["drift_type"] == "none", d["condition_id"],
                          d["condition_id"].str.rsplit("_", n=1).str[0])
    low = (d["feature_group"] == "low_importance").values
    n = len(d)
    Xn1 = {k: d[cols + ["batch_size"]].values.astype(float) for k, cols in N1.items()}
    Xatc, Xdoc = d[["atc_acc_change"]].values, d[["doc"]].values
    results = {"source": {"auc": auc_s, "accuracy": acc_s, "atc_t_mc": t_mc, "ac_src": ac_src,
                          "cbpe_calibrated": bool(cbpe.use_cal), "cbpe_source_auc": cbpe_src},
               "targets": {}}
    t0 = time.time()
    for t, tcol in TARGETS.items():
        y = d[tcol].values
        P = {m: np.tile(d[m].values.astype(float), (R, 1)) for m in ["CBPE", "CBPE_rel"]}
        for m in ["constant_mean", "ATC_Reg", "DoC_Reg"] + list(N1):
            P[m] = np.zeros((R, n))
        for r in range(R):
            folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r).split(d, d["drift_type"], d["group"])
            for k, (tr, te) in enumerate(folds):
                P["constant_mean"][r, te] = y[tr].mean()
                P["ATC_Reg"][r, te] = LinearRegression().fit(Xatc[tr], y[tr]).predict(Xatc[te])
                P["DoC_Reg"][r, te] = LinearRegression().fit(Xdoc[tr], y[tr]).predict(Xdoc[te])
                for m in N1:
                    P[m][r, te] = make_model(1000 * r + k).fit(Xn1[m][tr], y[tr]).predict(Xn1[m][te])
        print(f"  [{t}] CV done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        order = BASELINES + list(N1)
        results["targets"][t] = {}
        for label, idx in [("ALL conditions", np.ones(n, bool)),
                           ("WITHOUT low-importance conditions", ~low)]:
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
            best = min(BASELINES[1:], key=lambda b: res_t[b]["mae"][0])
            imp_best = (res_t[best]["mae"][0] - res_t["N1_full"]["mae"][0]) / res_t[best]["mae"][0] * 100
            imp_atc = (res_t["ATC_Reg"]["mae"][0] - res_t["N1_full"]["mae"][0]) / res_t["ATC_Reg"]["mae"][0] * 100
            res_t["improvement_over_ATC_pct"], res_t["best_baseline"] = imp_atc, best
            res_t["improvement_over_best_baseline_pct"] = imp_best
            print(f"\nImprovement of N1_full over ATC_Reg: {imp_atc:.1f}% (proposal target: above 30%) | "
                  f"over the best baseline ({best}): {imp_best:.1f}%")
            bt = {m: {dt: float(mean_absolute_error(y[idx & (d['drift_type'] == dt).values],
                                                    P[m].mean(0)[idx & (d['drift_type'] == dt).values]))
                      for dt in sorted(d["drift_type"].unique())} for m in order}
            res_t["mae_by_type"] = bt
            print("\nMAE by drift type (predictions averaged over repeats):")
            print(pd.DataFrame(bt).round(3).to_string())
            results["targets"][t][label] = res_t

    out = os.path.join(a.out_dir, "baselines_results_ieeecis_v12.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
