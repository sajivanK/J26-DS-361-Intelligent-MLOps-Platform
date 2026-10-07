"""
Analysis checks behind two Novelty 1 claims (UCI Adult, benchmark v1.2).
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Check 1  Point-mass sensitivity: retrain and evaluate N1 configs WITHOUT any condition that
         touches capital_gain or capital_loss (point-mass features that saturate KS).
         Question: do detectors still add value on top of prediction statistics?
Check 2  "Confidently wrong": do confidence-based baselines (ATC, DoC, CBPE) predict the wrong
         direction of change, and how often does model confidence RISE while accuracy FALLS?

Inputs: v1.2 benchmark CSV, baseline predictions CSV and baseline results JSON
(from run_baselines_v12.py), feature JSON, shared/ (for the condition list).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

DET_V11 = ["v11_ks_score", "v11_chi2_score", "v11_psi_score", "v11_mmd_score", "v11_lsdd_score", "v11_clf_auc"]
DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]
CONFIGS = {"constant_mean": None, "conf_3": CONF, "posprob_3": POS, "conf_plus_posprob": CONF + POS,
           "detectors_v11": DET_V11, "detectors_v12": DET_V12, "all_v12": DET_V12 + CONF,
           "everything": DET_V12 + CONF + POS}
TARGETS = {"fixed": "actual_accuracy_drop", "paired": "harm_paired"}
POINT_MASS = {"capital_gain", "capital_loss"}


def make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    for k in ["csv", "baseline_preds", "baseline_json", "feature_json", "shared_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    imp = json.load(open(a.feature_json))
    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    conds = bl.build_condition_list(imp["feature_groups"]["high_importance"]["features"],
                                    imp["feature_groups"]["low_importance"]["features"],
                                    imp["feature_types"], df["dataset"].iloc[0])
    touched = {}
    for c in conds:
        fs = set(c["features"])
        for x, z in (c.get("interaction_pairs") or []):
            fs |= {x, z}
        touched[c["condition_id"]] = fs
    out = {"check1_point_mass_excluded": {}, "check2_confidently_wrong": {}}

    # ── Check 1 ──────────────────────────────────────────────────────────
    pm = df["condition_id"].map(touched).apply(lambda s: bool(s & POINT_MASS))
    d = df[~pm].reset_index(drop=True)
    d["group"] = np.where(d["drift_type"] == "none", d["condition_id"],
                          d["condition_id"].str.rsplit("_", n=1).str[0])
    print("=" * 78 + "\nCHECK 1: N1 configs retrained WITHOUT capital_gain / capital_loss conditions")
    print(f"{len(d)} rows, {d['group'].nunique()} groups, {a.repeats} repeats x 5 folds\n" + "=" * 78)
    out["check1_point_mass_excluded"]["rows"] = int(len(d))
    for t, tcol in TARGETS.items():
        y = d[tcol].values
        maes = {c: [] for c in CONFIGS}
        for r in range(a.repeats):
            folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
                         .split(d, d["drift_type"], d["group"]))
            for c, cols in CONFIGS.items():
                p = np.zeros(len(d))
                X = None if cols is None else d[cols + ["batch_size"]].values.astype(float)
                for k, (tr, te) in enumerate(folds):
                    p[te] = y[tr].mean() if X is None else make_model(1000 * r + k).fit(X[tr], y[tr]).predict(X[te])
                maes[c].append(mean_absolute_error(y, p))
        print(f"\n{t}: MAE (pp), mean +/- sd over {a.repeats} repeats")
        for c in CONFIGS:
            print(f"  {c:20s} {np.mean(maes[c]):.3f} +/- {np.std(maes[c]):.3f}")
        dd = np.array(maes["everything"]) - np.array(maes["conf_plus_posprob"])
        print(f"  everything minus conf_plus_posprob: {dd.mean():+.3f} pp "
              f"(everything better in {(dd < 0).sum()}/{a.repeats} repeats)")
        out["check1_point_mass_excluded"][t] = {
            "mae": {c: [float(np.mean(v)), float(np.std(v))] for c, v in maes.items()},
            "everything_minus_conf_plus_posprob": float(dd.mean()),
            "repeats_everything_better": int((dd < 0).sum())}

    # ── Check 2 ──────────────────────────────────────────────────────────
    ac_src = float(json.load(open(a.baseline_json))["source"]["ac_src"])
    m = df.merge(pd.read_csv(a.baseline_preds), on="condition_id", validate="one_to_one")
    tab = m.groupby("drift_type")[["actual_accuracy_drop", "ATC_MC", "DoC_Feat", "CBPE"]].mean()
    dr = m[m["drift_type"] != "none"]
    share = float(((dr["conf_mean"] > ac_src) & (dr["actual_accuracy_drop"] > 2)).mean())
    wrong_sign = {b: float(((dr[b] < 0) & (dr["actual_accuracy_drop"] > 2)).sum() /
                           max(1, (dr["actual_accuracy_drop"] > 2).sum())) for b in ["ATC_MC", "DoC_Feat", "CBPE"]}
    print("\n" + "=" * 78 + "\nCHECK 2: confidence-based baselines vs true accuracy drop\n" + "=" * 78)
    print("Mean TRUE drop vs mean PREDICTED drop (pp; negative = predicts accuracy went UP):")
    print(tab.round(2).to_string())
    print(f"\nSource mean max-prob (clean held-out data): {ac_src:.4f}")
    print(f"Drift conditions where confidence ROSE above source while accuracy FELL by > 2 pp: {share:.1%}")
    print("Among drift conditions with a true drop > 2 pp, share where the baseline predicted a GAIN:")
    for b, v in wrong_sign.items():
        print(f"  {b:9s} {v:.1%}")
    out["check2_confidently_wrong"] = {"mean_by_type": tab.round(4).to_dict(), "source_ac": ac_src,
                                       "share_conf_up_and_drop_gt2": share,
                                       "share_predicted_gain_when_drop_gt2": wrong_sign}

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "analysis_checks_adult_v12.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
