"""
Novelty 3 on benchmark v1.2 (UCI Adult): drift-type classification from detector fingerprints.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Model: RandomForestClassifier(300 trees, class_weight="balanced"). batch_size is an input
(known at inference). Protocol as N1: groups = drift conditions, StratifiedGroupKFold by
drift_type, 5 folds x R repeats, mean +/- sd over repeats.
Label schemes: 5-class (contract: none, marginal_single, marginal_multi, interaction, combined)
and 4-class (proposal: marginal_single + marginal_multi merged into marginal).
Proposal target: overall accuracy at least 70%.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

DET_V11 = ["v11_ks_score", "v11_chi2_score", "v11_psi_score", "v11_mmd_score", "v11_lsdd_score", "v11_clf_auc"]
DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
CONF = ["conf_mean", "conf_std", "conf_pct_lt_06"]
POS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
L5 = ["none", "marginal_single", "marginal_multi", "interaction", "combined"]
L4 = ["none", "marginal", "interaction", "combined"]


def make_model(seed):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                  n_jobs=-1, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    for k in ["csv", "feature_json", "shared_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    R = a.repeats

    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                           df["condition_id"].str.rsplit("_", n=1).str[0])
    PER = [c for c in df.columns if c.startswith(("ks__", "chi2__", "psi__"))]
    configs = {"detectors_v11": DET_V11, "detectors_v12": DET_V12,
               "v12_plus_per_feature": DET_V12 + PER,
               "v12_per_feature_plus_pred_stats": DET_V12 + PER + CONF + POS}
    X = {c: df[cols + ["batch_size"]].values.astype(float) for c, cols in configs.items()}
    y5 = df["drift_type"].values
    y4 = np.where(np.isin(y5, ["marginal_single", "marginal_multi"]), "marginal", y5)
    schemes = {"5class": (y5, L5), "4class": (y4, L4)}
    n = len(df)
    print(f"Rows {n} | groups {df['group'].nunique()} | per-feature columns {len(PER)} | repeats {R} x 5 folds")
    print("Class counts:", pd.Series(y5).value_counts().to_dict())

    oof = {s: {c: np.empty((R, n), dtype=object) for c in configs} for s in schemes}
    t0 = time.time()
    for r in range(R):
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r).split(df, y5, df["group"]))
        for s, (y, _) in schemes.items():
            for c in configs:
                for k, (tr, te) in enumerate(folds):
                    oof[s][c][r, te] = make_model(1000 * r + k).fit(X[c][tr], y[tr]).predict(X[c][te])
        print(f"  repeat {r + 1}/{R} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    imp = json.load(open(a.feature_json))
    hi_f = imp["feature_groups"]["high_importance"]["features"][:5]
    lo_f = imp["feature_groups"]["low_importance"]["features"][:5]
    touched = {}
    for cnd in bl.build_condition_list(hi_f, lo_f, imp["feature_types"], df["dataset"].iloc[0]):
        fs = set(cnd["features"])
        for x_, z_ in (cnd.get("interaction_pairs") or []):
            fs |= {x_, z_}
        touched[cnd["condition_id"]] = fs
    tf = df["condition_id"].map(touched)
    pm = tf.apply(lambda s_: bool(s_ & {"capital_gain", "capital_loss"})).values

    res = {"repeats": R, "per_feature_columns": PER, "schemes": {}}
    for s, (y, L) in schemes.items():
        print("\n" + "=" * 84 + f"\n{s.upper()} | labels {L}\n" + "=" * 84)
        rows, res["schemes"][s] = [], {"configs": {}}
        for c in configs:
            acc = [accuracy_score(y, oof[s][c][r]) for r in range(R)]
            bal = [balanced_accuracy_score(y, oof[s][c][r]) for r in range(R)]
            mf1 = [f1_score(y, oof[s][c][r], average="macro") for r in range(R)]
            acc_pm = [accuracy_score(y[~pm], oof[s][c][r][~pm]) for r in range(R)]
            res["schemes"][s]["configs"][c] = {"accuracy": [float(np.mean(acc)), float(np.std(acc))],
                                               "balanced_accuracy": float(np.mean(bal)), "macro_f1": float(np.mean(mf1)),
                                               "accuracy_without_point_mass": float(np.mean(acc_pm))}
            rows.append([c, f"{np.mean(acc):.3f} +/- {np.std(acc):.3f}", f"{np.mean(bal):.3f}",
                         f"{np.mean(mf1):.3f}", f"{np.mean(acc_pm):.3f}"])
        print(pd.DataFrame(rows, columns=["config", "accuracy", "balanced_acc", "macro_F1",
                                          "acc w/o capital_gain/loss"]).to_string(index=False))
        best = max(configs, key=lambda c: res["schemes"][s]["configs"][c]["accuracy"][0])
        res["schemes"][s]["best"] = best
        print(f"\nBest config: {best}  (proposal target: accuracy >= 0.70)")

        P = oof[s][best]
        f1c = np.mean([f1_score(y, P[r], average=None, labels=L) for r in range(R)], axis=0)
        print("\nPer-class F1 (best config):")
        for lab, v in zip(L, f1c):
            print(f"  {lab:16s} {v:.3f}")
        cm = sum(confusion_matrix(y, P[r], labels=L) for r in range(R)).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        print("\nConfusion matrix (rows = true, columns = predicted, row-normalised, summed over repeats):")
        print(pd.DataFrame(cmn, index=L, columns=L).round(2).to_string())

        mag = df["magnitude"].values
        bucket = np.where(y5 == "none", "none (0.0)",
                          np.where(mag <= 0.3, "low (0.1-0.3)", np.where(mag <= 0.6, "mid (0.4-0.6)", "high (0.7-1.0)")))
        by_mag = {b: float(np.mean([accuracy_score(y[bucket == b], P[r][bucket == b]) for r in range(R)]))
                  for b in ["none (0.0)", "low (0.1-0.3)", "mid (0.4-0.6)", "high (0.7-1.0)"]}
        by_bs = {int(b): float(np.mean([accuracy_score(y[df["batch_size"].values == b],
                                                       P[r][df["batch_size"].values == b]) for r in range(R)]))
                 for b in sorted(df["batch_size"].unique())}
        print("\nAccuracy by drift magnitude:", {k: round(v, 3) for k, v in by_mag.items()})
        print("Accuracy by batch size:     ", {k: round(v, 3) for k, v in by_bs.items()})

        lofo = {}
        for f in hi_f + lo_f:
            test = tf.apply(lambda s_: f in s_).values
            pr = make_model(0).fit(X[best][~test], y[~test]).predict(X[best][test])
            lofo[f] = {"n": int(test.sum()), "accuracy": float(accuracy_score(y[test], pr))}
        print("Leave-one-feature-out accuracy (best config):",
              {k: round(v["accuracy"], 3) for k, v in lofo.items()})
        res["schemes"][s].update({"per_class_f1": dict(zip(L, map(float, f1c))),
                                  "confusion_row_normalised": pd.DataFrame(cmn, index=L, columns=L).round(4).to_dict(),
                                  "accuracy_by_magnitude": by_mag, "accuracy_by_batch_size": by_bs, "lofo": lofo})

    best5 = res["schemes"]["5class"]["best"]
    m = make_model(0).fit(X[best5], y5)
    names = configs[best5] + ["batch_size"]
    top = sorted(zip(names, m.feature_importances_), key=lambda z: -z[1])[:15]
    res["importance_5class_best"] = {k: float(v) for k, v in top}
    print("\nTop 15 RF importances (5-class, best config):")
    for k, v in top:
        print(f"  {k:24s} {v:.3f}")

    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"condition_id": df["condition_id"], "drift_type": y5,
                  "pred_5class": oof["5class"][best5][0],
                  "pred_4class": oof["4class"][res["schemes"]["4class"]["best"]][0]}).to_csv(
        os.path.join(a.out_dir, "novelty3_oof_preds_adult_v12.csv"), index=False)
    out = os.path.join(a.out_dir, "novelty3_results_adult_v12.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved: {out} (+ out-of-fold predictions CSV)   Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
