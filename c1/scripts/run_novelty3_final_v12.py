"""
Novelty 3 final (Option B) on benchmark v1.2 (UCI Adult) + classifier comparison.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Option B output (agreed 2026-10-08, pending supervisor / C2 agreement):
  marginal_extent        none / single / multi   (none+interaction -> none, single+combined -> single)
  interaction_suspected  probability that an interaction component is present (interaction or combined)
Final model: RandomForest(300, class_weight balanced) on the feature-agnostic fingerprint
(9 detector signals + 6 prediction stats + batch_size + top-3 sorted per-feature KS/Chi2/PSI + counts of
features above the no-drift 95th percentile, thresholds from TRAINING no-drift rows only).
Uses fingerprint / build_X from run_novelty3b_v12.py (same folder).

Part 1  classifier comparison on the same features: RF, LightGBM, HistGradientBoosting, LogisticRegression
        (5-class and marginal_extent), LogisticRegression C sweep, leave-one-feature-out RF vs best LR.
Part 2  final Option B evaluation, grouped CV 5 folds x R repeats, plus leave-one-feature-out.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_novelty3b_v12 as n3
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             brier_score_loss, confusion_matrix)
from lightgbm import LGBMClassifier

EXT_L = ["none", "single", "multi"]


def rf(s):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                  n_jobs=-1, random_state=s)


def lr(C):
    return lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


MODELS = {"RandomForest": rf,
          "LightGBM": lambda s: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                               class_weight="balanced", random_state=s, verbose=-1),
          "HistGradBoost": lambda s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                    class_weight="balanced", random_state=s),
          "LogisticReg": lr(1.0)}


def main():
    ap = argparse.ArgumentParser()
    for k in ["csv", "feature_json", "shared_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--compare_repeats", type=int, default=5)
    ap.add_argument("--final_repeats", type=int, default=10)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                           df["condition_id"].str.rsplit("_", n=1).str[0])
    fams = {f: [c for c in df.columns if c.startswith(f + "__")] for f in ["ks", "chi2", "psi"]}
    PER = sum(fams.values(), [])
    n = len(df)
    y5 = df["drift_type"].values
    yext = np.where(np.isin(y5, ["none", "interaction"]), "none", np.where(y5 == "marginal_multi", "multi", "single"))
    yflag = np.isin(y5, ["interaction", "combined"]).astype(int)

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
    res = {"part1_comparison": {}, "part2_final": {}}
    t0 = time.time()

    # ── Part 1: classifier comparison ────────────────────────────────────
    R1, Cs = a.compare_repeats, [0.03, 0.1, 0.3, 1, 3, 10]
    acc = {(m, t): [] for m in MODELS for t in ["5class", "extent"]}
    accC = {C: [] for C in Cs}
    for r in range(R1):
        P = {(m, t): np.empty(n, dtype=object) for m in MODELS for t in ["5class", "extent"]}
        PC = {C: np.empty(n, dtype=object) for C in Cs}
        for k, (tr_i, te_i) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
                                         .split(df, y5, df["group"])):
            tr = np.zeros(n, bool); tr[tr_i] = True
            X = n3.build_X(df, "agnostic", tr, fams, PER)
            for m, mk in MODELS.items():
                P[(m, "5class")][te_i] = mk(1000 * r + k).fit(X[tr], y5[tr]).predict(X[te_i])
                P[(m, "extent")][te_i] = mk(1000 * r + k).fit(X[tr], yext[tr]).predict(X[te_i])
            for C in Cs:
                PC[C][te_i] = lr(C)(0).fit(X[tr], y5[tr]).predict(X[te_i])
        for (m, t), p in P.items():
            acc[(m, t)].append(accuracy_score(y5 if t == "5class" else yext, p))
        for C in Cs:
            accC[C].append(accuracy_score(y5, PC[C]))
        print(f"  [part 1] repeat {r + 1}/{R1} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("\n" + "=" * 84 + "\nPART 1: classifiers on the same agnostic features (accuracy, mean +/- sd)\n" + "=" * 84)
    print(pd.DataFrame([[m] + [f"{np.mean(acc[(m, t)]):.3f} +/- {np.std(acc[(m, t)]):.3f}" for t in ["5class", "extent"]]
                        for m in MODELS], columns=["model", "5class", "marginal_extent"]).to_string(index=False))
    print("\nLogisticRegression C sweep (5-class):",
          {C: f"{np.mean(v):.3f}" for C, v in accC.items()})
    bestC = max(Cs, key=lambda C: np.mean(accC[C]))
    lofo_cmp = {}
    for f in hi_f + lo_f:
        test = tf.apply(lambda s_: f in s_).values
        X = n3.build_X(df, "agnostic", ~test, fams, PER)
        lofo_cmp[f] = {"n": int(test.sum()),
                       "RandomForest": float(accuracy_score(y5[test], rf(0).fit(X[~test], y5[~test]).predict(X[test]))),
                       f"LogReg_C{bestC}": float(accuracy_score(y5[test], lr(bestC)(0).fit(X[~test], y5[~test]).predict(X[test])))}
    print(f"\nLeave-one-feature-out 5-class accuracy, RF vs LogReg C={bestC}:")
    print(pd.DataFrame(lofo_cmp).T.round(3).to_string())
    res["part1_comparison"] = {"accuracy": {f"{m}|{t}": [float(np.mean(v)), float(np.std(v))] for (m, t), v in acc.items()},
                               "logreg_C_sweep_5class": {str(C): float(np.mean(v)) for C, v in accC.items()},
                               "lofo_5class": lofo_cmp}

    # ── Part 2: final Option B ───────────────────────────────────────────
    R2 = a.final_repeats
    pe = np.empty((R2, n), dtype=object)
    pf = np.zeros((R2, n))
    for r in range(R2):
        for k, (tr_i, te_i) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
                                         .split(df, y5, df["group"])):
            tr = np.zeros(n, bool); tr[tr_i] = True
            X = n3.build_X(df, "agnostic", tr, fams, PER)
            pe[r, te_i] = rf(1000 * r + k).fit(X[tr], yext[tr]).predict(X[te_i])
            m = rf(1000 * r + k).fit(X[tr], yflag[tr])
            pf[r, te_i] = m.predict_proba(X[te_i])[:, list(m.classes_).index(1)]
        print(f"  [part 2] repeat {r + 1}/{R2} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    ext_acc = [accuracy_score(yext, pe[r]) for r in range(R2)]
    ext_bal = [balanced_accuracy_score(yext, pe[r]) for r in range(R2)]
    ext_f1 = np.mean([f1_score(yext, pe[r], average=None, labels=EXT_L) for r in range(R2)], axis=0)
    cm = sum(confusion_matrix(yext, pe[r], labels=EXT_L) for r in range(R2)).astype(float)
    cmn = pd.DataFrame(cm / cm.sum(1, keepdims=True), index=EXT_L, columns=EXT_L).round(3)
    auc = [roc_auc_score(yflag, pf[r]) for r in range(R2)]
    brier = [brier_score_loss(yflag, pf[r]) for r in range(R2)]
    brier_ref = brier_score_loss(yflag, np.full(n, yflag.mean()))
    pm = pf.mean(0)
    bins = np.clip((pm * 5).astype(int), 0, 4)
    calib = {f"{b / 5:.1f}-{(b + 1) / 5:.1f}": {"n": int((bins == b).sum()),
                                                  "mean_predicted": float(pm[bins == b].mean()),
                                                  "observed_rate": float(yflag[bins == b].mean())}
             for b in range(5) if (bins == b).any()}
    mag = df["magnitude"].values
    auc_by_mag = {}
    for lo_, hi_, name in [(0.1, 0.3, "low"), (0.4, 0.6, "mid"), (0.7, 1.0, "high")]:
        mk = (y5 == "none") | (y5 == "marginal_single") | (y5 == "marginal_multi") | ((mag >= lo_) & (mag <= hi_))
        auc_by_mag[name] = float(np.mean([roc_auc_score(yflag[mk], pf[r][mk]) for r in range(R2)]))

    print("\n" + "=" * 84 + f"\nPART 2: FINAL OPTION B ({R2} repeats x 5 folds)\n" + "=" * 84)
    print(f"marginal_extent accuracy {np.mean(ext_acc):.3f} +/- {np.std(ext_acc):.3f} | balanced {np.mean(ext_bal):.3f}")
    print("per-class F1: " + ", ".join(f"{l} {v:.3f}" for l, v in zip(EXT_L, ext_f1)))
    print(f"confusion (rows = true, row-normalised):\n{cmn.to_string()}")
    print(f"\ninteraction_suspected AUC {np.mean(auc):.3f} +/- {np.std(auc):.3f} | "
          f"Brier {np.mean(brier):.3f} (always-predict-base-rate reference {brier_ref:.3f}; lower is better)")
    print("AUC by interaction magnitude (vs all non-interaction rows):", {k: round(v, 3) for k, v in auc_by_mag.items()})
    print("Calibration (mean predicted vs observed rate, by predicted-probability bin):")
    for b, v in calib.items():
        print(f"  {b}: n={v['n']:4d} predicted {v['mean_predicted']:.2f} observed {v['observed_rate']:.2f}")

    lofo = {}
    for f in hi_f + lo_f:
        test = tf.apply(lambda s_: f in s_).values
        X = n3.build_X(df, "agnostic", ~test, fams, PER)
        e = accuracy_score(yext[test], rf(0).fit(X[~test], yext[~test]).predict(X[test]))
        m = rf(0).fit(X[~test], yflag[~test])
        p = m.predict_proba(X[test])[:, list(m.classes_).index(1)]
        lofo[f] = {"n": int(test.sum()), "extent_accuracy": float(e),
                   "flag_auc": float(roc_auc_score(yflag[test], p)) if len(set(yflag[test])) == 2 else None}
    print("\nLeave-one-feature-out (final Option B):")
    print(pd.DataFrame(lofo).T.to_string())

    res["part2_final"] = {"extent_accuracy": [float(np.mean(ext_acc)), float(np.std(ext_acc))],
                          "extent_balanced_accuracy": float(np.mean(ext_bal)),
                          "extent_per_class_f1": dict(zip(EXT_L, map(float, ext_f1))),
                          "extent_confusion": cmn.to_dict(),
                          "flag_auc": [float(np.mean(auc)), float(np.std(auc))],
                          "flag_brier": float(np.mean(brier)), "flag_brier_base_rate": float(brier_ref),
                          "flag_auc_by_magnitude": auc_by_mag, "flag_calibration": calib, "lofo": lofo}
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"condition_id": df["condition_id"], "drift_type": y5, "true_extent": yext,
                  "pred_extent": pe[0], "p_interaction": pf[0]}).to_csv(
        os.path.join(a.out_dir, "novelty3_final_oof_adult_v12.csv"), index=False)
    out = os.path.join(a.out_dir, "novelty3_final_results_adult_v12.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved: {out}   Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
