"""
Novelty 3b on benchmark v1.2 (UCI Adult): improved drift-type features and a two-part output.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

All features are derived from the existing six detectors and the model's prediction statistics.
Configs:
  named_per_feature        previous best: 9 detector signals + 24 NAMED per-feature scores + pred stats
  agnostic                 9 detector signals + feature-AGNOSTIC fingerprint + pred stats
                           (top-3 sorted per-feature KS / Chi2 / PSI; count of features above the no-drift
                           95th percentile, thresholds learned per batch size from TRAINING no-drift rows only)
  agnostic_plus_residual   agnostic + "unexplained multivariate drift": C2ST / MMD / LSDD minus what the
                           marginal (single-feature) fingerprint predicts. Cross-fitted (GroupKFold) on the
                           training fold, so no leakage. Large residual = multivariate change not explained by
                           single features = evidence of interaction drift.
Targets: 5-class, 4-class, marginal_extent (none / single / multi), interaction_flag (yes / no),
and the two-part output recombined into 5 classes.
Protocol: grouped CV as N1 (StratifiedGroupKFold by drift_type, 5 folds x R repeats). RandomForest
(300 trees, class_weight balanced). Leave-one-feature-out for the best config.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, cross_val_predict

DET_V12 = ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean", "psi_max", "mmd", "lsdd", "clf_auc"]
PRED = ["conf_mean", "conf_std", "conf_pct_lt_06", "mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]
MULTI = ["clf_auc", "mmd", "lsdd"]
L5 = ["none", "marginal_single", "marginal_multi", "interaction", "combined"]
CONFIGS = ["named_per_feature", "agnostic", "agnostic_plus_residual"]


def rf(seed):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                  n_jobs=-1, random_state=seed)


def fingerprint(df, fams, tr):
    """Feature-agnostic fingerprint; thresholds use TRAINING no-drift rows only (tr = boolean mask)."""
    n, bs = len(df), df["batch_size"].values
    none_tr = tr & (df["drift_type"].values == "none")
    feats_all = sorted({c.split("__", 1)[1] for cols in fams.values() for c in cols})
    any_exceed = np.zeros((n, len(feats_all)), dtype=bool)
    blocks = []
    for fam, cols in fams.items():
        M = df[cols].values.astype(float)
        top = -np.sort(-M, axis=1)[:, :3]
        exceed = np.zeros_like(M, dtype=bool)
        for b in np.unique(bs):
            thr = np.quantile(M[none_tr & (bs == b)], 0.95, axis=0)
            exceed[bs == b] = M[bs == b] > thr
        for j, c in enumerate(cols):
            any_exceed[:, feats_all.index(c.split("__", 1)[1])] |= exceed[:, j]
        blocks += [top, exceed.sum(1, keepdims=True)]
    blocks.append(any_exceed.sum(1, keepdims=True))
    return np.hstack(blocks).astype(float)


def residuals(df, F, tr, seed=0):
    """Multivariate detector score minus what the marginal fingerprint predicts (cross-fitted)."""
    Xm = np.hstack([F, df[["batch_size"]].values.astype(float)])
    g = df["group"].values
    out = np.zeros((len(df), len(MULTI)))
    for j, col in enumerate(MULTI):
        y = df[col].values.astype(float)
        m = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
        out[tr, j] = y[tr] - cross_val_predict(m, Xm[tr], y[tr], groups=g[tr], cv=GroupKFold(n_splits=3))
        out[~tr, j] = y[~tr] - m.fit(Xm[tr], y[tr]).predict(Xm[~tr])
    return out


def build_X(df, cfg, tr, fams, PER):
    base = df[DET_V12 + PRED + ["batch_size"]].values.astype(float)
    if cfg == "named_per_feature":
        return np.hstack([base, df[PER].values.astype(float)])
    F = fingerprint(df, fams, tr)
    if cfg == "agnostic":
        return np.hstack([base, F])
    return np.hstack([base, F, residuals(df, F, tr)])


def recombine(ext, flag):
    out = np.where(ext == "multi", "marginal_multi",
           np.where(ext == "single", np.where(flag == "yes", "combined", "marginal_single"),
                    np.where(flag == "yes", "interaction", "none")))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ["csv", "feature_json", "shared_dir", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    R = a.repeats

    df = pd.read_csv(a.csv)
    assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all()
    df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                           df["condition_id"].str.rsplit("_", n=1).str[0])
    fams = {f: [c for c in df.columns if c.startswith(f + "__")] for f in ["ks", "chi2", "psi"]}
    PER = sum(fams.values(), [])
    y5 = df["drift_type"].values
    targets = {
        "5class": y5,
        "4class": np.where(np.isin(y5, ["marginal_single", "marginal_multi"]), "marginal", y5),
        "marginal_extent": np.where(np.isin(y5, ["none", "interaction"]), "none",
                                    np.where(y5 == "marginal_multi", "multi", "single")),
        "interaction_flag": np.where(np.isin(y5, ["interaction", "combined"]), "yes", "no"),
    }
    n = len(df)
    print(f"Rows {n} | groups {df['group'].nunique()} | repeats {R} x 5 folds")

    oof = {t: {c: np.empty((R, n), dtype=object) for c in CONFIGS} for t in targets}
    prob = {c: np.zeros((R, n)) for c in CONFIGS}
    t0 = time.time()
    for r in range(R):
        for k, (tr_i, te_i) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
                                         .split(df, y5, df["group"])):
            tr = np.zeros(n, bool); tr[tr_i] = True
            for c in CONFIGS:
                X = build_X(df, c, tr, fams, PER)
                for t, y in targets.items():
                    m = rf(1000 * r + k).fit(X[tr], y[tr])
                    oof[t][c][r, te_i] = m.predict(X[te_i])
                    if t == "interaction_flag":
                        prob[c][r, te_i] = m.predict_proba(X[te_i])[:, list(m.classes_).index("yes")]
        print(f"  repeat {r + 1}/{R} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = {"repeats": R, "results": {}}
    print("\n" + "=" * 92 + "\nACCURACY (mean +/- sd over repeats)\n" + "=" * 92)
    rows = []
    for c in CONFIGS:
        res["results"][c] = {}
        row = [c]
        for t, y in targets.items():
            acc = [accuracy_score(y, oof[t][c][r]) for r in range(R)]
            bal = [balanced_accuracy_score(y, oof[t][c][r]) for r in range(R)]
            res["results"][c][t] = {"accuracy": [float(np.mean(acc)), float(np.std(acc))],
                                    "balanced_accuracy": float(np.mean(bal))}
            row.append(f"{np.mean(acc):.3f}")
        rec = [accuracy_score(y5, recombine(oof["marginal_extent"][c][r], oof["interaction_flag"][c][r]))
               for r in range(R)]
        auc = [roc_auc_score(targets["interaction_flag"] == "yes", prob[c][r]) for r in range(R)]
        res["results"][c]["two_part_recombined_5class_accuracy"] = float(np.mean(rec))
        res["results"][c]["interaction_flag_auc"] = float(np.mean(auc))
        row += [f"{np.mean(rec):.3f}", f"{np.mean(auc):.3f}"]
        rows.append(row)
    print(pd.DataFrame(rows, columns=["config", "5class", "4class", "extent(3)", "inter_flag",
                                      "two-part->5class", "inter_flag AUC"]).to_string(index=False))
    print("Proposal target: 5-class / 4-class accuracy >= 0.70")

    best = max(CONFIGS, key=lambda c: res["results"][c]["5class"]["accuracy"][0])
    res["best_5class"] = best
    print(f"\nBest 5-class config: {best}")
    for t in ["5class", "marginal_extent", "interaction_flag"]:
        y, P = targets[t], oof[t][best]
        L = sorted(set(y), key=lambda v: L5.index(v) if v in L5 else 0)
        cm = sum(confusion_matrix(y, P[r], labels=L) for r in range(R)).astype(float)
        cmn = pd.DataFrame(cm / cm.sum(1, keepdims=True), index=L, columns=L).round(2)
        f1c = np.mean([f1_score(y, P[r], average=None, labels=L) for r in range(R)], axis=0)
        print(f"\n[{t}] confusion (rows = true, row-normalised):\n{cmn.to_string()}")
        print(f"[{t}] per-class F1: " + ", ".join(f"{l} {v:.3f}" for l, v in zip(L, f1c)))
        res["results"][best][f"{t}_confusion"] = cmn.to_dict()
        res["results"][best][f"{t}_per_class_f1"] = dict(zip(L, map(float, f1c)))

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
    print("\n" + "=" * 92 + "\nLEAVE-ONE-FEATURE-OUT accuracy (train without the feature, test on it)\n" + "=" * 92)
    lofo_rows, res["lofo"] = [], {}
    for f in hi_f + lo_f:
        test = tf.apply(lambda s_: f in s_).values
        row, res["lofo"][f] = [f, int(test.sum())], {}
        for c in ["named_per_feature", best] if best != "named_per_feature" else ["named_per_feature"]:
            X = build_X(df, c, ~test, fams, PER)
            for t in ["5class", "marginal_extent"]:
                y = targets[t]
                acc = accuracy_score(y[test], rf(0).fit(X[~test], y[~test]).predict(X[test]))
                res["lofo"][f][f"{c}_{t}"] = float(acc)
                row.append(f"{acc:.3f}")
        lofo_rows.append(row)
    cols = ["held_out", "n"] + [f"{c}_{t}" for c in (["named_per_feature", best] if best != "named_per_feature"
                                                     else ["named_per_feature"]) for t in ["5class", "extent"]]
    print(pd.DataFrame(lofo_rows, columns=cols).to_string(index=False))

    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"condition_id": df["condition_id"], "drift_type": y5,
                  "pred_5class": oof["5class"][best][0],
                  "pred_extent": oof["marginal_extent"][best][0],
                  "pred_interaction_flag": oof["interaction_flag"][best][0],
                  "p_interaction": prob[best][0]}).to_csv(
        os.path.join(a.out_dir, "novelty3b_oof_preds_adult_v12.csv"), index=False)
    out = os.path.join(a.out_dir, "novelty3b_results_adult_v12.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nSaved: {out}   Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
