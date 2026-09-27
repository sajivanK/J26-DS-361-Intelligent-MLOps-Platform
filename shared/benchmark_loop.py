"""
Benchmark Data Collection Loop - FINAL FROZEN FRAMEWORK
J26-DS-361 | C1 Research | Sajivan K (IT23172296)

VERSION: 1.0 FROZEN - DO NOT MODIFY BETWEEN PP1 AND PP2

TARGET: 1,600 conditions per dataset

DRIFT CONDITIONS (1,400):
    Marginal Single:  10 features x 10 magnitudes x 5 seeds = 500
    Marginal Multi:   6 combos   x 10 magnitudes x 5 seeds = 300
    Interaction:      6 pairs    x 10 magnitudes x 5 seeds = 300
    Combined:         6 configs  x 10 magnitudes x 5 seeds = 300

NO-DRIFT BASELINES (200):
    50 seeds x 4 batch sizes = 200

FEATURE SELECTION RULE (fixed):
    High importance: top 5 features by SHAP/gain
    Low importance:  bottom 5 features by SHAP/gain
    Each feature tested individually for marginal single
    Fixed combinations for multi/interaction/combined

NOVELTY TARGETS:
    N1 Harm Prediction:      MAE < 5pp on held-out 20%
    N2 Conformal Bounds:     Coverage >= 95% on held-out 20%
    N3 Drift Classification: Accuracy >= 70% on held-out 20% (4 classes)

PRODUCTION USE:
    Same 1,600 conditions used for client onboarding (background process)
    Onboarding runs asynchronously, client notified on completion
    5-second response time applies to monitoring only, not onboarding
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import ks_2samp, chi2_contingency
from injection.inject_drift import inject_drift


# ── FROZEN FRAMEWORK PARAMETERS ──────────────────────────────────────────────
MAGNITUDES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 123, 456, 789, 1234]
N_HIGH_FEATURES = 5
N_LOW_FEATURES = 5
N_NODRIFT_CONDITIONS = 200
NODRIFT_BATCH_SIZES = [500, 1000, 2000, 5000]
CHECKPOINT_EVERY = 50
CLASSIFIER_SUBSAMPLE = 1000
KERNEL_SUBSAMPLE = 500


# ── DETECTOR IMPLEMENTATIONS ─────────────────────────────────────────────────

def compute_ks_score(ref_features, drift_features, numeric_cols):
    scores = []
    for col in numeric_cols:
        try:
            stat, _ = ks_2samp(ref_features[col].values,
                               drift_features[col].values)
            scores.append(stat)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_chi2_score(ref_features, drift_features, numeric_cols, n_bins=10):
    scores = []
    for col in numeric_cols:
        try:
            all_vals = pd.concat([ref_features[col], drift_features[col]])
            bins = pd.cut(all_vals, bins=n_bins, retbins=True)[1]
            ref_c = pd.cut(ref_features[col], bins=bins).value_counts().sort_index() + 1e-10
            drift_c = pd.cut(drift_features[col], bins=bins).value_counts().sort_index() + 1e-10
            ref_c, drift_c = ref_c.align(drift_c, fill_value=1e-10)
            stat, _, _, _ = chi2_contingency(np.array([ref_c.values, drift_c.values]))
            scores.append(min(stat / (len(ref_features) + len(drift_features)), 1.0))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_psi_score(ref_features, drift_features, numeric_cols, n_bins=10):
    def psi_single(r, d):
        bp = np.unique(np.percentile(np.concatenate([r, d]),
                                     np.linspace(0, 100, n_bins + 1)))
        rp = np.histogram(r, bins=bp)[0] / len(r)
        dp = np.histogram(d, bins=bp)[0] / len(d)
        rp = np.where(rp == 0, 1e-10, rp)
        dp = np.where(dp == 0, 1e-10, dp)
        return float(np.sum((dp - rp) * np.log(dp / rp)))
    scores = []
    for col in numeric_cols:
        try:
            scores.append(psi_single(ref_features[col].values,
                                     drift_features[col].values))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_mmd_score(ref_features, drift_features,
                      n_sample=KERNEL_SUBSAMPLE, sigma=1.0):
    try:
        n = min(n_sample, len(ref_features), len(drift_features))
        rs = ref_features.sample(n=n, random_state=42).values.astype(float)
        ds = drift_features.sample(n=n, random_state=42).values.astype(float)
        mu = rs.mean(0); std = rs.std(0) + 1e-10
        rn = (rs - mu) / std; dn = (ds - mu) / std
        def rbf(X, Y):
            d = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            return np.exp(-np.sum(d**2, axis=2) / (2*sigma**2))
        return float(max(0.0, rbf(rn,rn).mean() + rbf(dn,dn).mean()
                         - 2*rbf(rn,dn).mean()))
    except Exception:
        return 0.0


def compute_lsdd_score(ref_features, drift_features,
                       n_sample=KERNEL_SUBSAMPLE, h=1.0):
    try:
        n = min(n_sample, len(ref_features), len(drift_features))
        rs = ref_features.sample(n=n, random_state=42).values.astype(float)
        ds = drift_features.sample(n=n, random_state=42).values.astype(float)
        mu = rs.mean(0); std = rs.std(0) + 1e-10
        rn = (rs - mu) / std; dn = (ds - mu) / std
        def gauss(X, Y):
            d = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            return np.exp(-np.sum(d**2, axis=2) / (2*h**2))
        return float(max(0.0, gauss(rn,rn).mean() + gauss(dn,dn).mean()
                         - 2*gauss(rn,dn).mean()))
    except Exception:
        return 0.0


def compute_classifier_auc(ref_features, drift_features,
                            n_sample=CLASSIFIER_SUBSAMPLE):
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        n = min(n_sample, len(ref_features), len(drift_features))
        X = pd.concat([ref_features.sample(n=n, random_state=42),
                       drift_features.sample(n=n, random_state=42)],
                      ignore_index=True)
        y = np.array([0]*n + [1]*n)
        clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                            use_label_encoder=False, eval_metric='logloss',
                            random_state=42, verbosity=0)
        return float(np.mean(cross_val_score(clf, X, y, cv=3,
                                             scoring='roc_auc', n_jobs=-1)))
    except Exception:
        return 0.5


def compute_confidence_statistics(model, drifted_df, target_col='target'):
    proba = model.predict_proba(drifted_df.drop(columns=[target_col]))[:, 1]
    return float(np.mean(proba)), float(np.std(proba)), float(np.mean(proba < 0.6))


def run_all_detectors(reference_df, drifted_df, target_col='target'):
    rf = reference_df.drop(columns=[target_col])
    df = drifted_df.drop(columns=[target_col])
    nc = rf.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "ks_score":   compute_ks_score(rf[nc], df[nc], nc),
        "chi2_score": compute_chi2_score(rf[nc], df[nc], nc),
        "psi_score":  compute_psi_score(rf[nc], df[nc], nc),
        "mmd_score":  compute_mmd_score(rf[nc], df[nc]),
        "lsdd_score": compute_lsdd_score(rf[nc], df[nc]),
        "clf_auc":    compute_classifier_auc(rf[nc], df[nc]),
    }


# ── CONDITION BUILDER ─────────────────────────────────────────────────────────

def build_condition_list(high_feats, low_feats, feature_types, dataset_name):
    """
    Build the full 1,600 condition list. FROZEN STRUCTURE.
    Asserts exactly 1,600 conditions are produced.
    """
    conditions = []
    hf = high_feats[:N_HIGH_FEATURES]
    lf = low_feats[:N_LOW_FEATURES]

    # ── MARGINAL SINGLE (500) ─────────────────────────────────────────────────
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for feat in hf:
                conditions.append({
                    "drift_type": "marginal_single", "features": [feat],
                    "magnitude": mag,
                    "feature_types": {feat: feature_types.get(feat, 'continuous')},
                    "seed": seed, "feature_group": "high_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_ms_high_{feat}_{mag}_{seed}"
                })
            for feat in lf:
                conditions.append({
                    "drift_type": "marginal_single", "features": [feat],
                    "magnitude": mag,
                    "feature_types": {feat: feature_types.get(feat, 'continuous')},
                    "seed": seed, "feature_group": "low_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_ms_low_{feat}_{mag}_{seed}"
                })

    # ── MARGINAL MULTI (300) ──────────────────────────────────────────────────
    multi_h = [hf[:2], hf[:3], hf[:5]]
    multi_l = [lf[:2], lf[:3], lf[:5]]
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for i, combo in enumerate(multi_h):
                ft = {f: feature_types.get(f, 'continuous') for f in combo}
                conditions.append({
                    "drift_type": "marginal_multi", "features": combo,
                    "magnitude": mag, "feature_types": ft,
                    "seed": seed, "feature_group": "high_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_mm_high_c{i+1}_{mag}_{seed}"
                })
            for i, combo in enumerate(multi_l):
                ft = {f: feature_types.get(f, 'continuous') for f in combo}
                conditions.append({
                    "drift_type": "marginal_multi", "features": combo,
                    "magnitude": mag, "feature_types": ft,
                    "seed": seed, "feature_group": "low_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_mm_low_c{i+1}_{mag}_{seed}"
                })

    # ── INTERACTION (300) ─────────────────────────────────────────────────────
    pairs_h = [(hf[0],hf[1]), (hf[0],hf[2]), (hf[1],hf[2])]
    pairs_l = [(lf[0],lf[1]), (lf[0],lf[2]), (lf[1],lf[2])]
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for i, (fa,fb) in enumerate(pairs_h):
                ft = {fa: feature_types.get(fa,'continuous'),
                      fb: feature_types.get(fb,'continuous')}
                conditions.append({
                    "drift_type": "interaction", "features": [fa, fb],
                    "magnitude": mag, "feature_types": ft,
                    "seed": seed, "feature_group": "high_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_int_high_p{i+1}_{mag}_{seed}"
                })
            for i, (fa,fb) in enumerate(pairs_l):
                ft = {fa: feature_types.get(fa,'continuous'),
                      fb: feature_types.get(fb,'continuous')}
                conditions.append({
                    "drift_type": "interaction", "features": [fa, fb],
                    "magnitude": mag, "feature_types": ft,
                    "seed": seed, "feature_group": "low_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_int_low_p{i+1}_{mag}_{seed}"
                })

    # ── COMBINED (300) ────────────────────────────────────────────────────────
    cfgs_h = [
        {"marginal": [hf[0]], "interaction": [(hf[1], hf[2])]},
        {"marginal": [hf[1]], "interaction": [(hf[0], hf[2])]},
        {"marginal": [hf[2]], "interaction": [(hf[0], hf[1])]},
    ]
    cfgs_l = [
        {"marginal": [lf[0]], "interaction": [(lf[1], lf[2])]},
        {"marginal": [lf[1]], "interaction": [(lf[0], lf[2])]},
        {"marginal": [lf[2]], "interaction": [(lf[0], lf[1])]},
    ]
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for i, cfg in enumerate(cfgs_h):
                af = cfg["marginal"] + [f for p in cfg["interaction"] for f in p]
                ft = {f: feature_types.get(f,'continuous') for f in af}
                conditions.append({
                    "drift_type": "combined",
                    "features": cfg["marginal"],
                    "interaction_pairs": cfg["interaction"],
                    "magnitude": mag, "feature_types": ft,
                    "seed": seed, "feature_group": "high_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_comb_high_cfg{i+1}_{mag}_{seed}"
                })
            for i, cfg in enumerate(cfgs_l):
                af = cfg["marginal"] + [f for p in cfg["interaction"] for f in p]
                ft = {f: feature_types.get(f,'continuous') for f in af}
                conditions.append({
                    "drift_type": "combined",
                    "features": cfg["marginal"],
                    "interaction_pairs": cfg["interaction"],
                    "magnitude": mag, "feature_types": ft,
                    "seed": seed, "feature_group": "low_importance",
                    "dataset": dataset_name,
                    "condition_id": f"{dataset_name}_comb_low_cfg{i+1}_{mag}_{seed}"
                })

    # ── NO-DRIFT BASELINES (200) ──────────────────────────────────────────────
    for seed in range(50):
        for batch_size in NODRIFT_BATCH_SIZES:
            conditions.append({
                "drift_type": "none", "features": [],
                "magnitude": 0.0, "batch_size": batch_size,
                "seed": seed, "feature_group": "none",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_none_{batch_size}_{seed}"
            })

    # Verify
    print(f"\nCondition count verification for {dataset_name}:")
    by_type = {}
    for c in conditions:
        by_type[c['drift_type']] = by_type.get(c['drift_type'], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:<20} {n}")
    print(f"  {'TOTAL':<20} {len(conditions)}")
    assert len(conditions) == 1600, \
        f"Expected 1600, got {len(conditions)}"
    print(f"  ASSERTION PASSED: exactly 1,600 conditions")

    return conditions


# ── BENCHMARK RUNNER ──────────────────────────────────────────────────────────

def run_benchmark(reference_df, model, high_feats, low_feats,
                  feature_types, dataset_name, save_path,
                  target_col='target'):

    print("=" * 70)
    print(f"BENCHMARK: {dataset_name.upper()} | FRAMEWORK v1.0 FROZEN")
    print("=" * 70)

    conditions = build_condition_list(
        high_feats, low_feats, feature_types, dataset_name)

    checkpoint_path = save_path.replace('.csv', '_checkpoint.csv')
    completed_ids = set()
    rows = []

    if os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        completed_ids = set(ckpt['condition_id'].values)
        rows = ckpt.to_dict('records')
        print(f"\nResuming from checkpoint: {len(completed_ids)} done")

    remaining = [c for c in conditions
                 if c['condition_id'] not in completed_ids]
    print(f"Remaining: {len(remaining)}\n")

    start_time = time.time()
    errors = []

    for i, cond in enumerate(remaining):
        t0 = time.time()
        try:
            result = inject_drift(
                reference_df=reference_df, model=model,
                drift_type=cond['drift_type'], features=cond['features'],
                magnitude=cond['magnitude'],
                feature_types=cond.get('feature_types', {}),
                interaction_pairs=cond.get('interaction_pairs'),
                batch_size=cond.get('batch_size'),
                target_col=target_col, seed=cond['seed']
            )
            drifted_df = result['drifted_df']
            gt = result['ground_truth']
            det = run_all_detectors(reference_df, drifted_df, target_col)
            mc, cs, pl = compute_confidence_statistics(model, drifted_df, target_col)

            rows.append({
                "condition_id":         cond['condition_id'],
                "dataset":              dataset_name,
                "drift_type":           cond['drift_type'],
                "magnitude":            cond['magnitude'],
                "feature_group":        cond.get('feature_group','none'),
                "injected_features":    str(gt['injected_features']),
                "seed":                 cond['seed'],
                "ks_score":             round(det['ks_score'],   6),
                "chi2_score":           round(det['chi2_score'], 6),
                "psi_score":            round(det['psi_score'],  6),
                "mmd_score":            round(det['mmd_score'],  6),
                "lsdd_score":           round(det['lsdd_score'], 6),
                "clf_auc":              round(det['clf_auc'],    6),
                "mean_confidence":      round(mc,                6),
                "confidence_std":       round(cs,                6),
                "pct_low_conf":         round(pl,                6),
                "actual_accuracy_drop": round(gt['actual_accuracy_drop'], 4),
                "baseline_accuracy":    round(gt['baseline_accuracy'],    4),
                "drifted_accuracy":     round(gt['drifted_accuracy'],     4),
                "baseline_auc":         gt['baseline_auc'],
                "drifted_auc":          gt['drifted_auc'],
                "is_malignant":         gt['is_malignant'],
                "severity":             gt['severity'],
                "condition_time_sec":   round(time.time() - t0, 2)
            })

            elapsed = time.time() - start_time
            eta = elapsed / (i+1) * (len(remaining)-i-1) / 60

            if (i+1) % 10 == 0 or i == 0:
                print(f"  [{i+1:4d}/{len(remaining)}] "
                      f"drop={gt['actual_accuracy_drop']:+6.2f}% "
                      f"ks={det['ks_score']:.3f} "
                      f"clf={det['clf_auc']:.3f} "
                      f"ETA:{eta:.0f}m "
                      f"{cond['condition_id'][:40]}")

            if (i+1) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                print(f"  ── checkpoint: {len(rows)} saved ──")

        except Exception as e:
            errors.append({"id": cond['condition_id'], "err": str(e)})
            print(f"  ERROR {cond['condition_id']}: {e}")

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(save_path, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"COMPLETE: {dataset_name.upper()}")
    print(f"  Conditions:  {len(benchmark_df)} / 1,600")
    print(f"  Errors:      {len(errors)}")
    print(f"  Total time:  {total/3600:.2f} hours")
    print(f"\nDrift type breakdown:")
    print(benchmark_df['drift_type'].value_counts().to_string())
    print(f"\nAccuracy drop stats:")
    print(benchmark_df['actual_accuracy_drop'].describe().round(3).to_string())
    print(f"\nMalignant: {benchmark_df['is_malignant'].sum()} "
          f"({benchmark_df['is_malignant'].mean()*100:.1f}%)")
    print(f"\nSaved: {save_path}")
    return benchmark_df


def run_benchmark_pipeline(dataset_name, data_dir, save_dir,
                           feature_importance_path, target_col='target'):

    model_files = {
        'uci_adult': 'base_model_adult.pkl',
        'ieee_cis':  'base_model_ieeecis.pkl',
        'elec2':     'base_model_elec2.pkl',
    }

    print(f"\nLoading artifacts: {dataset_name}")
    ref_df = pd.read_csv(f"{data_dir}/data/reference.csv")
    model  = joblib.load(f"{data_dir}/models/{model_files[dataset_name]}")

    with open(feature_importance_path) as f:
        imp = json.load(f)

    high_feats = imp['feature_groups']['high_importance']['features']
    low_feats  = imp['feature_groups']['low_importance']['features']

    feature_cols = [c for c in ref_df.columns if c != target_col]
    feature_types = {
        col: ('categorical' if ref_df[col].nunique() <= 15 else 'continuous')
        for col in feature_cols
    }

    os.makedirs(save_dir, exist_ok=True)
    return run_benchmark(
        reference_df=ref_df, model=model,
        high_feats=high_feats, low_feats=low_feats,
        feature_types=feature_types, dataset_name=dataset_name,
        save_path=f"{save_dir}/benchmark_{dataset_name}.csv",
        target_col=target_col
    )


if __name__ == "__main__":
    df_adult = run_benchmark_pipeline(
        'uci_adult', 'shared/uci_adult', 'shared/uci_adult',
        'shared/uci_adult/feature_importance_adult.json'
    )
    df_ieee = run_benchmark_pipeline(
        'ieee_cis', 'shared/ieee_cis', 'shared/ieee_cis',
        'shared/ieee_cis/feature_importance_ieeecis.json'
    )
    print(f"\nTotal: {len(df_adult) + len(df_ieee)} conditions")
