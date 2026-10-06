"""
Benchmark Data Collection Loop - v1.1
J26-DS-361 | C1 Research | Sajivan K (IT23172296)

VERSION: 1.1  (supersedes v1.0; v1.0 CSVs are kept untouched and are NOT used
              for any thesis number - see audit notes below)

WHAT CHANGED FROM v1.0 AND WHY (all found in the audit):
  1. Batch size is no longer a shortcut. v1.0: every drift batch was 26,048
     rows (the whole reference set) while no-drift batches were 500-5,000, so
     batch size alone separated "none" from "drift". v1.1: every condition,
     including none, uses the SAME balanced batch-size set (350 drift +
     50 none conditions per size). batch_size is saved as a column.
  2. Production batches are drawn from UNSEEN data (test.csv), not from the
     rows the model was trained on. Detectors compare reference.csv (what the
     model saw) with the production batch (what arrives later).
  3. Harm is measured against a FIXED held-out baseline (model accuracy on the
     full test set) -> column `actual_accuracy_drop` (pp). This keeps honest
     sampling noise in the target (batch accuracy really does wobble) which the
     meta-learner can learn from batch_size. The PAIRED harm (same rows before
     vs after injection, noise cancelled) is saved as `harm_paired` for
     sensitivity analysis.
  4. The three "confidence" signals are renamed to what they are: statistics of
     P(positive class). They are NOT calibrated confidence.
        mean_confidence  -> mean_pos_prob
        confidence_std   -> pos_prob_std
        pct_low_conf     -> pct_pos_prob_lt_06
  5. MMD and LSDD were the SAME formula in v1.0 (identical columns). v1.1 keeps
     MMD (RBF, biased MMD^2) and takes LSDD from alibi-detect's LSDDDrift, so
     the two are genuinely different estimators. If alibi-detect is not
     installed the run STOPS (no silent fallback to 0.0).
  6. Detector subsamples are seeded per condition (v1.0 used random_state=42
     for every condition).
  7. Every condition draws its OWN production batch. A first v1.1 draft seeded
     the batch only by (seed, batch_size), so 1,400 drift conditions reused
     just 20 distinct batches and their sampling noise was shared. Now
     batch_seed = crc32(condition_id) (stored in the CSV), so each condition
     gets its own batch rows and its own injection randomness. The original
     `seed` column is kept for bookkeeping.

TARGET: 1,600 conditions per dataset (unchanged)
    Marginal Single 500 | Marginal Multi 300 | Interaction 300 | Combined 300
    No-drift 200 (50 seeds x 4 batch sizes)

BATCH SIZES: [500, 1000, 2000, 3000]  (3000 is capped by UCI test = 6,513 rows;
    IEEE-CIS test has 40,000 rows and may use a larger set, passed in).
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import time
import zlib
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score
from injection.inject_drift import inject_drift


# ── FRAMEWORK PARAMETERS ─────────────────────────────────────────────────────
VERSION = "1.1"
MAGNITUDES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 123, 456, 789, 1234]
N_HIGH_FEATURES = 5
N_LOW_FEATURES = 5
N_NODRIFT_SEEDS = 50
BATCH_SIZES = [500, 1000, 2000, 3000]
CHECKPOINT_EVERY = 50
CLASSIFIER_SUBSAMPLE = 1000
KERNEL_SUBSAMPLE = 500
LSDD_PERMUTATIONS = 50

SIGNAL_COLS = ["ks_score", "chi2_score", "psi_score", "mmd_score",
               "lsdd_score", "clf_auc",
               "mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]


# ── DETECTORS ────────────────────────────────────────────────────────────────

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


def _subsample_standardised(ref_features, drift_features, n_sample, seed):
    n = min(n_sample, len(ref_features), len(drift_features))
    rs = ref_features.sample(n=n, random_state=seed).values.astype(float)
    ds = drift_features.sample(n=n, random_state=seed + 1).values.astype(float)
    mu = rs.mean(0)
    std = rs.std(0) + 1e-10
    return (rs - mu) / std, (ds - mu) / std


def compute_mmd_score(ref_features, drift_features,
                      n_sample=KERNEL_SUBSAMPLE, sigma=1.0, seed=0):
    """Biased RBF MMD^2 on standardised numeric features."""
    rn, dn = _subsample_standardised(ref_features, drift_features, n_sample, seed)

    def rbf(X, Y):
        d = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        return np.exp(-np.sum(d ** 2, axis=2) / (2 * sigma ** 2))

    return float(max(0.0, rbf(rn, rn).mean() + rbf(dn, dn).mean()
                     - 2 * rbf(rn, dn).mean()))


def compute_lsdd_score(ref_features, drift_features,
                       n_sample=KERNEL_SUBSAMPLE, seed=0):
    """
    LSDD distance from alibi-detect (a different estimator from MMD).
    No silent fallback: if alibi-detect is missing this raises.
    """
    from alibi_detect.cd import LSDDDrift
    rn, dn = _subsample_standardised(ref_features, drift_features, n_sample, seed)
    cd = LSDDDrift(rn.astype(np.float32), backend='pytorch',
                   p_val=0.05, n_permutations=LSDD_PERMUTATIONS)
    out = cd.predict(dn.astype(np.float32))
    return float(np.ravel(out['data']['distance'])[0])


def compute_classifier_auc(ref_features, drift_features,
                           n_sample=CLASSIFIER_SUBSAMPLE, seed=0):
    """XGBoost classifier two-sample test, 3-fold CV AUC (0.5 = no drift)."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    n = min(n_sample, len(ref_features), len(drift_features))
    X = pd.concat([ref_features.sample(n=n, random_state=seed),
                   drift_features.sample(n=n, random_state=seed + 1)],
                  ignore_index=True)
    y = np.array([0] * n + [1] * n)
    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                        eval_metric='logloss', random_state=42, verbosity=0)
    return float(np.mean(cross_val_score(clf, X, y, cv=3,
                                         scoring='roc_auc', n_jobs=-1)))


def compute_prediction_statistics(model, drifted_df, target_col='target'):
    """
    Statistics of P(positive class) on the production batch.
    NOT calibrated confidence. Blind to harm that does not change the
    predicted-positive rate.
    """
    proba = model.predict_proba(drifted_df.drop(columns=[target_col]))[:, 1]
    return float(np.mean(proba)), float(np.std(proba)), float(np.mean(proba < 0.6))


def run_all_detectors(reference_df, drifted_df, target_col='target', seed=0):
    rf = reference_df.drop(columns=[target_col])
    df = drifted_df.drop(columns=[target_col])
    nc = rf.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "ks_score":   compute_ks_score(rf[nc], df[nc], nc),
        "chi2_score": compute_chi2_score(rf[nc], df[nc], nc),
        "psi_score":  compute_psi_score(rf[nc], df[nc], nc),
        "mmd_score":  compute_mmd_score(rf[nc], df[nc], seed=seed),
        "lsdd_score": compute_lsdd_score(rf[nc], df[nc], seed=seed),
        "clf_auc":    compute_classifier_auc(rf[nc], df[nc], seed=seed),
    }


# ── CONDITION BUILDER ─────────────────────────────────────────────────────────

def build_condition_list(high_feats, low_feats, feature_types, dataset_name,
                         batch_sizes=BATCH_SIZES):
    """
    Build the 1,600-condition list. Drift conditions are assigned batch sizes
    round-robin so each size gets exactly 350 drift conditions; the 200
    no-drift conditions get 50 per size. Asserts exactly 1,600.
    """
    conditions = []
    hf = high_feats[:N_HIGH_FEATURES]
    lf = low_feats[:N_LOW_FEATURES]
    counter = [0]

    def add(cond):
        cond["batch_size"] = batch_sizes[counter[0] % len(batch_sizes)]
        counter[0] += 1
        cond["dataset"] = dataset_name
        conditions.append(cond)

    # ── MARGINAL SINGLE (500) ────────────────────────────────────────────────
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for grp, feats, tag in (("high_importance", hf, "high"),
                                    ("low_importance", lf, "low")):
                for feat in feats:
                    add({"drift_type": "marginal_single", "features": [feat],
                         "magnitude": mag,
                         "feature_types": {feat: feature_types.get(feat, 'continuous')},
                         "seed": seed, "feature_group": grp,
                         "condition_id": f"{dataset_name}_ms_{tag}_{feat}_{mag}_{seed}"})

    # ── MARGINAL MULTI (300) ─────────────────────────────────────────────────
    multi = {"high": [hf[:2], hf[:3], hf[:5]], "low": [lf[:2], lf[:3], lf[:5]]}
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for tag, grp in (("high", "high_importance"), ("low", "low_importance")):
                for i, combo in enumerate(multi[tag]):
                    ft = {f: feature_types.get(f, 'continuous') for f in combo}
                    add({"drift_type": "marginal_multi", "features": combo,
                         "magnitude": mag, "feature_types": ft,
                         "seed": seed, "feature_group": grp,
                         "condition_id": f"{dataset_name}_mm_{tag}_c{i+1}_{mag}_{seed}"})

    # ── INTERACTION (300) ────────────────────────────────────────────────────
    pairs = {"high": [(hf[0], hf[1]), (hf[0], hf[2]), (hf[1], hf[2])],
             "low":  [(lf[0], lf[1]), (lf[0], lf[2]), (lf[1], lf[2])]}
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for tag, grp in (("high", "high_importance"), ("low", "low_importance")):
                for i, (fa, fb) in enumerate(pairs[tag]):
                    ft = {fa: feature_types.get(fa, 'continuous'),
                          fb: feature_types.get(fb, 'continuous')}
                    add({"drift_type": "interaction", "features": [fa, fb],
                         "magnitude": mag, "feature_types": ft,
                         "seed": seed, "feature_group": grp,
                         "condition_id": f"{dataset_name}_int_{tag}_p{i+1}_{mag}_{seed}"})

    # ── COMBINED (300) ───────────────────────────────────────────────────────
    def cfgs(f):
        return [{"marginal": [f[0]], "interaction": [(f[1], f[2])]},
                {"marginal": [f[1]], "interaction": [(f[0], f[2])]},
                {"marginal": [f[2]], "interaction": [(f[0], f[1])]}]
    comb = {"high": cfgs(hf), "low": cfgs(lf)}
    for seed in SEEDS:
        for mag in MAGNITUDES:
            for tag, grp in (("high", "high_importance"), ("low", "low_importance")):
                for i, cfg in enumerate(comb[tag]):
                    af = cfg["marginal"] + [f for p in cfg["interaction"] for f in p]
                    ft = {f: feature_types.get(f, 'continuous') for f in af}
                    add({"drift_type": "combined", "features": cfg["marginal"],
                         "interaction_pairs": cfg["interaction"],
                         "magnitude": mag, "feature_types": ft,
                         "seed": seed, "feature_group": grp,
                         "condition_id": f"{dataset_name}_comb_{tag}_cfg{i+1}_{mag}_{seed}"})

    # ── NO-DRIFT (200) ───────────────────────────────────────────────────────
    for seed in range(N_NODRIFT_SEEDS):
        for bs in batch_sizes:
            conditions.append({
                "drift_type": "none", "features": [], "magnitude": 0.0,
                "batch_size": bs, "seed": seed, "feature_group": "none",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_none_{bs}_{seed}"})

    print(f"\nCondition count verification for {dataset_name} (v{VERSION}):")
    by_type = {}
    for c in conditions:
        by_type[c['drift_type']] = by_type.get(c['drift_type'], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:<20} {n}")
    print(f"  {'TOTAL':<20} {len(conditions)}")
    assert len(conditions) == 1600, f"Expected 1600, got {len(conditions)}"
    ids = [c['condition_id'] for c in conditions]
    assert len(ids) == len(set(ids)), "duplicate condition_id"
    bs_counts = pd.Series([c['batch_size'] for c in conditions]).value_counts()
    print("  batch-size balance (all conditions):", bs_counts.sort_index().to_dict())
    return conditions


# ── BENCHMARK RUNNER ──────────────────────────────────────────────────────────

def run_benchmark(reference_df, test_df, model, high_feats, low_feats,
                  feature_types, dataset_name, save_path,
                  target_col='target', batch_sizes=BATCH_SIZES):
    """
    reference_df: data the model was trained on (detector reference)
    test_df:      unseen data; production batches are sampled from it
    """
    print("=" * 70)
    print(f"BENCHMARK: {dataset_name.upper()} | FRAMEWORK v{VERSION}")
    print("=" * 70)
    assert max(batch_sizes) <= len(test_df), \
        f"largest batch {max(batch_sizes)} > test rows {len(test_df)}"

    # Fixed held-out baseline: model accuracy on the full test set
    X_test = test_df.drop(columns=[target_col])
    fixed_baseline_acc = float(accuracy_score(test_df[target_col].values,
                                              model.predict(X_test)))
    print(f"Fixed held-out baseline accuracy (test, n={len(test_df)}): "
          f"{fixed_baseline_acc:.4f}")

    conditions = build_condition_list(high_feats, low_feats, feature_types,
                                      dataset_name, batch_sizes)

    checkpoint_path = save_path.replace('.csv', '_checkpoint.csv')
    completed_ids = set()
    rows = []
    if os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        completed_ids = set(ckpt['condition_id'].values)
        rows = ckpt.to_dict('records')
        print(f"\nResuming from checkpoint: {len(completed_ids)} done")

    remaining = [c for c in conditions if c['condition_id'] not in completed_ids]
    print(f"Remaining: {len(remaining)}\n")

    start_time = time.time()
    errors = []

    for i, cond in enumerate(remaining):
        t0 = time.time()
        batch_seed = zlib.crc32(cond['condition_id'].encode()) % (2 ** 31)
        try:
            result = inject_drift(
                reference_df=reference_df, model=model,
                drift_type=cond['drift_type'], features=cond['features'],
                magnitude=cond['magnitude'],
                feature_types=cond.get('feature_types', {}),
                interaction_pairs=cond.get('interaction_pairs'),
                batch_size=cond['batch_size'],
                target_col=target_col, seed=batch_seed,
                source_df=test_df, paired=True
            )
            drifted_df = result['drifted_df']
            gt = result['ground_truth']
            det = run_all_detectors(reference_df, drifted_df, target_col,
                                    seed=batch_seed)
            mp, ps, pl = compute_prediction_statistics(model, drifted_df, target_col)

            harm_fixed = (fixed_baseline_acc - gt['drifted_accuracy']) * 100
            row = {
                "condition_id":       cond['condition_id'],
                "dataset":            dataset_name,
                "drift_type":         cond['drift_type'],
                "magnitude":          cond['magnitude'],
                "feature_group":      cond.get('feature_group', 'none'),
                "injected_features":  str(gt['injected_features']),
                "seed":               cond['seed'],
                "batch_seed":         batch_seed,
                "batch_size":         len(drifted_df),
                **{k: round(v, 6) for k, v in det.items()},
                "mean_pos_prob":      round(mp, 6),
                "pos_prob_std":       round(ps, 6),
                "pct_pos_prob_lt_06": round(pl, 6),
                # PRIMARY target: harm vs fixed held-out baseline (pp)
                "actual_accuracy_drop": round(harm_fixed, 4),
                "fixed_baseline_accuracy": round(fixed_baseline_acc, 4),
                # SENSITIVITY target: same rows before vs after injection (pp)
                "harm_paired":        round(gt['actual_accuracy_drop'], 4),
                "paired_baseline_accuracy": round(gt['baseline_accuracy'], 4),
                "drifted_accuracy":   round(gt['drifted_accuracy'], 4),
                "baseline_auc_paired": gt['baseline_auc'],
                "drifted_auc":        gt['drifted_auc'],
                "is_malignant":       bool(harm_fixed > 2.0),
                "is_malignant_paired": bool(gt['actual_accuracy_drop'] > 2.0),
                "benchmark_version":  VERSION,
                "condition_time_sec": round(time.time() - t0, 2),
            }
            rows.append(row)

            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(remaining) - i - 1) / 60
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:4d}/{len(remaining)}] "
                      f"drop={harm_fixed:+6.2f} paired={gt['actual_accuracy_drop']:+6.2f} "
                      f"ks={det['ks_score']:.3f} clf={det['clf_auc']:.3f} "
                      f"bs={len(drifted_df)} ETA:{eta:.0f}m "
                      f"{cond['condition_id'][:38]}")
            if (i + 1) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                print(f"  -- checkpoint: {len(rows)} saved --")

        except ImportError:
            # Missing library (e.g. alibi-detect): stop, never write zeros.
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            raise
        except Exception as e:
            errors.append({"id": cond['condition_id'], "err": str(e)})
            print(f"  ERROR {cond['condition_id']}: {e}")

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(save_path, index=False)
    if os.path.exists(checkpoint_path) and not errors:
        os.remove(checkpoint_path)

    total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"COMPLETE: {dataset_name.upper()}  v{VERSION}")
    print(f"  Conditions: {len(benchmark_df)} / 1,600   Errors: {len(errors)}")
    print(f"  Total time: {total/3600:.2f} hours")
    print(benchmark_df['drift_type'].value_counts().to_string())
    print("\nactual_accuracy_drop (fixed baseline):")
    print(benchmark_df['actual_accuracy_drop'].describe().round(3).to_string())
    print(f"\nSaved: {save_path}")
    return benchmark_df


def run_benchmark_pipeline(dataset_name, data_dir, save_dir,
                           feature_importance_path, target_col='target',
                           batch_sizes=BATCH_SIZES, save_name=None):
    model_files = {
        'uci_adult': 'base_model_adult.pkl',
        'ieee_cis':  'base_model_ieeecis.pkl',
        'elec2':     'base_model_elec2.pkl',
    }
    print(f"\nLoading artifacts: {dataset_name}")
    ref_df = pd.read_csv(f"{data_dir}/data/reference.csv")
    test_df = pd.read_csv(f"{data_dir}/data/test.csv")
    model = joblib.load(f"{data_dir}/models/{model_files[dataset_name]}")

    with open(feature_importance_path) as f:
        imp = json.load(f)
    high_feats = imp['feature_groups']['high_importance']['features']
    low_feats = imp['feature_groups']['low_importance']['features']

    # Use the saved feature types when present (v1.0 guessed by nunique<=15,
    # which mislabelled marital_status / native_country).
    if 'feature_types' in imp:
        feature_types = imp['feature_types']
    else:
        feature_cols = [c for c in ref_df.columns if c != target_col]
        feature_types = {c: ('categorical' if ref_df[c].nunique() <= 15
                             else 'continuous') for c in feature_cols}
        print("WARNING: feature_types not in importance JSON; guessed by nunique")

    os.makedirs(save_dir, exist_ok=True)
    name = save_name or f"benchmark_{dataset_name}_v11.csv"
    return run_benchmark(ref_df, test_df, model, high_feats, low_feats,
                         feature_types, dataset_name,
                         f"{save_dir}/{name}", target_col, batch_sizes)
