"""
Rescore the frozen v1.1 benchmark with the v1.2 C1 signals.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

WHAT THIS DOES
  The v1.1 CSV stores scores, not the drifted batches. Every condition's
  batch_seed is stored, so each batch is RECREATED exactly with the shared
  v1.1 injection code. Before any new score is trusted, the recreated batch
  must reproduce the v1.1 CSV:
      targets:      drifted_accuracy, harm_paired, actual_accuracy_drop
      old signals:  mean_pos_prob, pos_prob_std, pct_pos_prob_lt_06,
                    ks_score, chi2_score, psi_score (v1.1 functions)
  Any target mismatch stops the run. Then the v1.2 signals are computed:
      detectors (type-aware, per feature, mean + max) and max-prob confidence.

  Conditions, injections and targets are UNCHANGED from v1.1.
  Old v1.1 detector columns are kept with a "v11_" prefix for comparison.

USAGE (Colab)
  python c1/scripts/rescore_benchmark.py --shared_dir shared \
      --ref .../reference.csv --test .../test.csv --model .../base_model_adult.pkl \
      --feature_json .../feature_importance_adult_v11.json \
      --bench_csv .../benchmark_uci_adult_v11.csv \
      --out_csv .../c1_v12/benchmark_uci_adult_v12.csv [--verify_only 40]
"""

import argparse
import json
import os
import sys
import time
import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root -> "c1" package

from c1.src.profile import ReferenceProfile                       # noqa: E402
from c1.src.detectors import run_all_detectors, DETECTOR_SIGNALS   # noqa: E402
from c1.src.prediction_stats import (prediction_stats,             # noqa: E402
                                     CONFIDENCE_SIGNALS, POS_PROB_SIGNALS)

VERSION = "1.2"
TARGET_COLS = ["drifted_accuracy", "harm_paired", "actual_accuracy_drop"]
OLD_DETECTORS = ["ks_score", "chi2_score", "psi_score", "mmd_score", "lsdd_score", "clf_auc"]
# One unit in the last stored decimal (CSV rounds targets to 4 dp, signals to 6 dp)
TOL_4DP, TOL_6DP = 1.01e-4, 1.01e-6


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("--shared_dir", required=True)
    a.add_argument("--ref", required=True)
    a.add_argument("--test", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--feature_json", required=True)
    a.add_argument("--bench_csv", required=True)
    a.add_argument("--out_csv", required=True)
    a.add_argument("--target_col", default="target")
    a.add_argument("--verify_only", type=int, default=0,
                   help="check N conditions (spread over drift types), print timing, save nothing")
    a.add_argument("--checkpoint_every", type=int, default=50)
    a.add_argument("--no_lsdd", action="store_true", help="testing only")
    return a.parse_args()


def load_shared(shared_dir):
    sys.path.insert(0, os.path.abspath(shared_dir))
    import benchmark_loop as bl  # v1.1 (imports injection.inject_drift)
    assert bl.VERSION == "1.1", f"shared benchmark_loop is v{bl.VERSION}, expected 1.1"
    from injection.inject_drift import inject_drift
    return bl, inject_drift


def pick_verify_rows(bench, n):
    per_type = max(1, n // bench["drift_type"].nunique())
    idx = (bench.groupby("drift_type", group_keys=False)
                .apply(lambda g: g.iloc[np.linspace(0, len(g) - 1, min(per_type, len(g))).astype(int)])
                .index)
    return bench.loc[idx]


def main():
    args = parse_args()
    bl, inject_drift = load_shared(args.shared_dir)
    tc = args.target_col

    ref = pd.read_csv(args.ref)
    test = pd.read_csv(args.test)
    model = joblib.load(args.model)
    imp = json.load(open(args.feature_json))
    bench = pd.read_csv(args.bench_csv)
    feature_types = imp["feature_types"]
    high = imp["feature_groups"]["high_importance"]["features"]
    low = imp["feature_groups"]["low_importance"]["features"]
    dataset = bench["dataset"].iloc[0]
    print(f"v1.1 CSV: {len(bench)} rows | dataset={dataset} | version="
          f"{bench['benchmark_version'].unique().tolist()}")
    print("feature_types:", feature_types)

    # ── Pre-checks: same conditions, same seeds, same fixed baseline ───────
    conds = {c["condition_id"]: c for c in
             bl.build_condition_list(high, low, feature_types, dataset)}
    if set(conds) != set(bench["condition_id"]):
        raise SystemExit("STOP: condition list differs from the v1.1 CSV "
                         "(check the feature JSON is the one used for v1.1).")
    seeds = bench["condition_id"].map(lambda s: zlib.crc32(s.encode()) % (2 ** 31))
    if not (seeds.values == bench["batch_seed"].values).all():
        raise SystemExit("STOP: batch_seed in CSV does not match crc32(condition_id).")
    fixed = float(accuracy_score(test[tc].values, model.predict(test.drop(columns=[tc]))))
    if abs(round(fixed, 4) - bench["fixed_baseline_accuracy"].iloc[0]) > TOL_4DP:
        raise SystemExit(f"STOP: fixed baseline {fixed:.4f} != CSV "
                         f"{bench['fixed_baseline_accuracy'].iloc[0]} (wrong model or test file?)")
    print(f"Pre-checks passed: 1,600 conditions match, batch seeds match, "
          f"fixed baseline {fixed:.4f}")

    profile = ReferenceProfile(ref, feature_types, target_col=tc)
    print(f"Profile: {len(profile.continuous)} continuous {profile.continuous}")
    print(f"         {len(profile.categorical)} categorical {profile.categorical}")
    print(f"         MMD bandwidth {profile.mmd_sigma:.4f} on {profile.ref_kernel.shape} encoded rows")

    rf = ref.drop(columns=[tc])
    nc = rf.select_dtypes(include=[np.number]).columns.tolist()

    todo = pick_verify_rows(bench, args.verify_only) if args.verify_only else bench
    ckpt = args.out_csv.replace(".csv", "_checkpoint.csv")
    if not args.verify_only:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    rows, done = [], set()
    if not args.verify_only and os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        rows, done = prev.to_dict("records"), set(prev["condition_id"])
        print(f"Resuming from checkpoint: {len(done)} done")

    diffs = {c: 0.0 for c in TARGET_COLS + POS_PROB_SIGNALS + ["ks_score", "chi2_score",
                                                               "psi_score", "clf_auc"]}
    clf_mismatch = 0
    t_start, n_new = time.time(), 0
    remaining = todo[~todo["condition_id"].isin(done)]
    print(f"Conditions to process: {len(remaining)}\n")

    for _, old in remaining.iterrows():
        t0 = time.time()
        cond = conds[old["condition_id"]]
        res = inject_drift(reference_df=ref, model=model,
                           drift_type=cond["drift_type"], features=cond["features"],
                           magnitude=cond["magnitude"],
                           feature_types=cond.get("feature_types", {}),
                           interaction_pairs=cond.get("interaction_pairs"),
                           batch_size=cond["batch_size"], target_col=tc,
                           seed=int(old["batch_seed"]), source_df=test, paired=True)
        batch, gt = res["drifted_df"], res["ground_truth"]

        # ── Reproduction check (targets must match exactly) ───────────────
        new_t = {"drifted_accuracy": round(gt["drifted_accuracy"], 4),
                 "harm_paired": round(gt["actual_accuracy_drop"], 4),
                 "actual_accuracy_drop": round((fixed - gt["drifted_accuracy"]) * 100, 4)}
        for c, v in new_t.items():
            d = abs(v - old[c])
            diffs[c] = max(diffs[c], d)
            if d > TOL_4DP:
                raise SystemExit(f"STOP: {old['condition_id']} {c}: recreated {v} "
                                 f"vs v1.1 {old[c]}. Batches are NOT reproduced.")

        X = batch.drop(columns=[tc])
        ps = prediction_stats(model, X)
        old_sig = {"ks_score": bl.compute_ks_score(rf[nc], X[nc], nc),
                   "chi2_score": bl.compute_chi2_score(rf[nc], X[nc], nc),
                   "psi_score": bl.compute_psi_score(rf[nc], X[nc], nc)}
        for c in POS_PROB_SIGNALS:
            old_sig[c] = ps[c]
        for c, v in old_sig.items():
            d = abs(round(v, 6) - old[c])
            diffs[c] = max(diffs[c], d)
            if d > TOL_6DP:
                raise SystemExit(f"STOP: {old['condition_id']} {c}: recreated "
                                 f"{round(v, 6)} vs v1.1 {old[c]}.")

        # ── v1.2 signals ───────────────────────────────────────────────────
        det, per = run_all_detectors(profile, X, seed=int(old["batch_seed"]),
                                     use_lsdd=not args.no_lsdd)
        d = abs(round(det["clf_auc"], 6) - old["clf_auc"])
        diffs["clf_auc"] = max(diffs["clf_auc"], d)
        clf_mismatch += int(d > TOL_6DP)

        row = {k: old[k] for k in bench.columns if k not in OLD_DETECTORS}
        row.update({f"v11_{k}": old[k] for k in OLD_DETECTORS})
        row.update({k: round(det[k], 6) for k in DETECTOR_SIGNALS})
        row.update({k: round(ps[k], 6) for k in CONFIDENCE_SIGNALS})
        for name, scores in per.items():
            row.update({f"{name}__{f}": round(v, 6) for f, v in scores.items()})
        row["benchmark_version"] = VERSION
        row["rescore_time_sec"] = round(time.time() - t0, 2)
        rows.append(row)
        n_new += 1

        if n_new == 1 or n_new % 10 == 0:
            el = time.time() - t_start
            eta = el / n_new * (len(remaining) - n_new) / 60
            print(f"  [{n_new:4d}/{len(remaining)}] {old['condition_id'][:40]:40s} "
                  f"ks_max={det['ks_max']:.3f} chi2_max={det['chi2_max']:.3f} "
                  f"conf={ps['conf_mean']:.3f} ETA {eta:.1f}m")
        if not args.verify_only and n_new % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)

    el = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"REPRODUCTION CHECK PASSED on {n_new} conditions "
          f"(largest differences vs v1.1, 0 means identical):")
    for c, v in diffs.items():
        print(f"  {c:24s} {v:.2e}")
    print(f"clf_auc differs from v1.1 in {clf_mismatch}/{n_new} conditions "
          f"(same code; small differences can come from xgboost threading)")
    per_cond = el / max(n_new, 1)
    print(f"Time: {el/60:.1f} min, {per_cond:.2f} s per condition "
          f"-> full 1,600 run about {per_cond*1600/60:.0f} min")

    if args.verify_only:
        print("\nverify_only: nothing saved.")
        return
    out = pd.DataFrame(rows)
    assert len(out) == len(bench) and out["condition_id"].is_unique
    out = out.set_index("condition_id").loc[bench["condition_id"]].reset_index()
    out.to_csv(args.out_csv, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"\nSaved {len(out)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
