"""
Rescore the LSDD signal of the UCI Adult v1.2 benchmark with the seeded LSDD (engine 1.2.2).
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

WHY
  The UCI v1.2 benchmark was scored before the LSDD fix (commit e78fe50). alibi-detect's
  LSDDDrift picks its kernel centres with torch.randperm when it is built, so the old
  UCI LSDD column depends on the Colab session. The detector is now built inside
  torch.random.fork_rng with LSDD_INIT_SEED = 0. Only the LSDD column changes.

WHAT THIS DOES
  1. Recreates every batch exactly (stored batch_seed, shared inject_drift, paired=True).
  2. Reproduction check, STOP on any mismatch:
       targets  drifted_accuracy, harm_paired, actual_accuracy_drop   (4 dp)
       signals  ks_max, psi_max                                       (6 dp, deterministic)
     Reported, not stopping: chi2_max and mmd (should also match), old lsdd (expected to differ).
  3. Determinism check: rebuilds the LSDD detector from scratch and rescores 3 batches;
     the scores must be identical.
  4. Writes a NEW csv: same rows and columns, lsdd replaced, the old value kept as
     lsdd_unseeded_v121. The original CSV is not touched.

USAGE (Colab, repo root as cwd)
  python c1/scripts/rescore_lsdd_uci_v12.py --shared_dir shared \
      --ref .../uci_adult_data/reference.csv --test .../uci_adult_data/test.csv \
      --model .../uci_adult_models/base_model_adult.pkl \
      --feature_json .../c1_v11/feature_importance_adult_v11.json \
      --bench_csv .../c1_v12/benchmark_uci_adult_v12.csv \
      --out_csv .../c1_v12/benchmark_uci_adult_v12_lsddseeded.csv [--verify_only 40]
"""
import argparse
import importlib
import json
import os
import sys
import time
import zlib

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root -> "c1" package

from c1.src.profile import ReferenceProfile                  # noqa: E402
from c1.src.detectors.ks_detector import ks_detector          # noqa: E402
from c1.src.detectors.chi2_detector import chi2_detector      # noqa: E402
from c1.src.detectors.psi_detector import psi_detector        # noqa: E402
from c1.src.detectors.mmd_detector import mmd_detector        # noqa: E402

L = importlib.import_module("c1.src.detectors.lsdd_detector")
TOL_4DP, TOL_6DP = 1.01e-4, 1.01e-6
TARGETS = ["drifted_accuracy", "harm_paired", "actual_accuracy_drop"]
STRICT = ["ks_max", "psi_max"]
SOFT = ["chi2_max", "mmd"]


def parse_args():
    a = argparse.ArgumentParser()
    for k in ["shared_dir", "ref", "test", "model", "feature_json", "bench_csv", "out_csv"]:
        a.add_argument(f"--{k}", required=True)
    a.add_argument("--target_col", default="target")
    a.add_argument("--verify_only", type=int, default=0,
                   help="check N conditions (spread over drift types), print timing, save nothing")
    a.add_argument("--checkpoint_every", type=int, default=50)
    return a.parse_args()


def pick_verify_rows(bench, n):
    per_type = max(1, n // bench["drift_type"].nunique())
    idx = (bench.groupby("drift_type", group_keys=False)
                .apply(lambda g: g.iloc[np.linspace(0, len(g) - 1, min(per_type, len(g))).astype(int)])
                .index)
    return bench.loc[idx]


def main():
    a = parse_args()
    assert getattr(L, "LSDD_INIT_SEED", None) == 0, \
        "lsdd_detector.py is not the seeded version (pull commit e78fe50 first)"
    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    from injection.inject_drift import inject_drift
    tc = a.target_col

    ref, test = pd.read_csv(a.ref), pd.read_csv(a.test)
    model = joblib.load(a.model)
    imp = json.load(open(a.feature_json))
    bench = pd.read_csv(a.bench_csv)
    assert len(bench) == 1600 and (bench["benchmark_version"].astype(str) == "1.2").all()
    assert "lsdd_unseeded_v121" not in bench.columns, "this CSV was already rescored"
    hi = imp["feature_groups"]["high_importance"]["features"]
    lo = imp["feature_groups"]["low_importance"]["features"]
    dataset = bench["dataset"].iloc[0]
    sizes = sorted(int(b) for b in bench["batch_size"].unique())
    print(f"UCI v1.2 CSV: {len(bench)} rows | dataset={dataset} | batch sizes {sizes}")

    # ── Pre-checks ───────────────────────────────────────────────────────────
    try:
        cl = bl.build_condition_list(hi, lo, imp["feature_types"], dataset, batch_sizes=sizes)
    except TypeError:  # older benchmark_loop without the batch_sizes argument
        cl = bl.build_condition_list(hi, lo, imp["feature_types"], dataset)
    conds = {c["condition_id"]: c for c in cl}
    if set(conds) != set(bench["condition_id"]):
        raise SystemExit("STOP: condition list differs from the CSV (wrong feature JSON?)")
    seeds = bench["condition_id"].map(lambda s: zlib.crc32(s.encode()) % (2 ** 31))
    if not (seeds.values == bench["batch_seed"].values).all():
        raise SystemExit("STOP: batch_seed in CSV does not match crc32(condition_id)")
    fixed = float(accuracy_score(test[tc].values, model.predict(test.drop(columns=[tc]))))
    if abs(round(fixed, 4) - bench["fixed_baseline_accuracy"].iloc[0]) > TOL_4DP:
        raise SystemExit(f"STOP: fixed baseline {fixed:.4f} != CSV "
                         f"{bench['fixed_baseline_accuracy'].iloc[0]} (wrong model or test file?)")
    print(f"Pre-checks passed: 1,600 conditions match, seeds match, fixed baseline {fixed:.4f}")

    profile = ReferenceProfile(ref, imp["feature_types"], target_col=tc)

    def make_batch(cid):
        c = conds[cid]
        res = inject_drift(reference_df=ref, model=model, drift_type=c["drift_type"],
                           features=c["features"], magnitude=c["magnitude"],
                           feature_types=c.get("feature_types", {}),
                           interaction_pairs=c.get("interaction_pairs"),
                           batch_size=c["batch_size"], target_col=tc,
                           seed=int(bench.set_index("condition_id").at[cid, "batch_seed"]),
                           source_df=test, paired=True)
        return res["drifted_df"], res["ground_truth"]

    todo = pick_verify_rows(bench, a.verify_only) if a.verify_only else bench
    ckpt = a.out_csv.replace(".csv", "_checkpoint.csv")
    new_lsdd = {}
    if not a.verify_only and os.path.exists(ckpt):
        new_lsdd = dict(pd.read_csv(ckpt).values)
        print(f"Resuming from checkpoint: {len(new_lsdd)} done")
    remaining = todo[~todo["condition_id"].isin(new_lsdd)]
    print(f"Conditions to process: {len(remaining)}\n")

    diffs = {c: 0.0 for c in TARGETS + STRICT + SOFT}
    soft_mismatch = {c: 0 for c in SOFT}
    t_start, n = time.time(), 0
    for _, old in remaining.iterrows():
        cid, seed = old["condition_id"], int(old["batch_seed"])
        batch, gt = make_batch(cid)
        new_t = {"drifted_accuracy": round(gt["drifted_accuracy"], 4),
                 "harm_paired": round(gt["actual_accuracy_drop"], 4),
                 "actual_accuracy_drop": round((fixed - gt["drifted_accuracy"]) * 100, 4)}
        for c, v in new_t.items():
            d = abs(v - old[c])
            diffs[c] = max(diffs[c], d)
            if d > TOL_4DP:
                raise SystemExit(f"STOP: {cid} {c}: recreated {v} vs CSV {old[c]}")
        X = batch.drop(columns=[tc])
        sig = {**ks_detector(profile, X)[0], **psi_detector(profile, X)[0],
               **chi2_detector(profile, X, seed=seed)[0], **mmd_detector(profile, X, seed=seed)}
        for c in STRICT + SOFT:
            d = abs(round(sig[c], 6) - old[c])
            diffs[c] = max(diffs[c], d)
            if d > TOL_6DP:
                if c in STRICT:
                    raise SystemExit(f"STOP: {cid} {c}: recreated {round(sig[c], 6)} vs CSV {old[c]}")
                soft_mismatch[c] += 1
        new_lsdd[cid] = round(L.lsdd_detector(profile, X, seed=seed)["lsdd"], 6)
        n += 1
        if n == 1 or n % 25 == 0:
            el = time.time() - t_start
            print(f"  [{n:4d}/{len(remaining)}] {cid[:44]:44s} lsdd old {old['lsdd']:.4f} "
                  f"new {new_lsdd[cid]:.4f}  ETA {el / n * (len(remaining) - n) / 60:.1f}m", flush=True)
        if not a.verify_only and n % a.checkpoint_every == 0:
            pd.DataFrame(list(new_lsdd.items()), columns=["condition_id", "lsdd"]).to_csv(ckpt, index=False)

    el = time.time() - t_start
    print("\n" + "=" * 78 + f"\nREPRODUCTION CHECK PASSED on {n} conditions (largest difference vs CSV):")
    for c, v in diffs.items():
        print(f"  {c:22s} {v:.2e}" + (f"   ({soft_mismatch[c]} above 1e-6)" if c in SOFT else ""))

    # ── Determinism: a freshly built detector must give the same scores ──────
    L._CACHE.clear()
    chk = list(todo["condition_id"].iloc[:3])
    det_diff = 0.0
    for cid in chk:
        batch, _ = make_batch(cid)
        s = round(L.lsdd_detector(profile, batch.drop(columns=[tc]),
                                  seed=int(bench.set_index("condition_id").at[cid, "batch_seed"]))["lsdd"], 6)
        det_diff = max(det_diff, abs(s - new_lsdd[cid]))
    print(f"\nDETERMINISM: rebuilt LSDD detector, 3 batches, largest difference {det_diff:.2e} "
          f"-> {'PASSED' if det_diff == 0 else 'FAILED'}")
    if det_diff != 0:
        raise SystemExit("STOP: LSDD is still not deterministic")

    sub = todo[todo["condition_id"].isin(new_lsdd)].copy()
    sub["lsdd_new"] = sub["condition_id"].map(new_lsdd)
    print(f"\nOld vs new LSDD on {len(sub)} conditions: Spearman "
          f"{spearmanr(sub['lsdd'], sub['lsdd_new'])[0]:.3f}, "
          f"median |change| {np.median(np.abs(sub['lsdd'] - sub['lsdd_new'])):.4f}, "
          f"median old {sub['lsdd'].median():.4f}")
    for col in ["harm_paired", "actual_accuracy_drop"]:
        print(f"  Spearman with {col}: old {spearmanr(sub['lsdd'], sub[col])[0]:.3f}  "
              f"new {spearmanr(sub['lsdd_new'], sub[col])[0]:.3f}")
    print(f"Time: {el / 60:.1f} min, {el / max(n, 1):.2f} s per condition "
          f"-> full 1,600 about {el / max(n, 1) * 1600 / 60:.0f} min")

    if a.verify_only:
        print("\nverify_only: nothing saved.")
        return
    out = bench.copy()
    out.insert(out.columns.get_loc("lsdd") + 1, "lsdd_unseeded_v121", out["lsdd"])
    out["lsdd"] = out["condition_id"].map(new_lsdd)
    assert out["lsdd"].notna().all() and len(out) == 1600
    out.to_csv(a.out_csv, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"\nSaved {len(out)} rows -> {a.out_csv}")


if __name__ == "__main__":
    main()
