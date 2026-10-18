"""
IEEE-CIS benchmark with the v1.2 C1 signals, in one pass.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

SAME FRAMEWORK AS UCI ADULT v1.2
  Conditions:  shared benchmark_loop v1.1 build_condition_list (1,600 conditions,
               350 drift + 50 no-drift per batch size), same drift recipes.
  Batches:     batch_seed = crc32(condition_id), inject_drift(source_df=test, paired=True).
  Signals:     c1.src detectors (type-aware, per feature, mean + max), MMD, LSDD,
               classifier AUC, prediction statistics. Same column names as the
               UCI v1.2 CSV.
DIFFERENCES FOR IEEE-CIS (decided in Sprint 05 task 03)
  Batch sizes 2,000 / 4,000 / 6,000 / 10,000 (from the benchmark config).
  Harm metric is AUC drop (3% fraud: accuracy hardly moves when fraud detection fails):
    actual_auc_drop  = (AUC on full test - AUC on drifted batch) x 100   [primary]
    harm_auc_paired  = (AUC on same rows before injection - after) x 100 [sensitivity]
  Accuracy columns are kept for the accuracy-vs-AUC comparison.
  No v1.1 IEEE benchmark exists, so instead of a reproduction check the dry run
  checks determinism (re-running conditions gives identical numbers).

MODES
  --dry_run           40 conditions (each drift type x batch size twice + 2 no-drift
                      per size), timing per step, checks; saves dry_run_ieee_cis_v12.csv
  --part i --n_parts k  full run, part i of k (conditions i, i+k, i+2k, ...), so the
                      run can be split over parallel Colab sessions. Checkpointed.
  --merge             join the k part files into benchmark_ieee_cis_v12.csv

USAGE (Colab, from the c1-sajivan repo root)
  python c1/scripts/run_benchmark_ieee_v12.py --shared_dir /content/repo_main/shared \
     --ref .../reference.csv --test .../test.csv --model .../base_model_ieeecis.pkl \
     --feature_json .../c1_ieee/feature_importance_ieeecis_v12.json \
     --config .../c1_ieee/benchmark_config_ieeecis_v12.json --out_dir .../c1_ieee --dry_run
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root -> "c1" package

from c1.src.profile import ReferenceProfile                                  # noqa: E402
from c1.src.detectors import run_all_detectors, DETECTOR_SIGNALS             # noqa: E402
from c1.src.detectors.ks_detector import ks_detector                         # noqa: E402
from c1.src.detectors.chi2_detector import chi2_detector                     # noqa: E402
from c1.src.detectors.psi_detector import psi_detector                       # noqa: E402
from c1.src.detectors.mmd_detector import mmd_detector                       # noqa: E402
from c1.src.detectors.lsdd_detector import lsdd_detector                     # noqa: E402
from c1.src.detectors.clf_detector import clf_detector                       # noqa: E402
from c1.src.prediction_stats import (prediction_stats, CONFIDENCE_SIGNALS,   # noqa: E402
                                     POS_PROB_SIGNALS)

VERSION = "1.2"
DATASET = "ieee_cis"
DRIFT_TYPES = ["marginal_single", "marginal_multi", "interaction", "combined"]
STEPS = ["inject", "ks", "chi2", "psi", "mmd", "lsdd", "clf", "pred"]
TIME_COLS = [f"t_{s}" for s in STEPS] + ["condition_time_sec"]
TOL_4DP = 1.01e-4


def parse_args():
    a = argparse.ArgumentParser()
    for k in ["shared_dir", "ref", "test", "model", "feature_json", "config", "out_dir"]:
        a.add_argument(f"--{k}", required=True)
    a.add_argument("--target_col", default="target")
    a.add_argument("--dry_run", action="store_true")
    a.add_argument("--part", type=int, default=0)
    a.add_argument("--n_parts", type=int, default=1)
    a.add_argument("--merge", action="store_true")
    a.add_argument("--checkpoint_every", type=int, default=25)
    return a.parse_args()


# ── one condition ────────────────────────────────────────────────────────────

def detectors_timed(profile, X, seed, t):
    """Same calls, same order, same seeds as c1.src.detectors.run_all_detectors,
    with a timer around each detector (checked against it in the dry run)."""
    scores, per = {}, {}
    for name, fn in (("ks", lambda: ks_detector(profile, X)),
                     ("chi2", lambda: chi2_detector(profile, X, seed=seed)),
                     ("psi", lambda: psi_detector(profile, X))):
        t0 = time.time()
        s, p = fn()
        t[f"t_{name}"] = time.time() - t0
        scores.update(s)
        per[name] = p
    for name, fn in (("mmd", mmd_detector), ("lsdd", lsdd_detector), ("clf", clf_detector)):
        t0 = time.time()
        scores.update(fn(profile, X, seed=seed))
        t[f"t_{name}"] = time.time() - t0
    return scores, per


def make_batch(cond, ctx):
    seed = zlib.crc32(cond["condition_id"].encode()) % (2 ** 31)
    res = ctx["inject_drift"](
        reference_df=ctx["ref"], model=ctx["model"], drift_type=cond["drift_type"],
        features=cond["features"], magnitude=cond["magnitude"],
        feature_types=cond.get("feature_types", {}),
        interaction_pairs=cond.get("interaction_pairs"),
        batch_size=cond["batch_size"], target_col=ctx["tc"], seed=seed,
        source_df=ctx["test"], paired=True)
    return seed, res["drifted_df"], res["ground_truth"]


def run_condition(cond, ctx):
    tc, t, t_all = ctx["tc"], {}, time.time()
    t0 = time.time()
    seed, batch, gt = make_batch(cond, ctx)
    t["t_inject"] = time.time() - t0
    X, y = batch.drop(columns=[tc]), batch[tc].values

    det, per = detectors_timed(ctx["profile"], X, seed, t)

    t0 = time.time()
    ps = prediction_stats(ctx["model"], X)
    auc_d = float(roc_auc_score(y, ctx["model"].predict_proba(X)[:, 1]))
    t["t_pred"] = time.time() - t0
    if abs(round(auc_d, 4) - gt["drifted_auc"]) > TOL_4DP:
        raise RuntimeError(f"{cond['condition_id']}: AUC {auc_d:.4f} != injection AUC "
                           f"{gt['drifted_auc']} (batch rows differ)")

    acc_drop = (ctx["acc_fixed"] - gt["drifted_accuracy"]) * 100
    row = {
        "condition_id": cond["condition_id"], "dataset": DATASET,
        "drift_type": cond["drift_type"], "magnitude": cond["magnitude"],
        "feature_group": cond.get("feature_group", "none"),
        "injected_features": str(gt["injected_features"]),
        "seed": cond["seed"], "batch_seed": seed, "batch_size": len(batch),
        **{k: round(det[k], 6) for k in DETECTOR_SIGNALS},
        **{k: round(ps[k], 6) for k in POS_PROB_SIGNALS + CONFIDENCE_SIGNALS},
        # PRIMARY target for IEEE-CIS (pp)
        "harm_metric": "auc_drop",
        "actual_auc_drop": round((ctx["auc_fixed"] - auc_d) * 100, 4),
        "fixed_baseline_auc": round(ctx["auc_fixed"], 4),
        # SENSITIVITY target: same rows before vs after (baseline AUC is stored to 4 dp)
        "harm_auc_paired": round((gt["baseline_auc"] - auc_d) * 100, 4),
        "baseline_auc_paired": gt["baseline_auc"],
        "drifted_auc": round(auc_d, 6),
        # accuracy columns, same definitions as UCI v1.2 (for the accuracy-vs-AUC comparison)
        "actual_accuracy_drop": round(acc_drop, 4),
        "fixed_baseline_accuracy": round(ctx["acc_fixed"], 4),
        "harm_paired": round(gt["actual_accuracy_drop"], 4),
        "paired_baseline_accuracy": round(gt["baseline_accuracy"], 4),
        "drifted_accuracy": round(gt["drifted_accuracy"], 4),
        "is_malignant": bool(acc_drop > 2.0),
        "is_malignant_paired": bool(gt["actual_accuracy_drop"] > 2.0),
    }
    for name, scores in per.items():
        row.update({f"{name}__{f}": round(v, 6) for f, v in scores.items()})
    row["benchmark_version"] = VERSION
    row.update({k: round(v, 3) for k, v in t.items()})
    row["condition_time_sec"] = round(time.time() - t_all, 2)
    return row, (X, seed, det)


# ── condition selection ──────────────────────────────────────────────────────

def pick_dry_run(conds, sizes):
    """Every drift type x batch size twice (one high-importance, one low-importance
    condition; magnitudes alternate between the largest and smallest available so
    both ends are tested), plus 2 no-drift conditions per batch size = 40."""
    out = []
    for dt in DRIFT_TYPES:
        for j, bs in enumerate(sizes):
            for grp in ("high_importance", "low_importance"):
                g = [c for c in conds if c["drift_type"] == dt and c["batch_size"] == bs
                     and c["feature_group"] == grp]
                mags = sorted({c["magnitude"] for c in g})
                big = (j % 2 == 0) == (grp == "high_importance")
                want = mags[-1] if big else mags[0]
                out.append(next(c for c in g if c["magnitude"] == want))
    for bs in sizes:
        out += [c for c in conds if c["drift_type"] == "none" and c["batch_size"] == bs
                and c["seed"] in (0, 1)]
    assert len(out) == 40 and len({c["condition_id"] for c in out}) == 40
    return out


# ── reports ──────────────────────────────────────────────────────────────────

def dry_run_report(R, ctx, cfg, sizes):
    pd.set_option("display.width", 250)
    noise = {int(k): v["auc_sd_pp"] for k, v in cfg["clean_batch_auc_noise"].items()}
    ok = True

    print("\n" + "=" * 100 + "\nCHECK 1  columns and missing values\n" + "=" * 100)
    expected = (["condition_id", "dataset", "drift_type", "magnitude", "feature_group",
                 "injected_features", "seed", "batch_seed", "batch_size"]
                + DETECTOR_SIGNALS + POS_PROB_SIGNALS + CONFIDENCE_SIGNALS
                + ["actual_auc_drop", "harm_auc_paired", "actual_accuracy_drop", "harm_paired"])
    miss = [c for c in expected if c not in R.columns]
    per_cols = [c for c in R.columns if "__" in c]
    n_ks = sum(c.startswith("ks__") for c in per_cols)
    n_chi = sum(c.startswith("chi2__") for c in per_cols)
    n_psi = sum(c.startswith("psi__") for c in per_cols)
    nan_cols = R.columns[R.isna().any()].tolist()
    print(f"  {len(R.columns)} columns | per-feature: ks {n_ks} (continuous), "
          f"chi2 {n_chi} (categorical), psi {n_psi} (all)")
    print(f"  missing expected columns: {miss or 'none'} | columns with NaN: {nan_cols or 'none'}")
    ok &= not miss and not nan_cols and n_ks == len(ctx["profile"].continuous) \
        and n_chi == len(ctx["profile"].categorical) and n_psi == len(ctx["profile"].features)

    print("\n" + "=" * 100 + "\nCHECK 2  clean batches: AUC drop should sit inside the noise\n" + "=" * 100)
    nd = R[R["drift_type"] == "none"].copy()
    nd["noise_sd_pp"] = nd["batch_size"].map(noise)
    nd["z"] = nd["actual_auc_drop"] / nd["noise_sd_pp"]
    print(nd[["condition_id", "batch_size", "actual_auc_drop", "harm_auc_paired",
              "actual_accuracy_drop", "noise_sd_pp", "z"]].round(3).to_string(index=False))
    # paired baseline AUC comes from inject_drift rounded to 4 dp, so up to 0.005 pp of
    # rounding is expected; anything above 0.01 pp would mean the rows differ
    print(f"  paired harm on clean batches must be 0 up to rounding (< 0.01 pp): "
          f"max |harm_auc_paired| = {nd['harm_auc_paired'].abs().max():.4f}")
    ok &= bool((nd["z"].abs() < 3).all()) and nd["harm_auc_paired"].abs().max() < 0.01

    print("\n" + "=" * 100 + "\nCHECK 3  drift: harm by group and magnitude (AUC vs accuracy)\n" + "=" * 100)
    D = R[R["drift_type"] != "none"].copy()
    D["mag_end"] = np.where(D["magnitude"] >= 0.5, "largest", "smallest")
    print(D.groupby(["feature_group", "drift_type", "mag_end"])[
        ["magnitude", "actual_auc_drop", "harm_auc_paired", "actual_accuracy_drop", "ks_max",
         "chi2_max", "mmd", "clf_auc"]].mean().round(3).to_string())
    hi = D[D["feature_group"] == "high_importance"]
    rho = spearmanr(hi["magnitude"], hi["harm_auc_paired"]).correlation
    big = hi.loc[hi["mag_end"] == "largest", "harm_auc_paired"].mean()
    small = hi.loc[hi["mag_end"] == "smallest", "harm_auc_paired"].mean()
    print(f"\n  high-importance group: mean paired AUC harm at largest magnitude {big:.2f} pp vs "
          f"smallest {small:.2f} pp | Spearman(magnitude, harm) = {rho:.2f}")
    print(f"  low-importance group: mean paired AUC harm "
          f"{D.loc[D['feature_group'] == 'low_importance', 'harm_auc_paired'].mean():.2f} pp "
          "(expected near 0: the model barely uses these features)")
    ok &= big > small

    print("\n" + "=" * 100 + "\nRUNTIME  seconds per condition by batch size and step\n" + "=" * 100)
    tcols = [f"t_{s}" for s in STEPS] + ["condition_time_sec"]
    T = R.iloc[1:]  # first condition includes one-time LSDD set-up
    print(T.groupby("batch_size")[tcols].median().round(2).to_string())
    print(f"  (first condition took {R['condition_time_sec'].iloc[0]:.1f} s including one-time set-up)")
    per_size = T.groupby("batch_size")["condition_time_sec"].median()
    total_h = float(sum(per_size.get(bs, np.nan) * 400 for bs in sizes)) / 3600
    print(f"\n  projected full run (400 conditions per batch size): {total_h:.1f} h in one session"
          f" | {total_h / 2:.1f} h each over 2 parallel sessions | {total_h / 3:.1f} h each over 3")
    return ok


def determinism_check(R, picks, ctx):
    """Re-run 3 conditions from scratch; every number must match (clf reported only)."""
    print("\n" + "=" * 100 + "\nCHECK 4  determinism: re-run 3 conditions\n" + "=" * 100)
    chosen = [next(c for c in picks if c["drift_type"] == "marginal_single"),
              next(c for c in picks if c["drift_type"] == "interaction"),
              next(c for c in picks if c["drift_type"] == "none")]
    first = R.set_index("condition_id")
    worst, clf_worst = 0.0, 0.0
    for c in chosen:
        row, _ = run_condition(c, ctx)
        for k, v in row.items():
            if k in TIME_COLS or not isinstance(v, (int, float, np.floating)) or isinstance(v, bool):
                continue
            d = abs(float(v) - float(first.loc[c["condition_id"], k]))
            if k == "clf_auc":
                clf_worst = max(clf_worst, d)
            else:
                worst = max(worst, d)
    print(f"  largest difference over all numeric columns (except clf_auc and times): {worst:.2e}")
    print(f"  largest clf_auc difference: {clf_worst:.2e} (xgboost threading can cause tiny differences)")
    return worst == 0.0


def faithful_check(R, cond, ctx):
    """The timed detector calls must give exactly what run_all_detectors gives."""
    _, batch, _ = make_batch(cond, ctx)
    X = batch.drop(columns=[ctx["tc"]])
    seed = zlib.crc32(cond["condition_id"].encode()) % (2 ** 31)
    ref_scores, _ = run_all_detectors(ctx["profile"], X, seed=seed)
    got = R.set_index("condition_id").loc[cond["condition_id"]]
    d = max(abs(round(ref_scores[k], 6) - got[k]) for k in DETECTOR_SIGNALS if k != "clf_auc")
    print(f"  timed detector calls vs run_all_detectors, largest difference: {d:.2e}")
    return d == 0.0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    a = parse_args()
    tc = a.target_col
    final = os.path.join(a.out_dir, "benchmark_ieee_cis_v12.csv")

    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    assert bl.VERSION == "1.1", f"shared benchmark_loop is v{bl.VERSION}, expected 1.1"
    from injection.inject_drift import inject_drift

    cfg = json.load(open(a.config))
    imp = json.load(open(a.feature_json))
    sizes = cfg["batch_sizes"]
    hi = imp["feature_groups"]["high_importance"]["features"]
    lo = imp["feature_groups"]["low_importance"]["features"]
    assert hi == cfg["high_importance"] and lo == cfg["low_importance"], "groups differ from config"
    conds = bl.build_condition_list(hi, lo, imp["feature_types"], DATASET, batch_sizes=sizes)
    order = [c["condition_id"] for c in conds]

    if a.merge:
        parts = sorted(glob.glob(os.path.join(a.out_dir, "benchmark_ieee_cis_v12_part*of*.csv")))
        parts = [p for p in parts if "checkpoint" not in p]
        M = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        print(f"merging {len(parts)} part files: {len(M)} rows")
        if not M["condition_id"].is_unique or set(M["condition_id"]) != set(order):
            raise SystemExit(f"STOP: merged rows do not equal the 1,600 conditions "
                             f"(missing {len(set(order) - set(M['condition_id']))}, "
                             f"duplicates {M['condition_id'].duplicated().sum()})")
        M = M.set_index("condition_id").loc[order].reset_index()
        M.to_csv(final, index=False)
        print(f"Saved {len(M)} rows -> {final}")
        print(M["drift_type"].value_counts().to_string())
        print(M.groupby("batch_size")["condition_time_sec"].median().round(2).to_string())
        return

    ref, test = pd.read_csv(a.ref), pd.read_csv(a.test)
    model = joblib.load(a.model)
    X_test = test.drop(columns=[tc])
    auc_fixed = float(roc_auc_score(test[tc], model.predict_proba(X_test)[:, 1]))
    acc_fixed = float(accuracy_score(test[tc], model.predict(X_test)))
    if abs(round(auc_fixed, 4) - cfg["fixed_baseline_auc"]) > TOL_4DP or \
            abs(round(acc_fixed, 4) - cfg["fixed_baseline_accuracy"]) > TOL_4DP:
        raise SystemExit(f"STOP: baselines {auc_fixed:.4f}/{acc_fixed:.4f} differ from config "
                         f"(wrong model or test file?)")
    assert max(sizes) <= len(test)
    print(f"Fixed baselines on full test: AUC {auc_fixed:.4f} | accuracy {acc_fixed:.4f}")

    t0 = time.time()
    profile = ReferenceProfile(ref, imp["feature_types"], target_col=tc)
    print(f"Profile: {len(profile.continuous)} continuous, {len(profile.categorical)} categorical, "
          f"kernel input {profile.ref_kernel.shape}, MMD bandwidth {profile.mmd_sigma:.4f} "
          f"({time.time() - t0:.1f} s)")
    ctx = dict(ref=ref, test=test, model=model, tc=tc, profile=profile,
               auc_fixed=auc_fixed, acc_fixed=acc_fixed, inject_drift=inject_drift)

    if a.dry_run:
        todo = pick_dry_run(conds, sizes)
        out_csv, ckpt = os.path.join(a.out_dir, "dry_run_ieee_cis_v12.csv"), None
    else:
        assert 0 <= a.part < a.n_parts
        todo = conds[a.part::a.n_parts]
        out_csv = os.path.join(a.out_dir, f"benchmark_ieee_cis_v12_part{a.part}of{a.n_parts}.csv")
        ckpt = out_csv.replace(".csv", "_checkpoint.csv")

    rows, done = [], set()
    if ckpt and os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        rows, done = prev.to_dict("records"), set(prev["condition_id"])
        print(f"Resuming from checkpoint: {len(done)} done")
    remaining = [c for c in todo if c["condition_id"] not in done]
    print(f"Conditions to run: {len(remaining)} of {len(todo)}\n")

    t_start, errors = time.time(), []
    for i, c in enumerate(remaining):
        try:
            row, _ = run_condition(c, ctx)
        except ImportError:
            if ckpt:
                pd.DataFrame(rows).to_csv(ckpt, index=False)
            raise
        except Exception as e:
            errors.append((c["condition_id"], repr(e)))
            print(f"  ERROR {c['condition_id']}: {e!r}")
            continue
        rows.append(row)
        if i == 0 or (i + 1) % 10 == 0 or a.dry_run:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(remaining) - i - 1) / 60
            print(f"  [{i + 1:4d}/{len(remaining)}] {c['condition_id'][:44]:44s} bs={row['batch_size']:>5} "
                  f"auc_drop={row['actual_auc_drop']:+6.2f} paired={row['harm_auc_paired']:+6.2f} "
                  f"acc_drop={row['actual_accuracy_drop']:+5.2f} t={row['condition_time_sec']:5.1f}s "
                  f"ETA {eta:.0f}m", flush=True)
        if ckpt and len(rows) % a.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)

    R = pd.DataFrame(rows)
    R.to_csv(out_csv, index=False)
    print(f"\nSaved {len(R)} rows -> {out_csv} | errors: {len(errors)} | "
          f"{(time.time() - t_start) / 60:.1f} min")
    for e in errors:
        print("  ", e)

    if a.dry_run:
        ok = not errors
        ok &= dry_run_report(R, ctx, cfg, sizes)
        print("\n" + "=" * 100 + "\nCHECK 5  timed detectors equal run_all_detectors\n" + "=" * 100)
        ok &= faithful_check(R, todo[0], ctx)
        ok &= determinism_check(R, todo, ctx)
        print("\n" + "=" * 100)
        print("DRY RUN RESULT: " + ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED, read the checks above"))
        print("=" * 100)
    elif ckpt and os.path.exists(ckpt) and not errors and len(R) == len(todo):
        os.remove(ckpt)


if __name__ == "__main__":
    main()
