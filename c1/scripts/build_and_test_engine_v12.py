"""
Build the C1 engine from the v1.2 benchmark, save it, reload it, and test detect()
on FRESH drifted batches of every drift type.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Fresh = new random batches (seeds 900000+), magnitudes NOT in the benchmark grid
(0.15 / 0.55 / 0.95), batch sizes 1000 / 1500 / 2500 (1500 and 2500 never trained on).
Same 28 drift recipes as the benchmark (new kinds of drift are tested by
leave-one-feature-out and IEEE-CIS, not here).

USAGE (Colab, from the repo root)
  python c1/scripts/build_and_test_engine_v12.py --shared_dir shared \
     --ref .../reference.csv --test .../test.csv --model .../base_model_adult.pkl \
     --feature_json .../feature_importance_adult_v11.json \
     --bench_csv .../benchmark_uci_adult_v12.csv --out_dir .../c1_models_v12
"""
import argparse, json, os, sys, time, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root
from c1.src.engine import C1Engine, ENGINE_VERSION            # noqa: E402

MAGS = [0.15, 0.55, 0.95]
TEST_SIZES = [1000, 1500, 2500]
N_NODRIFT = 30
SEED0 = 900000
EXT = {"none": "none", "interaction": "none", "marginal_single": "single",
       "combined": "single", "marginal_multi": "multi"}
KEYS = ["evidence_version", "engine_version", "model_id", "reference_dataset_uri",
        "current_dataset_uri", "batch_size", "drift_detected", "predicted_harm", "harm_interval",
        "harm_metric", "harm_severity", "marginal_extent", "interaction_suspected",
        "affected_features", "detector_scores", "detectors_fired", "detectors_fired_list",
        "prediction_stats", "warnings", "runtime_sec"]


def check_output(o, features):
    assert list(o) == KEYS, f"keys differ: {list(o)}"
    assert o["engine_version"] == ENGINE_VERSION
    assert isinstance(o["drift_detected"], bool)
    lo, hi = o["harm_interval"]
    assert lo <= o["predicted_harm"] <= hi, "predicted_harm outside its interval"
    assert o["harm_metric"] == "accuracy_drop"
    assert o["harm_severity"] in ("none", "low", "moderate", "high", "critical")
    assert o["marginal_extent"] in ("none", "single", "multi")
    assert 0.0 <= o["interaction_suspected"] <= 1.0
    assert set(o["affected_features"]) <= set(features)
    if o["marginal_extent"] == "none":
        assert o["affected_features"] == [], "affected features listed with no marginal shift"
    assert len(o["detector_scores"]) == 9 and len(o["prediction_stats"]) == 6
    assert o["detectors_fired"] == len(o["detectors_fired_list"]) <= 6
    assert o["drift_detected"] == (o["marginal_extent"] != "none" or o["detectors_fired"] >= 3)
    json.dumps(o)  # must be JSON-serialisable


def main():
    ap = argparse.ArgumentParser()
    for k in ["shared_dir", "ref", "test", "model", "feature_json", "bench_csv", "out_dir"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--target_col", default="target")
    a = ap.parse_args()
    tc = a.target_col
    t_all = time.time()

    sys.path.insert(0, os.path.abspath(a.shared_dir))
    import benchmark_loop as bl
    from injection.inject_drift import inject_drift
    imp = json.load(open(a.feature_json))
    ft = imp["feature_types"]
    hi = imp["feature_groups"]["high_importance"]["features"]
    lo = imp["feature_groups"]["low_importance"]["features"]
    ref, test = pd.read_csv(a.ref), pd.read_csv(a.test)
    base = joblib.load(a.model)

    # ── build, save, reload ──────────────────────────────────────────────
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "c1_engine_adult_v12.pkl")
    t0 = time.time()
    C1Engine().train(a.bench_csv, a.ref, a.test, a.model, ft, target_col=tc).save(path)
    eng = C1Engine.load(path)  # everything below uses the reloaded file only
    print(f"Engine trained, saved and reloaded ({(time.time() - t0) / 60:.1f} min, "
          f"{os.path.getsize(path) / 1e6:.1f} MB): {path}")
    print(json.dumps(eng.summary(), indent=1))
    fixed = eng.p

    # ── fresh test batches ───────────────────────────────────────────────
    seen, templates = set(), []
    for c in bl.build_condition_list(hi, lo, ft, "uci_adult"):
        if c["drift_type"] == "none":
            continue
        key = (c["drift_type"], tuple(c["features"]),
               tuple(map(tuple, c.get("interaction_pairs") or [])))
        if key not in seen:
            seen.add(key)
            templates.append(c)
    jobs = [(c, m) for c in templates for m in MAGS] + [(None, 0.0)] * N_NODRIFT
    print(f"\nTesting detect() on {len(jobs)} fresh batches ({len(templates)} drift templates x "
          f"{len(MAGS)} magnitudes + {N_NODRIFT} no-drift), batch sizes {TEST_SIZES}")

    # warm-up call (first LSDD call builds its permutation test; not counted in runtime)
    tw = time.time()
    eng.detect(test.drop(columns=[tc]).sample(1000, random_state=1), seed=1)
    print(f"Warm-up call: {time.time() - tw:.1f} s (one-time, at service start)")

    rows, examples = [], {}
    for i, (c, mag) in enumerate(jobs):
        bs, seed = TEST_SIZES[i % len(TEST_SIZES)], SEED0 + i
        dt = "none" if c is None else c["drift_type"]
        res = inject_drift(reference_df=ref, model=base, drift_type=dt,
                           features=[] if c is None else c["features"], magnitude=mag,
                           feature_types={} if c is None else c.get("feature_types", {}),
                           interaction_pairs=None if c is None else c.get("interaction_pairs"),
                           batch_size=bs, target_col=tc, seed=seed, source_df=test, paired=True)
        gt = res["ground_truth"]
        o = eng.detect(res["drifted_df"].drop(columns=[tc]), seed=seed,
                       current_dataset_uri=f"fresh_test_{i}.csv")
        check_output(o, eng.features)
        true_fixed = (fixed - gt["drifted_accuracy"]) * 100
        true_paired = gt["actual_accuracy_drop"]
        pred, (l, h) = o["predicted_harm"] * 100, o["harm_interval"]
        inj = [] if dt in ("none", "interaction") else list(c["features"])
        aff = o["affected_features"]
        rows.append({
            "i": i, "drift_type": dt, "magnitude": mag, "batch_size": bs,
            "features": ",".join(inj), "true_drop_pp": true_fixed, "true_paired_pp": true_paired,
            "pred_pp": pred, "abs_err_fixed": abs(pred - true_fixed),
            "abs_err_paired": abs(pred - true_paired),
            "covered": l * 100 <= true_fixed <= h * 100, "width_pp": (h - l) * 100,
            "true_extent": EXT[dt], "pred_extent": o["marginal_extent"],
            "extent_ok": EXT[dt] == o["marginal_extent"],
            "p_interaction": o["interaction_suspected"],
            "true_interaction": int(dt in ("interaction", "combined")),
            "drift_detected": o["drift_detected"], "harm_severity": o["harm_severity"],
            "injected_found": np.nan if not inj else float(set(inj) <= set(aff)),
            "top1_correct": np.nan if not inj else float(bool(aff) and aff[0] in inj),
            "n_affected": len(aff), "detectors_fired": o["detectors_fired"],
            "runtime_s": o["runtime_sec"], "n_warnings": len(o["warnings"])})
        if dt not in examples and (mag == 0.55 or c is None):
            examples[dt] = o
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(jobs)} ({(time.time() - t_all) / 60:.1f} min)", flush=True)
    R = pd.DataFrame(rows)

    # ── report ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 250)
    print("\n" + "=" * 110 + "\nDETECT() ON FRESH DRIFT, BY TRUE DRIFT TYPE\n" + "=" * 110)
    agg = R.groupby("drift_type").agg(
        n=("i", "size"), true_drop_pp=("true_drop_pp", "mean"), MAE_fixed=("abs_err_fixed", "mean"),
        MAE_paired=("abs_err_paired", "mean"), coverage=("covered", "mean"), width_pp=("width_pp", "mean"),
        extent_acc=("extent_ok", "mean"), mean_p_interaction=("p_interaction", "mean"),
        drift_detected=("drift_detected", "mean"), injected_found=("injected_found", "mean"),
        top1_correct=("top1_correct", "mean"), n_affected=("n_affected", "mean"),
        detectors_fired=("detectors_fired", "mean"), runtime_s=("runtime_s", "mean"))
    print(agg.round(3).to_string())

    auc = roc_auc_score(R["true_interaction"], R["p_interaction"])
    drift = R["drift_type"] != "none"
    print(f"\nOverall: MAE fixed {R['abs_err_fixed'].mean():.3f} pp | coverage {R['covered'].mean():.3f} | "
          f"extent accuracy {R['extent_ok'].mean():.3f} | interaction AUC {auc:.3f} | "
          f"mean runtime {R['runtime_s'].mean():.2f} s (max {R['runtime_s'].max():.2f})")

    print("\nBy batch size (coverage, mean width pp):")
    print(R.groupby("batch_size")[["covered", "width_pp", "abs_err_fixed"]].mean().round(3).to_string())

    nd = R[~drift]
    print(f"\nNo-drift batches (n={len(nd)}): drift_detected false alarms {nd['drift_detected'].mean():.1%} | "
          f"mean affected features listed {nd['n_affected'].mean():.2f} | "
          f"severity counts {nd['harm_severity'].value_counts().to_dict()}")
    print(f"Drift batches (n={drift.sum()}): drift_detected rate {R.loc[drift, 'drift_detected'].mean():.1%}")
    print("\nDrift present but harm low (drift_detected = True, harm_severity = none), by type:")
    print(R[R["drift_detected"] & (R["harm_severity"] == "none")].groupby("drift_type").size().to_string())

    print("\n" + "=" * 110 + "\nEXAMPLE detect() OUTPUT, ONE PER TRUE DRIFT TYPE (magnitude 0.55)\n" + "=" * 110)
    for dt in ["marginal_single", "marginal_multi", "interaction", "combined", "none"]:
        if dt in examples:
            print(f"\n--- true drift type: {dt} ---")
            print(json.dumps(examples[dt], indent=1))

    R.to_csv(os.path.join(a.out_dir, "engine_test_adult_v12.csv"), index=False)
    summ = {"engine": eng.summary(), "n_batches": len(R), "magnitudes": MAGS, "batch_sizes": TEST_SIZES,
            "by_type": json.loads(agg.round(4).to_json(orient="index")),
            "overall": {"mae_fixed": float(R["abs_err_fixed"].mean()), "coverage": float(R["covered"].mean()),
                        "extent_accuracy": float(R["extent_ok"].mean()), "interaction_auc": float(auc),
                        "mean_runtime_s": float(R["runtime_s"].mean())},
            "by_batch_size": json.loads(R.groupby("batch_size")[["covered", "width_pp", "abs_err_fixed"]]
                                        .mean().round(4).to_json(orient="index")),
            "no_drift": {"false_alarm_rate": float(nd["drift_detected"].mean()),
                         "mean_affected_listed": float(nd["n_affected"].mean())},
            "drift_detected_rate_on_drift": float(R.loc[drift, "drift_detected"].mean()),
            "examples": examples}
    json.dump(summ, open(os.path.join(a.out_dir, "engine_test_adult_v12.json"), "w"), indent=2)
    print(f"\nSaved test results to {a.out_dir}   Total time {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
