"""
Re-run the UCI Adult fresh test (114 batches) on an EXISTING engine, without retraining, and compare
every batch with the original Colab results (engine_test_adult_v12.csv from c1_models_v123).
Purpose: show that a moved or re-saved engine (e.g. the portable folder with the XGBoost base model
as JSON, on Windows) reproduces the published engine 1.2.3 fresh-test results batch by batch.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Same batches as build_and_test_engine_v12.py: same drift templates, magnitudes, batch sizes and seeds
(constants and the output checks are imported from that script). The base model used to compute the
true harm is the engine's own base model.

USAGE (from the repo root)
  python c1/scripts/retest_engine_fresh.py --engine C:/J26/data/uci/portable \
      --ref .../reference.csv --test .../test.csv --feature_json .../feature_importance_adult_v11.json \
      --colab_csv .../engine_test_adult_v12.csv --out_csv .../engine_retest_adult_laptop.csv
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from sklearn.metrics import roc_auc_score                           # noqa: E402
from c1.src.engine_io import load_engine                            # noqa: E402
from build_and_test_engine_v12 import (MAGS, TEST_SIZES, N_NODRIFT, SEED0, EXT,  # noqa: E402
                                       check_output)

NUM_COLS = ["true_drop_pp", "true_paired_pp", "pred_pp", "width_pp", "p_interaction"]
EXACT_COLS = ["drift_type", "magnitude", "batch_size", "covered", "pred_extent", "drift_detected",
              "harm_severity", "n_affected", "detectors_fired"]


def main():
    ap = argparse.ArgumentParser()
    for k in ["engine", "ref", "test", "feature_json", "colab_csv", "out_csv"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--shared_dir", default=os.path.join(ROOT, "shared"))
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

    eng = load_engine(a.engine)
    base, fixed = eng.base_model, eng.p
    print(f"Engine loaded: {a.engine} | {len(eng.features)} features | fixed baseline accuracy {fixed:.4f}")

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
    print(f"Re-testing {len(jobs)} fresh batches ({len(templates)} templates x {len(MAGS)} magnitudes "
          f"+ {N_NODRIFT} no-drift), batch sizes {TEST_SIZES}")

    tw = time.time()
    eng.detect(test.drop(columns=[tc]).sample(1000, random_state=1), seed=1)
    print(f"Warm-up call: {time.time() - tw:.1f} s")

    rows = []
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
        pred, (l, h) = o["predicted_harm"] * 100, o["harm_interval"]
        rows.append({"i": i, "drift_type": dt, "magnitude": mag, "batch_size": bs,
                     "true_drop_pp": true_fixed, "true_paired_pp": gt["actual_accuracy_drop"],
                     "pred_pp": pred, "abs_err_fixed": abs(pred - true_fixed),
                     "covered": l * 100 <= true_fixed <= h * 100, "width_pp": (h - l) * 100,
                     "pred_extent": o["marginal_extent"], "extent_ok": EXT[dt] == o["marginal_extent"],
                     "p_interaction": o["interaction_suspected"],
                     "true_interaction": int(dt in ("interaction", "combined")),
                     "drift_detected": o["drift_detected"], "harm_severity": o["harm_severity"],
                     "n_affected": len(o["affected_features"]), "detectors_fired": o["detectors_fired"],
                     "runtime_s": o["runtime_sec"]})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(jobs)} ({(time.time() - t_all) / 60:.1f} min)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(a.out_csv, index=False)

    auc = roc_auc_score(R["true_interaction"], R["p_interaction"])
    nd = R[R["drift_type"] == "none"]
    print(f"\nThis run: MAE fixed {R['abs_err_fixed'].mean():.3f} pp | coverage {R['covered'].mean():.3f} | "
          f"extent accuracy {R['extent_ok'].mean():.3f} | interaction AUC {auc:.3f} | "
          f"false alarms {nd['drift_detected'].mean():.1%} | mean runtime {R['runtime_s'].mean():.2f} s")

    # ── batch-by-batch comparison with the Colab run ────────────────────
    C = pd.read_csv(a.colab_csv)
    assert len(C) == len(R), f"Colab CSV has {len(C)} rows, this run {len(R)}"
    bad = []
    for col in EXACT_COLS:
        m = (C[col].astype(str).values != R[col].astype(str).values)
        if m.any():
            bad.append(f"{col}: {int(m.sum())} batches differ (first at row {int(np.argmax(m))})")
    for col in NUM_COLS:
        d = np.abs(C[col].values.astype(float) - R[col].values.astype(float))
        if np.nanmax(d) > 1e-9:
            bad.append(f"{col}: max abs diff {np.nanmax(d):.3e} ({int((d > 1e-9).sum())} batches)")
    print(f"Colab run: MAE fixed {C['abs_err_fixed'].mean():.3f} pp | coverage {C['covered'].mean():.3f} | "
          f"extent accuracy {(C['true_extent'] == C['pred_extent']).mean():.3f}")
    print(f"\nCompared {len(R)} batches x {len(EXACT_COLS) + len(NUM_COLS)} fields with the Colab run:")
    for b in bad:
        print("  ", b)
    print("RETEST", "IDENTICAL TO COLAB" if not bad else "DIFFERS FROM COLAB",
          f"| total time {(time.time() - t_all) / 60:.1f} min | saved {a.out_csv}")


if __name__ == "__main__":
    main()