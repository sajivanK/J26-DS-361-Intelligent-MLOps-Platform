"""
Gate test: load a saved C1 engine and run detect() on two fixed batches (one clean, one with
a single-feature drift). Run it in Colab and on the laptop with the same arguments, then
compare the two JSON files. Identical outputs mean the laptop environment is correct.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

USAGE (from the repo root)
  python c1/scripts/gate_test_engine.py --dataset uci --engine <path>/c1_engine_adult_v12.pkl \
      --ref <path>/reference.csv --test <path>/test.csv --feature_json <path>/feature_importance.json \
      --out gate_uci_laptop.json
  python c1/scripts/gate_test_engine.py --compare gate_uci_colab.json gate_uci_laptop.json
"""
import argparse, json, os, platform, sys, time, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "shared"))

SETTINGS = {"uci": {"bs": 1500, "model_id": "adult-xgb-v1"},
            "ieee": {"bs": 5000, "model_id": "ieeecis-lgbm-v1"}}
SKIP = {"runtime_sec", "evidence_version", "current_dataset_uri"}  # differ on every run


def compare(a_path, b_path, tol=1e-9):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    print(f"A: {A['env']}\nB: {B['env']}\n")
    ok = True
    for name in A["outputs"]:
        a, b = A["outputs"][name], B["outputs"][name]
        diffs = []
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            va, vb = a.get(k), b.get(k)
            if isinstance(va, dict) and isinstance(vb, dict):
                for kk in sorted(set(va) | set(vb)):
                    x, y = va.get(kk), vb.get(kk)
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))
                            and abs(x - y) <= tol) and x != y:
                        diffs.append(f"{k}.{kk}: {x} vs {y}")
            elif isinstance(va, float) and isinstance(vb, float):
                if abs(va - vb) > tol:
                    diffs.append(f"{k}: {va} vs {vb}")
            elif va != vb:
                diffs.append(f"{k}: {va} vs {vb}")
        print(f"[{name}] {'IDENTICAL' if not diffs else f'{len(diffs)} DIFFERENCES'}")
        for d in diffs:
            print("   ", d)
        ok &= not diffs
    print("\nGATE TEST", "PASSED" if ok else "FAILED")
    return ok


def run(a):
    import joblib, numpy as np, pandas as pd, sklearn
    from c1.src.engine import ENGINE_VERSION
    from c1.src.engine_io import load_engine
    from injection.inject_drift import inject_drift

    s = SETTINGS[a.dataset]
    t0 = time.time()
    eng = load_engine(a.engine)
    print(f"Engine loaded in {time.time() - t0:.1f} s: version {ENGINE_VERSION}, "
          f"{len(eng.features)} features, batch sizes {eng.sizes}")
    summ = eng.summary()
    print("N2 quantiles by extent:", summ.get("n2_quantile_by_extent"))

    ref, test = pd.read_csv(a.ref), pd.read_csv(a.test)
    tc = a.target_col
    imp = json.load(open(a.feature_json))
    ft = imp["feature_types"]
    top = imp["feature_groups"]["high_importance"]["features"][0]

    tw = time.time()
    eng.detect(test.drop(columns=[tc]).sample(1000 if a.dataset == "uci" else 2000, random_state=1), seed=1)
    print(f"Warm-up call: {time.time() - tw:.1f} s")

    outputs = {}
    clean = test.sample(s["bs"], random_state=7).drop(columns=[tc])
    outputs["clean"] = eng.detect(clean, model_id=s["model_id"], seed=7)
    res = inject_drift(reference_df=ref, model=eng.base_model, drift_type="marginal_single",
                       features=[top], magnitude=0.55, feature_types={top: ft[top]},
                       batch_size=s["bs"], target_col=tc, seed=11, source_df=test, paired=True)
    outputs[f"single_{top}"] = eng.detect(res["drifted_df"].drop(columns=[tc]),
                                          model_id=s["model_id"], seed=11)

    for name, o in outputs.items():
        print(f"\n[{name}] drift_detected={o['drift_detected']}  harm={o['predicted_harm'] * 100:.3f} pp  "
              f"interval=[{o['harm_interval'][0] * 100:.2f}, {o['harm_interval'][1] * 100:.2f}]  "
              f"extent={o['marginal_extent']}  severity={o['harm_severity']}  "
              f"fired={o['detectors_fired_list']}  features={o['affected_features']}  "
              f"runtime={o['runtime_sec']} s")

    env = {"python": platform.python_version(), "os": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "joblib": joblib.__version__}
    for mod in ["xgboost", "lightgbm", "torch", "alibi_detect", "scipy"]:
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            env[mod] = f"missing ({type(e).__name__})"
    json.dump({"dataset": a.dataset, "env": env, "summary": summ, "outputs": outputs},
              open(a.out, "w"), indent=2, default=str)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    ap.add_argument("--dataset", choices=list(SETTINGS))
    for k in ["engine", "ref", "test", "feature_json", "out"]:
        ap.add_argument(f"--{k}")
    ap.add_argument("--target_col", default="target")
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    missing = [k for k in ["dataset", "engine", "ref", "test", "feature_json", "out"] if not getattr(a, k)]
    if missing:
        ap.error(f"missing arguments: {missing}")
    run(a)