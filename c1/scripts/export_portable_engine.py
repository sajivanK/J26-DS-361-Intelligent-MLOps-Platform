"""
Export a C1 engine .pkl to a portable folder (XGBoost base model as JSON), then reload it and check
that the reloaded base model gives the same predictions as the original on the test set.
Run where the original .pkl loads (Colab / Linux).
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

USAGE (from the repo root)
  python c1/scripts/export_portable_engine.py --engine .../c1_models_v123/c1_engine_adult_v12.pkl \
      --out_dir .../c1_models_v123/portable --test .../uci_adult_data/test.csv
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root
from c1.src.engine import C1Engine          # noqa: E402
from c1.src.engine_io import export_portable, load_engine  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target_col", default="target")
    a = ap.parse_args()

    import numpy as np, pandas as pd
    info = export_portable(a.engine, a.out_dir)
    print("Wrote", a.out_dir)
    for k, v in info.items():
        print(f"  {k}: {v}")

    orig = C1Engine.load(a.engine)
    port = load_engine(a.out_dir)
    X = pd.read_csv(a.test).drop(columns=[a.target_col])[orig.features]
    p0, p1 = orig.base_model.predict_proba(X), port.base_model.predict_proba(X)
    l0, l1 = orig.base_model.predict(X), port.base_model.predict(X)
    diff = float(np.abs(p0 - p1).max())
    print(f"\nTest rows: {len(X)} | max |proba diff|: {diff:.3e} | label mismatches: {int((l0 != l1).sum())}")
    print("PORTABLE EXPORT", "OK" if diff == 0.0 and (l0 == l1).all() else "CHECK: predictions differ")