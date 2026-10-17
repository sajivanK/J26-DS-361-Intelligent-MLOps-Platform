"""
Portable save/load for the C1 engine (no change to how the engine detects).
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

WHY
  The engine .pkl is a joblib pickle. Everything in it unpickles on Windows and Linux except the
  XGBoost base model: an XGBoost pickle made on Linux (Colab) fails on Windows with
  "input stream corrupted", even with the same xgboost version (3.4.1). XGBoost's own JSON format
  is the documented portable format, so a portable engine stores the base model as JSON and the
  rest of the engine as a pickle without it.

LAYOUT (one folder per engine)
  <out_dir>/c1_engine_<name>_core.pkl   engine with base_model removed
  <out_dir>/base_model_xgb.json         XGBoost base model (only when the base model is XGBoost)
  <out_dir>/portable_manifest.json      file list + sha256 + source engine sha256

  LightGBM base models (IEEE-CIS) unpickle fine on both systems; they stay inside the core pickle.

USE
  from c1.src.engine_io import load_engine
  eng = load_engine("path/to/engine.pkl")       # a normal engine file
  eng = load_engine("path/to/portable_folder")  # a portable folder
"""
import hashlib
import json
import os

import joblib

MANIFEST = "portable_manifest.json"
XGB_FILE = "base_model_xgb.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_portable(engine_pkl, out_dir):
    """Load a normal engine .pkl and write the portable folder. Returns the manifest dict."""
    from c1.src.engine import C1Engine, ENGINE_VERSION

    eng = C1Engine.load(engine_pkl)
    os.makedirs(out_dir, exist_ok=True)
    model = eng.base_model
    info = {"engine_version": ENGINE_VERSION,
            "source_pkl": os.path.basename(engine_pkl),
            "source_sha256": _sha256(engine_pkl),
            "base_model_class": f"{type(model).__module__}.{type(model).__name__}"}

    files = []
    if type(model).__module__.split(".")[0] == "xgboost":
        model.save_model(os.path.join(out_dir, XGB_FILE))  # sklearn wrapper: keeps classes etc.
        info["base_model_file"] = XGB_FILE
        files.append(XGB_FILE)
        eng.base_model = None
    core = os.path.basename(engine_pkl).replace(".pkl", "_core.pkl")
    try:
        joblib.dump(eng, os.path.join(out_dir, core), compress=3)
    finally:
        eng.base_model = model
    info["core_file"] = core
    files.insert(0, core)
    info["sha256"] = {f: _sha256(os.path.join(out_dir, f)) for f in files}
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(info, f, indent=2)
    return info


def load_engine(path, verify=True):
    """Load an engine from a normal .pkl or from a portable folder."""
    from c1.src.engine import C1Engine

    if not os.path.isdir(path):
        return C1Engine.load(path)
    with open(os.path.join(path, MANIFEST)) as f:
        info = json.load(f)
    if verify:
        for name, digest in info["sha256"].items():
            got = _sha256(os.path.join(path, name))
            if got != digest:
                raise ValueError(f"{name}: sha256 {got[:12]} does not match manifest {digest[:12]}")
    eng = C1Engine.load(os.path.join(path, info["core_file"]))
    if info.get("base_model_file"):
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(os.path.join(path, info["base_model_file"]))
        eng.base_model = model
    return eng