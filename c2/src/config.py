"""
C2 configuration: paths and shared artifacts.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Every other C2 module imports from here, so paths are defined in one place.
Feature types are always read from feature_importance_adult.json (never hardcoded),
as required by the C2 research guide.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# ── PATHS ──────────────────────────────────────────────────────────────────
C2_DIR = Path(__file__).resolve().parents[1]          # .../c2
REPO_ROOT = C2_DIR.parent                             # repo root
SHARED_DIR = REPO_ROOT / "shared"

UCI_DIR = SHARED_DIR / "uci_adult"
REFERENCE_CSV = UCI_DIR / "data" / "reference.csv"
TEST_CSV = UCI_DIR / "data" / "test.csv"
MODEL_PKL = UCI_DIR / "models" / "base_model_adult.pkl"
FEATURE_JSON = UCI_DIR / "feature_importance_adult.json"
CONTRACTS_DIR = SHARED_DIR / "contracts"

RESULTS_DIR = C2_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
BASELINE_ACCURACY = 0.8744   # model accuracy on test.csv (shared v1.1)

# shared/ must be on the path so `from injection.inject_drift import inject_drift` works
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))


# ── LOADERS ────────────────────────────────────────────────────────────────
def _check_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(REPO_ROOT)}. Download it from the team Drive "
            f"folder J26-DS-361-Datasets (it is not in GitHub because of .gitignore)."
        )


def load_feature_info():
    """Feature types and importance groups, straight from the shared JSON."""
    with open(FEATURE_JSON) as f:
        info = json.load(f)
    return {
        "feature_types": info["feature_types"],
        "high_importance": info["feature_groups"]["high_importance"]["features"],
        "low_importance": info["feature_groups"]["low_importance"]["features"],
        "shap_importance": info["shap_importance_ranked"],
    }


def load_artifacts():
    """Load reference data, test data, the base model and feature info."""
    for p in (REFERENCE_CSV, TEST_CSV, MODEL_PKL, FEATURE_JSON):
        _check_exists(p)
    ref_df = pd.read_csv(REFERENCE_CSV)
    test_df = pd.read_csv(TEST_CSV)
    model = joblib.load(MODEL_PKL)
    info = load_feature_info()
    feature_names = [c for c in ref_df.columns if c != TARGET_COL]
    return {
        "ref_df": ref_df,
        "test_df": test_df,
        "model": model,
        "feature_names": feature_names,
        **info,
    }


if __name__ == "__main__":
    # Quick self-check: python3.11 c2/src/config.py
    from sklearn.metrics import accuracy_score

    a = load_artifacts()
    X_test = a["test_df"].drop(columns=[TARGET_COL])
    acc = accuracy_score(a["test_df"][TARGET_COL], a["model"].predict(X_test))
    print(f"reference.csv : {a['ref_df'].shape}")
    print(f"test.csv      : {a['test_df'].shape}")
    print(f"test accuracy : {acc:.4f}  (expected {BASELINE_ACCURACY})")
    print(f"high importance: {a['high_importance']}")
    print(f"low importance : {a['low_importance']}")
    print(f"native_country type: {a['feature_types']['native_country']}")
    from injection.inject_drift import inject_drift  # noqa: F401
    print("inject_drift import: OK")
    print("CONFIG OK" if abs(acc - BASELINE_ACCURACY) < 0.001 else "WARNING: accuracy differs")