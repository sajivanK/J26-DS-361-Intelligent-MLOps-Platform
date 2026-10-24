"""
Backend settings shared by all four components.
J26-DS-361 | Intelligent MLOps Platform

Paths come from environment variables so the same code runs on every member's machine.
Models and data stay OUTSIDE the repo (the repo is public).
"""
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]           # <repo>/webapp/backend/app/settings.py
CONTRACTS_DIR = REPO_ROOT / "shared" / "contracts"
default_data_dir = REPO_ROOT / "data" if (REPO_ROOT / "data").is_dir() else Path(r"C:\J26\data")
DATA_DIR = Path(os.environ.get("J26_DATA_DIR", default_data_dir))

# Datasets the platform can monitor. Every component receives the dataset id in upstream["context"].
DATASETS = {
    "uci": {"label": "UCI Adult (income, XGBoost)", "model_id": "adult-xgb-v1"},
    "ieee": {"label": "IEEE-CIS Fraud (LightGBM)", "model_id": "ieeecis-lgbm-v1"},
}

# C1 (Sajivan): one engine per dataset; a path may be an engine .pkl or a portable folder.
C1_ENGINES = {
    "uci": os.environ.get("C1_ENGINE_UCI", str(DATA_DIR / "uci" / "portable")),
    "ieee": os.environ.get("C1_ENGINE_IEEE", str(DATA_DIR / "ieee" / "c1_engine_ieeecis_v12.pkl")),
}

# Drift Lab (platform test bench): unseen labelled rows to build batches from, and the feature groups
# (high / low importance) used in the C1 benchmark. Loaded on first use, not at startup.
LAB_SOURCE = {
    "uci": os.environ.get("LAB_SOURCE_UCI", str(DATA_DIR / "uci" / "test.csv")),
    "ieee": os.environ.get("LAB_SOURCE_IEEE", str(DATA_DIR / "ieee" / "test.csv")),
}
LAB_FEATURE_INFO = {
    "uci": os.environ.get("LAB_FEATURES_UCI", str(DATA_DIR / "uci" / "feature_importance_adult_v11.json")),
    "ieee": os.environ.get("LAB_FEATURES_IEEE", str(DATA_DIR / "ieee" / "feature_importance_ieeecis_v12.json")),
}
TARGET_COL = "target"

# Latest pipeline run, kept so the Overview page can show it after a restart (outside the repo).
RUNS_DIR = Path(os.environ.get("J26_RUNS_DIR", str(DATA_DIR / "platform_runs")))

# C2 (Tharmika): the engine's data and model files default to shared/uci_adult in the repo (from the
# team Drive, git-ignored); another place can be set with C2_REFERENCE_CSV, C2_TEST_CSV, C2_MODEL_PKL
# and C2_PREPROCESSOR_PKL (read by c2/src/config.py). The C2 run history and audit log stay outside
# the repo, in the data folder.
C2_RUNS_FILE = Path(os.environ.get("C2_RUNS_FILE", str(DATA_DIR / "c2" / "runs.jsonl")))
C2_AUDIT_LOG = Path(os.environ.get("C2_AUDIT_LOG", str(DATA_DIR / "c2" / "diagnosis_log.jsonl")))

# Bind the integrated diagnosis to the same team artifacts used by monitoring/C3.
# Explicit owner-provided paths still take precedence; C3 verifies fitted-model identity.
for variable, candidate in {
    "C2_REFERENCE_CSV": DATA_DIR / "uci/reference.csv",
    "C2_TEST_CSV": DATA_DIR / "uci/test.csv",
    "C2_MODEL_PKL": DATA_DIR / "c2/base_model_adult.pkl",
}.items():
    if candidate.is_file():
        os.environ.setdefault(variable, str(candidate))

# C3 (Mathusan): model storage and per-model onboarding context.
C3_MODEL_DIR = Path(os.environ.get("C3_MODEL_DIR", str(DATA_DIR / "c3")))
from component3.src.integration_storage import upload_root, contexts_file
C3_UPLOAD_DIR = upload_root()

LABEL_COLUMNS = ("target",)        # labels never reach the monitoring components
MAX_UPLOAD_MB = 50


def c3_data_context(model_id):
    """Optional onboarding JSON keyed by exact model ID; requests may supply it directly."""
    path = contexts_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        value = data.get(model_id, {}) if isinstance(data, dict) else {}
        return dict(value) if isinstance(value, dict) else {}
    except (OSError, ValueError):
        # Optional C3 onboarding must not prevent C1/C2 batch ingestion.
        # C3 will reject explicitly when the required context is unavailable.
        return {}
FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]   # Vite dev server
