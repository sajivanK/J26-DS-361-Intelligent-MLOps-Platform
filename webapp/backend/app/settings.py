"""
Backend settings shared by all four components.
J26-DS-361 | Intelligent MLOps Platform

Paths come from environment variables so the same code runs on every member's machine.
Models and data stay OUTSIDE the repo (the repo is public).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]           # <repo>/webapp/backend/app/settings.py
CONTRACTS_DIR = REPO_ROOT / "shared" / "contracts"
DATA_DIR = Path(os.environ.get("J26_DATA_DIR", r"C:\J26\data"))

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

# C2 / C3 / C4: add your own paths here when your real code is plugged in, e.g.
# C2_MODEL_DIR = os.environ.get("C2_MODEL_DIR", str(DATA_DIR / "c2"))

LABEL_COLUMNS = ("target",)        # labels never reach the monitoring components
MAX_UPLOAD_MB = 50
FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]   # Vite dev server
