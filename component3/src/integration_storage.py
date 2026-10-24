"""C3 storage defaults; no model or batch artifacts are written into the repo."""
import os
from pathlib import Path


def data_root():
    repo_data = Path(__file__).resolve().parents[2] / "data"
    default_dir = repo_data if repo_data.is_dir() else Path(r"C:\J26\data")
    return Path(os.environ.get("J26_DATA_DIR", default_dir))


def upload_root():
    return Path(os.environ.get("C3_UPLOAD_DIR", str(data_root() / "platform_runs/c3_batches")))


def run_root():
    return Path(os.environ.get("C3_RUNS_DIR", str(data_root() / "platform_runs/c3_repairs")))


def contexts_file():
    return Path(os.environ.get("C3_CONTEXTS_FILE", str(data_root() / "c3/c3_contexts.json")))
