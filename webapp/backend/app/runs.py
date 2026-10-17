"""
The latest full pipeline run, shown on the Overview page. Kept in memory and in one JSON file outside the
repo (settings.RUNS_DIR), so it survives a backend restart. Saving is best-effort: a failure never
breaks a run.
"""
import json
import threading

from .settings import RUNS_DIR

_FILE = RUNS_DIR / "latest_pipeline_run.json"
_lock = threading.Lock()
_latest = None


def save_latest(run):
    global _latest
    with _lock:
        _latest = run
        try:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            _FILE.write_text(json.dumps(run, default=str), encoding="utf-8")
        except OSError as e:
            print(f"[runs] could not save latest run: {e}", flush=True)


def latest():
    global _latest
    with _lock:
        if _latest is None and _FILE.exists():
            try:
                _latest = json.loads(_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _latest = None
        return _latest
