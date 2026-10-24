"""Persistent C4 recorded-assessment history; no live repair/deployment execution."""
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from ..components import c4_worker_client as worker

router = APIRouter(prefix="/api/c4/workspace", tags=["C4 assessment history"])
log = logging.getLogger(__name__)


def database_path():
    configured = os.getenv("C4_HISTORY_DB")
    if configured:
        return Path(configured).expanduser()
    # Default outside the Git checkout, alongside the existing shared data.
    repo = Path(__file__).resolve().parents[4]
    data = Path(os.getenv("J26_DATA_DIR", str(repo.parent / "data")))
    return data / "c4_ui" / "assessment_history.sqlite3"


def connect():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS assessments (
        id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
        case_id TEXT NOT NULL, decision TEXT NOT NULL,
        payload TEXT NOT NULL, sha256 TEXT NOT NULL)""")
    return connection


def validate(payload):
    result = payload.get("result", {})
    if (payload.get("mode") != "verified_revised_recorded_replay"
            or result.get("method") != "hybrid"
            or result.get("family") != "adaptive"
            or result.get("decision") not in {"HUMAN_REVIEW", "ESCALATE", "AUTO_DEPLOY"}
            or result.get("production_release_authorized") is not False
            or result.get("future_targets_used") is not False
            or not isinstance(result.get("evidence_id"), str)
            or not result["evidence_id"]):
        raise ValueError("Unexpected recorded-assessment response.")


def summary(row):
    return dict(id=row["id"], recorded_at=row["recorded_at"],
                case_id=row["case_id"], decision=row["decision"],
                mode="verified_revised_recorded_replay", sha256=row["sha256"])


def save(payload, identifier):
    validate(payload)
    contents = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(contents.encode("utf-8")) > 2_000_000:
        raise ValueError("Assessment exceeds history size limit.")
    timestamp = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    connection = connect()
    try:
        with connection:
            connection.execute("INSERT OR IGNORE INTO assessments VALUES (?, ?, ?, ?, ?, ?)",
                               (identifier, timestamp, payload["result"]["evidence_id"],
                                payload["result"]["decision"], contents, digest))
        row = connection.execute("SELECT * FROM assessments WHERE id=?", (identifier,)).fetchone()
        return summary(row)
    finally:
        connection.close()


def load(identifier):
    connection = connect()
    try:
        row = connection.execute("SELECT * FROM assessments WHERE id=?", (identifier,)).fetchone()
    finally:
        connection.close()
    if row is None:
        raise HTTPException(404, "Unknown saved C4 assessment.")
    if hashlib.sha256(row["payload"].encode("utf-8")).hexdigest() != row["sha256"]:
        raise ValueError("Saved assessment integrity check failed.")
    payload = json.loads(row["payload"])
    validate(payload)
    return {"entry": summary(row), "assessment": payload}


class RecordedRequest(BaseModel):
    request_id: UUID
    model_config = ConfigDict(extra="forbid")


@router.post("/evaluate")
def evaluate(request: RecordedRequest):
    identifier = str(request.request_id)
    try:
        try:
            return load(identifier)  # Retry with the same request ID is idempotent.
        except HTTPException as error:
            if error.status_code != 404:
                raise
        payload = worker.revised_evaluate("revised-outer-case", "hybrid", "adaptive")
        save(payload, identifier)
        return load(identifier)
    except worker.C4Unavailable:
        raise HTTPException(503, "C4 worker or evidence is unavailable. No new history entry saved.")
    except HTTPException:
        raise
    except Exception:
        log.exception("C4 recorded assessment/history failed")
        raise HTTPException(500, "C4 assessment could not be saved or verified. Inspect backend logs.")


@router.get("/history")
def history(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    try:
        connection = connect()
        try:
            rows = connection.execute("SELECT id, recorded_at, case_id, decision, sha256 FROM assessments ORDER BY recorded_at DESC, id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
            unique = connection.execute("SELECT COUNT(DISTINCT case_id) FROM assessments").fetchone()[0]
        finally:
            connection.close()
        return {"entries": [summary(row) for row in rows], "total": total,
                "unique_cases": unique, "offset": offset, "limit": limit,
                "scope": "Recorded assessment events; not live repairs or independent experiments."}
    except Exception:
        log.exception("C4 history read failed")
        raise HTTPException(503, "C4 assessment history is unavailable.")


@router.get("/history/{identifier}")
def detail(identifier: UUID):
    try:
        return load(str(identifier))
    except HTTPException:
        raise
    except Exception:
        log.exception("C4 saved assessment verification failed")
        raise HTTPException(503, "Saved assessment failed verification or could not be read.")
