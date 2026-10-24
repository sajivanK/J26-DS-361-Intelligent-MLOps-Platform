"""Versioned operational candidate persistence, separate from research artifacts.

Reload verifies stored byte integrity. It does not deserialize a model or grant
release approval. Callers must retain the returned manifest digest as an audit
anchor; a checksum stored beside editable files alone is not authentication.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from c4.models.runtime_candidate import RuntimeCandidateResult


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encode(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(data):
    def reject(value): raise ValueError(f"Invalid JSON constant: {value}")
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("Duplicate JSON key.")
            out[key] = value
        return out
    value = json.loads(data.decode("utf-8"), parse_constant=reject, object_pairs_hook=pairs)
    if not isinstance(value, dict): raise ValueError("Expected JSON object.")
    return value


def _validate_result(result):
    report = result.report
    if not isinstance(result.candidate_model_bytes, bytes) or not result.candidate_model_bytes:
        raise ValueError("Candidate model bytes are missing.")
    if report.get("status") != "CANDIDATE_EVALUATED_RELEASE_PENDING":
        raise ValueError("Expected evaluated candidate with release still pending.")
    if report.get("candidate_model_sha256") != _sha(result.candidate_model_bytes):
        raise ValueError("Candidate bytes differ from evaluation report.")
    for key in ("candidate_training_executed", "candidate_reload_verified", "model_runtime_validated"):
        if report.get(key) is not True: raise ValueError(f"Missing prerequisite: {key}")
    for key in ("production_release_authorized", "forecast_applicable", "N1_executed",
                "N2_executed", "future_outcomes_read", "validation_used_for_training"):
        if report.get(key) is not False: raise ValueError(f"Unsupported report state: {key}")
    if report.get("artifact_persistence_completed") is not False:
        raise ValueError("Expected original evaluation report before persistence.")
    if not isinstance(report.get("run_id"), str) or not report["run_id"].strip():
        raise ValueError("Missing runtime run ID.")
    _encode(report)  # Reject non-JSON data and NaN before creating output.


@dataclass(frozen=True)
class RuntimeArtifact:
    directory: Path
    manifest_sha256: str


def save_runtime_candidate(result: RuntimeCandidateResult, *, data_root, version: str) -> RuntimeArtifact:
    """Write one new version; leave partial directories intact on I/O failure."""
    if (not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", version)
            or version.endswith(".") or version.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Invalid operational artifact version.")
    _validate_result(result)
    # Snapshot before serialization; never add fields to the caller's report.
    report_bytes = _encode(deepcopy(result.report))
    directory = Path(data_root).expanduser().resolve() / "c4_artifacts" / "operational_candidates" / "adult" / version
    manifest = {
        "schema_version": "1.0", "role": "operational_candidate_evaluation",
        "version": version, "run_id": result.report["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {"candidate.ubj": _sha(result.candidate_model_bytes), "evaluation.json": _sha(report_bytes)},
        "artifact_persistence_completed": True, "production_release_authorized": False,
    }
    manifest_bytes = _encode(manifest)
    directory.mkdir(parents=True, exist_ok=False)
    for name, contents in (("candidate.ubj", result.candidate_model_bytes), ("evaluation.json", report_bytes)):
        with (directory / name).open("xb") as stream: stream.write(contents)
        if (directory / name).read_bytes() != contents:
            raise ValueError("Artifact write verification failed; preserve partial output.")
    # Manifest is the completion record and is written last.
    with (directory / "manifest.json").open("xb") as stream: stream.write(manifest_bytes)
    artifact = RuntimeArtifact(directory, _sha(manifest_bytes))
    load_runtime_candidate(artifact.directory, expected_manifest_sha256=artifact.manifest_sha256)
    return artifact


def load_runtime_candidate(directory, *, expected_manifest_sha256: str):
    """Return verified model bytes and original evaluation report plus manifest."""
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("An independently retained manifest SHA256 is required.")
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    if _sha(raw) != expected_manifest_sha256: raise ValueError("Manifest checksum mismatch.")
    manifest = _read_json(raw)
    if (manifest.get("schema_version") != "1.0"
            or manifest.get("role") != "operational_candidate_evaluation"
            or manifest.get("artifact_persistence_completed") is not True
            or manifest.get("production_release_authorized") is not False):
        raise ValueError("Invalid operational completion manifest.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"candidate.ubj", "evaluation.json"}:
        raise ValueError("Unexpected operational artifact file manifest.")
    contents = {}
    for name, checksum in files.items():
        contents[name] = (directory / name).read_bytes()
        if _sha(contents[name]) != checksum: raise ValueError(f"Artifact checksum mismatch: {name}")
    result = RuntimeCandidateResult(contents["candidate.ubj"], _read_json(contents["evaluation.json"]))
    _validate_result(result)
    if result.report["run_id"] != manifest.get("run_id"):
        raise ValueError("Evaluation run ID differs from manifest.")
    return result, manifest
