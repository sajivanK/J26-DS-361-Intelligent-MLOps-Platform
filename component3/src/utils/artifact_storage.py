"""Artifact storage management for Component 3.

Provides portable, deterministic, atomic, and cryptographically verified storage
for synthetic repair datasets, candidate models, and execution manifests.

Supports:
- Local development (default: component3/results/runs/)
- Configurable team deployment storage root via C3_ARTIFACT_STORAGE_ROOT
- Forward-slash portable relative URIs for cross-platform contracts
- Atomic writes via temporary files to avoid partial writes
- SHA-256 integrity verification and clear file-not-found handling
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "component3" / "results" / "runs"
STORAGE_ENV_VAR = "C3_ARTIFACT_STORAGE_ROOT"


def get_configured_storage_root() -> Path:
    """Return the active storage root directory.

    Checks C3_ARTIFACT_STORAGE_ROOT environment variable, falling back to
    repo-relative component3/results/runs/.
    """
    env_path = os.getenv(STORAGE_ENV_VAR)
    if env_path and env_path.strip():
        p = Path(env_path.strip())
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p.resolve()
    return DEFAULT_RUNS_DIR.resolve()


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file in 64KB chunks.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IsADirectoryError: If the path is a directory.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Cannot compute SHA-256: file not found at '{p}'.")
    if p.is_dir():
        raise IsADirectoryError(f"Cannot compute SHA-256: path is a directory '{p}'.")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_integrity(filepath: Path | str, expected_hash: str) -> tuple[bool, str]:
    """Verify that a file exists and matches the expected SHA-256 hash.

    Returns:
        (is_valid, message)
    """
    p = Path(filepath)
    if not p.exists():
        return False, f"Artifact file does not exist: '{p}'"
    if not p.is_file():
        return False, f"Artifact path is not a regular file: '{p}'"

    actual_hash = compute_file_sha256(p)
    if actual_hash.lower() != expected_hash.lower():
        return (
            False,
            f"SHA-256 integrity mismatch for '{p.name}': expected {expected_hash}, got {actual_hash}",
        )
    return True, f"Integrity verified for '{p.name}' ({actual_hash[:16]}...)"


def to_portable_relative_uri(path: Path | str, base_root: Path | None = None) -> str:
    """Convert a path into a portable, forward-slash relative URI.

    Avoids leaking Windows drive letters (e.g., 'D:\\...') into external contracts.
    """
    p = Path(path)
    root = (base_root or REPO_ROOT).resolve()
    try:
        resolved = p.resolve()
        rel = resolved.relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        # Fallback if path is outside root: return clean posix name or string
        return str(p).replace("\\", "/")


def resolve_portable_uri(uri: str, base_root: Path | None = None) -> Path:
    """Resolve a portable forward-slash URI to a concrete filesystem Path.

    Checks existence and raises FileNotFoundError with actionable context.
    """
    clean_uri = uri.strip().replace("\\", "/")
    root = (base_root or REPO_ROOT).resolve()

    p = Path(clean_uri)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (root / p).resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Portable URI '{uri}' could not be resolved to an existing file. Looked at: '{resolved}'."
        )
    return resolved


class ArtifactStorageManager:
    """Manages creation, atomic writing, and retrieval of repair artifacts."""

    def __init__(self, base_directory: Path | str | None = None) -> None:
        if base_directory is not None:
            p = Path(base_directory)
            self.base_dir = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        else:
            self.base_dir = get_configured_storage_root()

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_directory(self, run_id: str | None = None) -> Path:
        """Create a dedicated, unique directory for an execution run."""
        rid = run_id or f"RUN-{uuid.uuid4().hex[:10].upper()}"
        run_dir = self.base_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def atomic_write_dataframe(df: pd.DataFrame, dest_path: Path | str) -> tuple[Path, str]:
        """Atomically write a DataFrame to CSV using a temporary file and compute SHA-256."""
        p = Path(dest_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            df.to_csv(tmp, index=False)
            tmp.replace(p)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        sha = compute_file_sha256(p)
        return p, sha

    @staticmethod
    def atomic_write_json(data: Any, dest_path: Path | str, indent: int = 2) -> tuple[Path, str]:
        """Atomically write serializable data to JSON and compute SHA-256."""
        p = Path(dest_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            tmp.replace(p)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        sha = compute_file_sha256(p)
        return p, sha

    @staticmethod
    def atomic_save_model(model_obj: Any, dest_path: Path | str) -> tuple[Path, str]:
        """Atomically dump a fitted model to disk using joblib and compute SHA-256."""
        p = Path(dest_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            joblib.dump(model_obj, tmp)
            tmp.replace(p)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        sha = compute_file_sha256(p)
        return p, sha
