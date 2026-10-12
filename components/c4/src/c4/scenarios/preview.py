"""Verified offline transformation previews, separate from experiments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from c4.data import resolve_data_root
from c4.data.split_manifest import load_adult_split_manifest
from c4.scenarios.adapter import transform_role
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule


@dataclass(frozen=True)
class RolePreviewArtifact:
    directory: Path
    metadata_path: Path
    metadata_sha256: str


def save_role_preview(
    *,
    manifest_directory: str | Path,
    config: ScenarioConfig,
    true_slice: SliceRule,
    role: str,
    preview_id: str,
    data_root: str | Path | None = None,
) -> RolePreviewArtifact:
    """Load verified role rows, transform them and save a preview.

    Supports the project's processed numeric CSV format: one header
    followed by one record per line, without blank or multiline records.

    This function does not run models, generate upstream evidence
    or authorize benchmark execution.
    """

    if role not in {"repair", "release_validation"}:
        raise ValueError("Only repair and release_validation roles are allowed.")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", preview_id):
        raise ValueError("Preview ID has an invalid format.")

    root = resolve_data_root(data_root)
    directory = (
        root / "c4_artifacts" / "scenario_previews" / "adult"
        / preview_id / role
    )

    if directory.exists():
        raise FileExistsError(f"Preview already exists: {directory}")

    manifest_path = Path(manifest_directory) / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_adult_split_manifest(
        manifest_directory,
        data_root=root,
    )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during verification.")

    ids = manifest["partitions"][role]["source_row_ids"]
    selected_ids = set(ids)
    source_path = root / "uci_adult_data" / "test.csv"

    # Parse only the selected role's records.
    clean = pd.read_csv(
        source_path,
        skiprows=lambda line_number: (
            line_number > 0
            and line_number - 1 not in selected_ids
        ),
    )

    if len(clean) != len(ids):
        raise ValueError("Source CSV record layout is incompatible.")

    clean.index = sorted(selected_ids)
    clean = clean.loc[ids].copy()

    # Verify the source still matches the manifest after parsing.
    actual_source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    if actual_source_hash != (
        manifest["source_checksums"]["test_csv_sha256"]
    ):
        raise ValueError("Source CSV changed during preview preparation.")

    transformed = transform_role(
        clean,
        config=config,
        true_slice=true_slice,
        role=role,
    )

    clean_bytes = clean.to_csv(
        index=True, lineterminator="\n"
    ).encode("utf-8")
    transformed_bytes = transformed.frame.to_csv(
        index=True, lineterminator="\n"
    ).encode("utf-8")
    membership_bytes = pd.DataFrame({
        "source_row_id": clean.index,
        "true_slice_member": transformed.true_membership.to_numpy(),
    }).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")

    metadata = {
        "artifact_schema_version": "0.1",
        "purpose": "development_preview",
        "benchmark_execution": False,
        "preview_id": preview_id,
        "role": role,
        "split_manifest_version": manifest["manifest_version"],
        "split_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "source_test_csv_sha256": actual_source_hash,
        "source_row_ids": ids,
        "transformation": transformed.provenance,
        "files": {
            "clean.csv": hashlib.sha256(clean_bytes).hexdigest(),
            "transformed.csv": hashlib.sha256(
                transformed_bytes
            ).hexdigest(),
            "true_membership.csv": hashlib.sha256(
                membership_bytes
            ).hexdigest(),
        },
    }
    metadata_bytes = (
        json.dumps(
            metadata, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
    ).encode("utf-8")
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "clean.csv").write_bytes(clean_bytes)
    (directory / "transformed.csv").write_bytes(transformed_bytes)
    (directory / "true_membership.csv").write_bytes(membership_bytes)

    metadata_path = directory / "metadata.json"
    metadata_path.write_bytes(metadata_bytes)
    (directory / "metadata.sha256").write_text(
        metadata_sha256 + "\n",
        encoding="utf-8",
    )

    return RolePreviewArtifact(
        directory=directory,
        metadata_path=metadata_path,
        metadata_sha256=metadata_sha256,
    )