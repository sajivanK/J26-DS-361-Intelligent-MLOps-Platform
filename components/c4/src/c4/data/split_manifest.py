"""Versioned provenance for Adult role partitions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from c4.data.adult import load_adult_dataset, resolve_data_root
from c4.data.splits import AdultSplitConfig, split_adult_roles


@dataclass(frozen=True)
class SplitManifestArtifact:
    manifest_path: Path
    checksum_path: Path
    manifest_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def save_adult_split_manifest(
    *,
    manifest_version: str = "v1",
    data_root: str | Path | None = None,
    config: AdultSplitConfig | None = None,
) -> SplitManifestArtifact:
    """Persist partition IDs without storing future performance outcomes."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", manifest_version):
        raise ValueError("Manifest version has an invalid format.")

    root = resolve_data_root(data_root)
    directory = (
        root / "c4_artifacts" / "splits" / "adult" / manifest_version
    )

    if directory.exists():
        raise FileExistsError(
            f"Split manifest version already exists: {directory}"
        )

    config = config or AdultSplitConfig()
    reference_path = root / "uci_adult_data" / "reference.csv"
    test_path = root / "uci_adult_data" / "test.csv"

    source_hashes = {
        "reference_csv_sha256": _file_sha256(reference_path),
        "test_csv_sha256": _file_sha256(test_path),
    }

    dataset = load_adult_dataset(root)
    splits = split_adult_roles(dataset, config)

    # Refuse to record provenance if files changed during this operation.
    if (
        _file_sha256(reference_path)
        != source_hashes["reference_csv_sha256"]
        or _file_sha256(test_path)
        != source_hashes["test_csv_sha256"]
    ):
        raise ValueError("Source CSV changed during manifest creation.")

    partitions = {}

    for role in ("repair", "release_validation", "sealed_future"):
        frame = getattr(splits, role)
        partitions[role] = {
            "rows": len(frame),
            "source_row_ids": [int(index) for index in frame.index],
        }

    payload = {
        "manifest_schema_version": "1.0",
        "manifest_version": manifest_version,
        "dataset": "uci_adult",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_row_id_definition": (
            "Zero-based data-row position in the original test.csv, "
            "excluding its header."
        ),
        "split_config": asdict(config),
        "source_checksums": source_hashes,
        "reference_rows": len(splits.reference),
        "holdout_audit": asdict(splits.audit),
        "partitions": partitions,
        "future_note": (
            "Future row IDs are partition metadata. No future model "
            "predictions, performance metrics or margins are stored."
        ),
    }

    contents = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    checksum = hashlib.sha256(contents).hexdigest()

    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "manifest.json"
    checksum_path = directory / "manifest.sha256"

    manifest_path.write_bytes(contents)
    checksum_path.write_text(checksum + "\n", encoding="utf-8")

    return SplitManifestArtifact(
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        manifest_sha256=checksum,
    )


def load_adult_split_manifest(
    manifest_directory: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify manifest integrity and its current source CSV checksums."""

    directory = Path(manifest_directory)
    contents = (directory / "manifest.json").read_bytes()
    expected = (directory / "manifest.sha256").read_text(
        encoding="utf-8"
    ).strip()

    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError("Split manifest checksum does not match.")

    manifest = json.loads(contents)

    if (
        manifest.get("manifest_schema_version") != "1.0"
        or manifest.get("dataset") != "uci_adult"
    ):
        raise ValueError("Unsupported split manifest.")

    root = resolve_data_root(data_root)

    for filename, field in [
        ("reference.csv", "reference_csv_sha256"),
        ("test.csv", "test_csv_sha256"),
    ]:
        actual = _file_sha256(root / "uci_adult_data" / filename)

        if actual != manifest["source_checksums"][field]:
            raise ValueError(
                f"Source CSV checksum differs from manifest: {filename}"
            )

    role_sets = []

    for role in ("repair", "release_validation", "sealed_future"):
        partition = manifest["partitions"][role]
        ids = partition["source_row_ids"]

        if (
            any(type(value) is not int or value < 0 for value in ids)
            or len(ids) != partition["rows"]
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Manifest contains invalid partition IDs.")

        role_sets.append(set(ids))

    if any(
        role_sets[first] & role_sets[second]
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Manifest role partitions overlap.")

    if sum(map(len, role_sets)) != manifest["holdout_audit"]["eligible_rows"]:
        raise ValueError("Manifest partition counts are inconsistent.")

    return manifest


def assert_release_membership(
    frame: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    """Require the complete release partition with preserved source IDs."""

    expected = manifest["partitions"]["release_validation"]["source_row_ids"]

    if (
        not frame.index.is_unique
        or not pd.api.types.is_integer_dtype(frame.index.dtype)
        or len(frame) != len(expected)
        or set(frame.index) != set(expected)
    ):
        raise ValueError(
            "Evaluation rows do not match the release-validation partition."
        )


def verify_clean_release_source(
    clean_frame: pd.DataFrame,
    manifest: dict[str, object],
    *,
    data_root: str | Path | None = None,
) -> None:
    """Verify clean release contents against the processed source CSV.

    This reader assumes the project's numeric processed CSV format:
    one header line followed by one data record per line, without blank
    lines or multiline records.
    """

    assert_release_membership(clean_frame, manifest)
    root = resolve_data_root(data_root)
    source_path = root / "uci_adult_data" / "test.csv"

    if _file_sha256(source_path) != (
        manifest["source_checksums"]["test_csv_sha256"]
    ):
        raise ValueError("Source CSV checksum differs from manifest.")

    release_ids = set(
        manifest["partitions"]["release_validation"]["source_row_ids"]
    )

    # Parse release records only. Future records are skipped.
    source_release = pd.read_csv(
        source_path,
        skiprows=lambda line_number: (
            line_number > 0
            and line_number - 1 not in release_ids
        ),
    )

    if len(source_release) != len(release_ids):
        raise ValueError("Source CSV record layout is incompatible.")

    source_release.index = sorted(release_ids)
    expected = source_release.loc[clean_frame.index]

    try:
        pd.testing.assert_frame_equal(
            clean_frame,
            expected,
            check_dtype=False,
            check_exact=True,
            check_names=False,
        )
    except AssertionError as error:
        raise ValueError(
            "Clean release contents differ from the original source rows."
        ) from error

    if _file_sha256(source_path) != (
        manifest["source_checksums"]["test_csv_sha256"]
    ):
        raise ValueError("Source CSV changed during release verification.")
    