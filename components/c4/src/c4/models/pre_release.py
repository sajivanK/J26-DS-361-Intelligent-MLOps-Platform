"""Save decision-time pre-release evidence outside the repository."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from c4.contracts.models import C2Diagnosis
from c4.data import resolve_data_root
from c4.models.candidate import load_candidate
from c4.models.champion import _git_revision, load_champion
from c4.models.evaluation import evaluate_pre_release

from c4.data.split_manifest import (
    assert_release_membership,
    load_adult_split_manifest,
    verify_clean_release_source,
)

from c4.scenarios.diagnosis import diagnosed_slice_mask


@dataclass(frozen=True)
class PreReleaseArtifact:
    evidence_path: Path
    checksum_path: Path
    evidence_sha256: str


def save_pre_release_evidence(
    *,
    champion_directory: str | Path,
    candidate_directory: str | Path,
    split_manifest_directory: str | Path,
    clean_release_validation: pd.DataFrame,
    release_validation: pd.DataFrame,
    diagnosis: C2Diagnosis,
    rule_feature_space: str,
    scenario_id: str,
    data_root: str | Path | None = None,
) -> PreReleaseArtifact:
    """Evaluate using C2 membership computed on clean release rows.

    Both clean and scenario-transformed frames must retain the original
    release source indices in the same order.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scenario_id):
        raise ValueError("Scenario ID has an invalid format.")

    root = resolve_data_root(data_root)

    manifest_directory = Path(split_manifest_directory)
    manifest_path = manifest_directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()

    manifest = load_adult_split_manifest(
        manifest_directory,
        data_root=root,
    )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Split manifest changed during verification.")

    assert_release_membership(clean_release_validation, manifest)
    assert_release_membership(release_validation, manifest)

    if not clean_release_validation.index.equals(
        release_validation.index
    ):
        raise ValueError(
            "Clean and evaluated release rows must have identical index order."
        )

    if clean_release_validation.columns.tolist() != (
        release_validation.columns.tolist()
    ):
        raise ValueError("Clean and evaluated release schemas differ.")

    verify_clean_release_source(
        clean_release_validation,
        manifest,
        data_root=root,
    )

    diagnosed_mask = diagnosed_slice_mask(
        clean_release_validation,
        diagnosis,
        rule_feature_space=rule_feature_space,
    )

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    directory = (
        root / "c4_artifacts" / "pre_release" / "adult" / scenario_id
    )

    if directory.exists():
        raise FileExistsError(
            f"Pre-release evidence already exists: {directory}"
        )

    # Source indices identify positions in the original test CSV.
    if not pd.api.types.is_integer_dtype(release_validation.index.dtype):
        raise ValueError("Source row indices must be integers.")

    if (release_validation.index < 0).any():
        raise ValueError("Source row indices must be nonnegative.")

    champion, champion_metadata = load_champion(champion_directory)
    candidate, candidate_metadata = load_candidate(candidate_directory)

    if (
        candidate_metadata["champion_model_sha256"]
        != champion_metadata["model_sha256"]
    ):
        raise ValueError(
            "Candidate belongs to a different champion artifact."
        )

    report = evaluate_pre_release(
        champion,
        candidate,
        release_validation,
        diagnosed_mask,
    )

    # Store the actual evaluated rows for later reproduction.
    clean_release_bytes = clean_release_validation.to_csv(
        index=True,
        index_label="source_row_id",
        lineterminator="\n",
    ).encode("utf-8")

    release_bytes = release_validation.to_csv(
        index=True,
        index_label="source_row_id",
        lineterminator="\n",
    ).encode("utf-8")

    membership_bytes = pd.DataFrame({
        "source_row_id": release_validation.index,
        "diagnosed_slice_member": diagnosed_mask.to_numpy(dtype=bool),
    }).to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")

    payload = {
        "evidence_schema_version": "1.0",
        "scenario_id": scenario_id,
        "split_manifest_version": manifest["manifest_version"],
        "split_manifest_sha256": manifest_sha256,
        "evaluation_partition": "release_validation",
        "release_membership_verified": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "slice_basis": "c2_diagnosed_slice",
        "c2_diagnosis": diagnosis.model_dump(mode="json"),
        "champion_version": champion_metadata["artifact_version"],
        "champion_model_sha256": champion_metadata["model_sha256"],
        "candidate_version": candidate_metadata["artifact_version"],
        "candidate_model_sha256": candidate_metadata["model_sha256"],
        "repair_version": candidate_metadata["c3_package"]["repair_version"],
        "repair_csv_sha256": candidate_metadata["repair_csv_sha256"],
        "release_validation_csv_sha256": hashlib.sha256(
            release_bytes
        ).hexdigest(),
        "diagnosed_membership_csv_sha256": hashlib.sha256(
            membership_bytes
        ).hexdigest(),
        "report": report,
        "provenance_note": (
            "Source checksums, release row IDs and clean release contents "
            "were verified. C2 membership was computed before perturbation. "
            "The scenario harness must establish that the evaluated "
            "transformations preserve source-row identity."
        ),
        "diagnosis_rule_feature_space": rule_feature_space,
        "membership_basis": "clean_release_before_perturbation",
        "clean_release_validation_csv_sha256": hashlib.sha256(
            clean_release_bytes
        ).hexdigest(),
        "clean_release_contents_verified": True,
    }

    evidence_bytes = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()

    # Refuse overwriting an existing scenario evidence directory.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "clean_release_validation.csv").write_bytes(
        clean_release_bytes
    )
    (directory / "release_validation.csv").write_bytes(release_bytes)
    (directory / "diagnosed_membership.csv").write_bytes(membership_bytes)

    evidence_path = directory / "evidence.json"
    checksum_path = directory / "evidence.sha256"
    evidence_path.write_bytes(evidence_bytes)
    checksum_path.write_text(
        evidence_sha256 + "\n",
        encoding="utf-8",
    )

    return PreReleaseArtifact(
        evidence_path=evidence_path,
        checksum_path=checksum_path,
        evidence_sha256=evidence_sha256,
    )
