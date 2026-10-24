"""Prepare repair-only data for independent synthetic-data experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from c4.contracts.models import C2Diagnosis
from c4.data import resolve_data_root
from c4.data.split_manifest import load_adult_split_manifest
from c4.scenarios.diagnosis import diagnosed_slice_mask


@dataclass(frozen=True)
class PreparedRepairData:
    generator_fit: pd.DataFrame
    internal_validation: pd.DataFrame
    validation_diagnosed_membership: pd.Series
    provenance: dict[str, object]


def prepare_repair_data(
    *,
    preview_directory: str | Path,
    manifest_directory: str | Path,
    diagnosis: C2Diagnosis,
    rule_feature_space: str,
    data_root: str | Path | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
) -> PreparedRepairData:
    """Split verified repair rows before fitting any generator.

    Diagnosis membership is calculated on clean repair rows.
    Generator inputs and internal validation use their corresponding
    scenario-transformed rows.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be between zero and one.")

    if type(split_seed) is not int or not 0 <= split_seed <= 2**32 - 1:
        raise ValueError("Split seed must be a valid unsigned integer.")

    root = resolve_data_root(data_root)
    directory = Path(preview_directory)

    metadata_bytes = (directory / "metadata.json").read_bytes()
    expected_metadata_hash = (
        directory / "metadata.sha256"
    ).read_text(encoding="utf-8").strip()

    if hashlib.sha256(metadata_bytes).hexdigest() != expected_metadata_hash:
        raise ValueError("Preview metadata checksum does not match.")

    metadata = json.loads(metadata_bytes)

    if (
        metadata.get("purpose") != "development_preview"
        or metadata.get("role") != "repair"
    ):
        raise ValueError("A repair-role development preview is required.")

    manifest_path = Path(manifest_directory) / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_adult_split_manifest(
        manifest_directory,
        data_root=root,
    )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during verification.")

    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    if manifest_hash != metadata["split_manifest_sha256"]:
        raise ValueError("Preview links to a different split manifest.")

    if metadata["source_test_csv_sha256"] != (
        manifest["source_checksums"]["test_csv_sha256"]
    ):
        raise ValueError("Preview source checksum differs from manifest.")

    file_bytes = {}

    for filename in ("clean.csv", "transformed.csv", "true_membership.csv"):
        contents = (directory / filename).read_bytes()

        if hashlib.sha256(contents).hexdigest() != (
            metadata["files"][filename]
        ):
            raise ValueError(f"Preview file checksum differs: {filename}")

        file_bytes[filename] = contents

    # Parse the exact bytes that were verified.
    import io

    clean = pd.read_csv(io.BytesIO(file_bytes["clean.csv"]), index_col=0)
    transformed = pd.read_csv(
        io.BytesIO(file_bytes["transformed.csv"]),
        index_col=0,
    )
    expected_ids = manifest["partitions"]["repair"]["source_row_ids"]

    if (
        not clean.index.is_unique
        or clean.index.tolist() != expected_ids
        or not transformed.index.equals(clean.index)
        or transformed.columns.tolist() != clean.columns.tolist()
    ):
        raise ValueError("Preview rows do not match the repair partition.")

    if not transformed["target"].equals(clean["target"]):
        raise ValueError("Repair transformation changed target labels.")

    membership = diagnosed_slice_mask(
        clean,
        diagnosis,
        rule_feature_space=rule_feature_space,
    )

    # Split the entire repair role first, before filtering generator inputs.
    train_ids, validation_ids = train_test_split(
        clean.index.to_numpy(),
        test_size=validation_fraction,
        stratify=clean["target"].to_numpy(),
        random_state=split_seed,
    )
    train_ids = sorted(int(value) for value in train_ids)
    validation_ids = sorted(int(value) for value in validation_ids)

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Internal repair partitions overlap.")

    if set(train_ids) | set(validation_ids) != set(expected_ids):
        raise RuntimeError("Internal repair partitions are incomplete.")

    diagnosed_train_ids = [
        row_id for row_id in train_ids
        if bool(membership.loc[row_id])
    ]
    generator_fit = transformed.loc[diagnosed_train_ids].copy()
    internal_validation = transformed.loc[validation_ids].copy()

    if len(generator_fit) < 2:
        raise ValueError("Insufficient diagnosed generator-fitting rows.")

    if set(generator_fit["target"].unique()) != {0, 1}:
        raise ValueError("Generator-fitting data requires both target classes.")

    return PreparedRepairData(
        generator_fit=generator_fit,
        internal_validation=internal_validation,
        validation_diagnosed_membership=membership.loc[
            validation_ids
        ].copy(),
        provenance={
            "purpose": "offline_repair_preparation",
            "source_role": "repair",
            "preview_metadata_sha256": expected_metadata_hash,
            "split_manifest_sha256": manifest_hash,
            "diagnosis": diagnosis.model_dump(mode="json"),
            "rule_feature_space": rule_feature_space,
            "validation_fraction": validation_fraction,
            "split_seed": split_seed,
            "repair_train_row_ids": train_ids,
            "generator_fit_row_ids": diagnosed_train_ids,
            "internal_validation_row_ids": validation_ids,
            "release_validation_used": False,
            "sealed_future_used": False,
        },
    )