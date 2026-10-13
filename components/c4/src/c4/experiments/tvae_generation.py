"""Independent TVAE generation for offline research preparation."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from c4.data import FEATURE_COLUMNS, TARGET_COLUMN, resolve_data_root
from c4.data.split_manifest import load_adult_split_manifest
from c4.models.champion import _git_revision
from c4.scenarios.feature_metadata import load_adult_feature_metadata


@dataclass(frozen=True)
class TVAEGenerationConfig:
    epochs: int = 100
    batch_size: int = 256
    n_rows: int = 500
    seed: int = 42

    def __post_init__(self):
        for name in ("epochs", "batch_size", "n_rows"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be a valid unsigned integer.")


@dataclass(frozen=True)
class GeneratedRepairArtifact:
    directory: Path
    synthetic_path: Path
    metadata_path: Path


def _make_tvae(config):
    # Lazy import keeps the research dependency outside core C4 imports.
    from ctgan import TVAE

    return TVAE(
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        loss_factor=2,
        batch_size=config.batch_size,
        epochs=config.epochs,
        enable_gpu=False,
        verbose=False,
    )


def generate_tvae_repair(
    *,
    preparation_directory: str | Path,
    manifest_directory: str | Path,
    artifact_version: str,
    data_root: str | Path | None = None,
    config: TVAEGenerationConfig | None = None,
) -> GeneratedRepairArtifact:
    """Generate data without claiming C3 approval or measured quality."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_version):
        raise ValueError("Artifact version has an invalid format.")

    config = config or TVAEGenerationConfig()
    root = resolve_data_root(data_root)
    preparation = Path(preparation_directory)
    directory = (
        root / "c4_artifacts" / "offline_repair" / "adult"
        / artifact_version
    )

    if directory.exists():
        raise FileExistsError(f"Repair version already exists: {directory}")

    metadata_bytes = (preparation / "metadata.json").read_bytes()
    preparation_hash = hashlib.sha256(metadata_bytes).hexdigest()

    if preparation_hash != (
        preparation / "metadata.sha256"
    ).read_text(encoding="utf-8").strip():
        raise ValueError("Preparation metadata checksum differs.")

    preparation_metadata = json.loads(metadata_bytes)

    if (
        preparation_metadata.get("purpose")
        != "development_repair_preparation"
        or preparation_metadata["provenance"]["source_role"] != "repair"
    ):
        raise ValueError("Verified repair-only preparation is required.")

    manifest_path = Path(manifest_directory) / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_adult_split_manifest(
        manifest_directory,
        data_root=root,
    )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during verification.")

    if hashlib.sha256(manifest_bytes).hexdigest() != (
        preparation_metadata["provenance"]["split_manifest_sha256"]
    ):
        raise ValueError("Preparation links to a different manifest.")

    verified_files = {}

    for filename in (
        "generator_fit.csv",
        "internal_validation.csv",
        "validation_diagnosed_membership.csv",
    ):
        contents = (preparation / filename).read_bytes()

        if hashlib.sha256(contents).hexdigest() != (
            preparation_metadata["files"][filename]
        ):
            raise ValueError(f"Preparation file checksum differs: {filename}")

        verified_files[filename] = contents

    fit = pd.read_csv(
        io.BytesIO(verified_files["generator_fit.csv"]),
        index_col=0,
    )
    provenance = preparation_metadata["provenance"]
    fit_ids = provenance["generator_fit_row_ids"]
    validation_ids = provenance["internal_validation_row_ids"]
    repair_ids = set(manifest["partitions"]["repair"]["source_row_ids"])

    if (
        not fit.index.is_unique
        or fit.index.tolist() != fit_ids
        or not set(fit_ids) <= repair_ids
        or not set(validation_ids) <= repair_ids
        or set(fit_ids) & set(validation_ids)
    ):
        raise ValueError("Generator-fitting row provenance is invalid.")

    columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

    if fit.columns.tolist() != columns:
        raise ValueError("Generator-fitting schema is invalid.")

    if len(fit) < 2 or set(fit[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Generator fitting requires both binary classes.")

    if (
        not all(pd.api.types.is_numeric_dtype(fit[c]) for c in columns)
        or not np.isfinite(fit.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Generator-fitting values must be numeric and finite.")

    feature_metadata = load_adult_feature_metadata()
    discrete_columns = [
        feature
        for feature, kind in feature_metadata.feature_types.items()
        if kind == "categorical"
    ] + [TARGET_COLUMN]

    for column in discrete_columns:
        values = fit[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Discrete column has fractional codes: {column}")

    generator = _make_tvae(config)
    generator.set_random_state(config.seed)
    generator.fit(fit, discrete_columns=discrete_columns)

    # Claim a new version after fitting; never overwrite existing artifacts.
    directory.mkdir(parents=True, exist_ok=False)
    generator_path = directory / "generator.pkl"

    # Save the fitted generator BEFORE sampling to retain its sampling state.
    joblib.dump(generator, generator_path)
    synthetic = generator.sample(config.n_rows)

    if (
        not isinstance(synthetic, pd.DataFrame)
        or synthetic.columns.tolist() != columns
        or len(synthetic) != config.n_rows
    ):
        raise ValueError("TVAE generated an invalid shape or schema.")

    if (
        not all(pd.api.types.is_numeric_dtype(synthetic[c]) for c in columns)
        or not np.isfinite(synthetic.to_numpy(dtype=float)).all()
    ):
        raise ValueError("TVAE generated nonnumeric or nonfinite values.")

    for column in discrete_columns:
        if not synthetic[column].isin(fit[column].unique()).all():
            raise ValueError(f"TVAE generated unknown discrete values: {column}")

    synthetic_path = directory / "synthetic_unvalidated.csv"
    synthetic.to_csv(synthetic_path, index=False)

    metadata = {
        "purpose": "development_synthetic_repair",
        "validation_status": "UNVALIDATED",
        "repair_package_approved": False,
        "generator": "TVAE",
        "artifact_version": artifact_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "generation_config": asdict(config),
        "architecture": {
            "embedding_dim": 128,
            "compress_dims": [128, 128],
            "decompress_dims": [128, 128],
            "l2scale": 1e-5,
            "loss_factor": 2,
            "enable_gpu": False,
        },
        "discrete_columns": discrete_columns,
        "feature_metadata_sha256": feature_metadata.source_sha256,
        "preparation_metadata_sha256": preparation_hash,
        "generator_fit_row_ids": fit_ids,
        "internal_validation_row_ids": validation_ids,
        "generator_fit_rows": len(fit),
        "synthetic_rows": len(synthetic),
        "generator_sha256": hashlib.sha256(
            generator_path.read_bytes()
        ).hexdigest(),
        "synthetic_csv_sha256": hashlib.sha256(
            synthetic_path.read_bytes()
        ).hexdigest(),
        "quality_metrics": None,
        "privacy_status": "NOT_EVALUATED",
        "release_validation_used_for_fitting": False,
        "sealed_future_used_for_fitting": False,
        "environment": {
            "python": platform.python_version(),
            "ctgan": version("ctgan"),
            "torch": version("torch"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    metadata_path = directory / "metadata.json"
    saved_metadata = (
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    metadata_path.write_bytes(saved_metadata)
    (directory / "metadata.sha256").write_text(
        hashlib.sha256(saved_metadata).hexdigest() + "\n",
        encoding="utf-8",
    )

    losses = getattr(generator, "loss_values", None)
    if isinstance(losses, pd.DataFrame):
        losses.to_csv(directory / "training_loss.csv", index=False)

    return GeneratedRepairArtifact(
        directory=directory,
        synthetic_path=synthetic_path,
        metadata_path=metadata_path,
    )