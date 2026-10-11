"""Candidate retraining using externally supplied C3 repair data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import hashlib
import io
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone

import sklearn
import xgboost

from c4.data import resolve_data_root
from c4.models.champion import (
    _git_revision,
    _sha256_file,
    load_champion,
)

from xgboost import XGBClassifier

from c4.contracts.models import C3RepairPackage
from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.models.champion import ChampionConfig, train_champion
from c4.repair import LoadedRepairPackage, load_repair_package


@dataclass(frozen=True)
class CandidateTrainingResult:
    model: XGBClassifier
    config: ChampionConfig
    repair: LoadedRepairPackage
    reference_rows: int
    repair_rows: int
    training_rows: int
    actual_augmentation_ratio: float


def train_candidate(
    reference: pd.DataFrame,
    package: C3RepairPackage,
    *,
    data_root: str | Path | None = None,
    config: ChampionConfig | None = None,
) -> CandidateTrainingResult:
    """Train a fresh candidate on reference plus approved repair rows.

    Release-validation and sealed-future data are not inputs.
    The caller must ensure C3 generated the package from eligible
    repair data only.
    """

    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

    if reference.columns.tolist() != expected_columns:
        raise ValueError("Reference dataframe has an invalid schema.")

    if reference.empty:
        raise ValueError("Reference dataframe is empty.")

    if not all(
        pd.api.types.is_numeric_dtype(reference[column])
        for column in expected_columns
    ):
        raise ValueError("Reference columns must be numeric.")

    if not np.isfinite(
        reference[expected_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Reference contains missing or infinite values."
        )

    if set(reference[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(
            "Reference training requires both binary classes."
        )

    repair = load_repair_package(package, data_root=data_root)
    config = config or ChampionConfig()

    training = pd.concat(
        [reference, repair.frame],
        ignore_index=True,
    )

    # Reuse the agreed configuration and training procedure.
    # This creates a new model; it does not modify the champion.
    model = train_champion(training, config)

    return CandidateTrainingResult(
        model=model,
        config=config,
        repair=repair,
        reference_rows=len(reference),
        repair_rows=len(repair.frame),
        training_rows=len(training),
        actual_augmentation_ratio=len(repair.frame) / len(reference),
    )

@dataclass(frozen=True)
class CandidateArtifact:
    model_path: Path
    metadata_path: Path
    metadata: dict[str, object]


def _validate_version(version: str) -> None:
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )

    if (
        not version
        or not version[0].isascii()
        or not version[0].isalnum()
        or any(character not in allowed for character in version)
    ):
        raise ValueError(
            "Version must start with a letter or number and contain "
            "only ASCII letters, numbers, '.', '_' or '-'."
        )


def train_and_save_candidate(
    package: C3RepairPackage,
    *,
    artifact_version: str,
    champion_version: str = "v1",
    data_root: str | Path | None = None,
) -> CandidateArtifact:
    """Train and save a candidate using the champion's configuration."""

    _validate_version(artifact_version)
    _validate_version(champion_version)
    root = resolve_data_root(data_root)

    directory = (
        root / "c4_artifacts" / "candidate" / "adult"
        / artifact_version
    )

    if directory.exists():
        raise FileExistsError(
            f"Candidate version already exists: {directory}"
        )

    champion_directory = (
        root / "c4_artifacts" / "champion" / "adult"
        / champion_version
    )
    champion, champion_metadata = load_champion(champion_directory)

    if champion_metadata.get("artifact_version") != champion_version:
        raise ValueError("Champion version does not match metadata.")

    if champion.get_booster().feature_names != list(FEATURE_COLUMNS):
        raise ValueError("Champion feature schema is incompatible.")

    config = ChampionConfig(**champion_metadata["training_config"])

    # Read and hash the same reference bytes. No test/future CSV is read.
    reference_path = root / "uci_adult_data" / "reference.csv"
    reference_bytes = reference_path.read_bytes()
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()

    if reference_sha256 != champion_metadata["reference_csv_sha256"]:
        raise ValueError(
            "Reference CSV differs from the champion's training data."
        )

    reference = pd.read_csv(io.BytesIO(reference_bytes))
    result = train_candidate(
        reference,
        package,
        data_root=root,
        config=config,
    )

    # Claim a new directory; existing versions are never overwritten.
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "candidate.ubj"
    metadata_path = directory / "metadata.json"
    result.model.save_model(model_path)

    metadata: dict[str, object] = {
        "artifact_schema_version": "1.0",
        "artifact_version": artifact_version,
        "role": "candidate",
        "dataset": "uci_adult",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "champion_version": champion_version,
        "champion_model_sha256": champion_metadata["model_sha256"],
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "training_config": asdict(config),
        "training_strategy": "fresh_fit_reference_plus_repair",
        "reference_rows": result.reference_rows,
        "repair_rows": result.repair_rows,
        "training_rows": result.training_rows,
        "actual_augmentation_ratio": result.actual_augmentation_ratio,
        "reference_csv_sha256": reference_sha256,
        "repair_csv_sha256": result.repair.dataset_sha256,
        "repair_dataset_path": str(result.repair.dataset_path),
        "c3_package": package.model_dump(mode="json"),
        "model_sha256": _sha256_file(model_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "data_usage": {
            "reference_used_for_training": True,
            "approved_repair_used_for_training": True,
            "release_validation_used_for_training": False,
            "sealed_future_used_for_training": False,
        },
        "provenance_note": (
            "C3 must establish that repair generation excluded "
            "release-validation and sealed-future data."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    return CandidateArtifact(
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def load_candidate(
    artifact_directory: str | Path,
) -> tuple[XGBClassifier, dict[str, object]]:
    """Reload a candidate and verify its model checksum and schema."""

    directory = Path(artifact_directory)
    model_path = directory / "candidate.ubj"
    metadata_path = directory / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("role") != "candidate":
        raise ValueError("Artifact metadata does not describe a candidate.")

    if _sha256_file(model_path) != metadata.get("model_sha256"):
        raise ValueError("Candidate model checksum does not match metadata.")

    if metadata.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Candidate metadata feature schema is incompatible.")

    model = XGBClassifier()
    model.load_model(model_path)

    if model.get_booster().feature_names != list(FEATURE_COLUMNS):
        raise ValueError("Candidate model feature schema is incompatible.")

    return model, metadata