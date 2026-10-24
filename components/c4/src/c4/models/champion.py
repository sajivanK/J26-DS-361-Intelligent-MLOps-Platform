"""Reproducible UCI Adult champion training and versioning."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

from c4.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_adult_dataset,
    resolve_data_root,
)


@dataclass(frozen=True)
class ChampionConfig:
    """Frozen XGBoost champion configuration."""

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    eval_metric: str = "logloss"
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class ChampionArtifact:
    """Paths and metadata for a saved champion."""

    model_path: Path
    metadata_path: Path
    metadata: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def train_champion(
    reference: pd.DataFrame,
    config: ChampionConfig | None = None,
) -> XGBClassifier:
    """Train the champion using only the reference dataset."""

    config = config or ChampionConfig()

    if tuple(reference.columns) != (*FEATURE_COLUMNS, TARGET_COLUMN):
        raise ValueError("Reference dataframe has an invalid schema.")

    if reference.empty:
        raise ValueError("Reference dataframe is empty.")

    features = reference.loc[:, FEATURE_COLUMNS]
    target = reference[TARGET_COLUMN]

    if set(target.unique()) != {0, 1}:
        raise ValueError(
            "Champion training requires both binary target classes."
        )

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        **asdict(config),
    )

    model.fit(features, target)

    return model


def train_and_save_champion(
    data_root: str | Path | None = None,
    artifact_version: str = "v1",
    config: ChampionConfig | None = None,
) -> ChampionArtifact:
    """Train and save a versioned champion outside the Git repository."""

    if not artifact_version:
        raise ValueError("Artifact version must not be empty.")

    if any(
        character not in "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in artifact_version
    ):
        raise ValueError(
            "Artifact version may contain only letters, numbers, '.', "
            "'_' and '-'."
        )

    config = config or ChampionConfig()
    root = resolve_data_root(data_root)
    dataset = load_adult_dataset(root)

    reference_path = (
        root / "uci_adult_data" / "reference.csv"
    )
    artifact_directory = (
        root
        / "c4_artifacts"
        / "champion"
        / "adult"
        / artifact_version
    )
    model_path = artifact_directory / "champion.ubj"
    metadata_path = artifact_directory / "metadata.json"

    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Champion version already exists: {artifact_directory}"
        )

    artifact_directory.mkdir(parents=True, exist_ok=True)

    model = train_champion(dataset.reference, config)
    model.save_model(model_path)

    target_counts = (
        dataset.reference[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    metadata: dict[str, object] = {
        "artifact_schema_version": "1.0",
        "artifact_version": artifact_version,
        "role": "champion",
        "dataset": "uci_adult",
        "training_partition": "reference_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "reference_rows": len(dataset.reference),
        "target_counts": {
            str(label): int(count)
            for label, count in target_counts.items()
        },
        "training_config": asdict(config),
        "reference_csv_sha256": _sha256_file(reference_path),
        "model_sha256": _sha256_file(model_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "data_usage": {
            "reference_used_for_training": True,
            "repair_used_for_training": False,
            "release_validation_used_for_training": False,
            "sealed_future_used_for_training": False,
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return ChampionArtifact(
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def load_champion(
    artifact_directory: str | Path,
) -> tuple[XGBClassifier, dict[str, object]]:
    """Load and verify a native XGBoost champion artifact."""

    directory = Path(artifact_directory)
    model_path = directory / "champion.ubj"
    metadata_path = directory / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Champion model missing: {model_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Champion metadata missing: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_hash = metadata.get("model_sha256")
    actual_hash = _sha256_file(model_path)

    if expected_hash != actual_hash:
        raise ValueError("Champion model checksum does not match metadata.")

    model = XGBClassifier()
    model.load_model(model_path)

    return model, metadata