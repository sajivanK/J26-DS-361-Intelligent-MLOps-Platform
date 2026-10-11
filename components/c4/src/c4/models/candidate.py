"""Candidate retraining using externally supplied C3 repair data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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