"""Load and validate externally supplied C3 repair datasets."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from c4.contracts.models import C3RepairPackage
from c4.data import FEATURE_COLUMNS


@dataclass(frozen=True)
class LoadedRepairPackage:
    package: C3RepairPackage
    dataset_path: Path
    dataset_sha256: str
    frame: pd.DataFrame


def load_repair_package(
    package: C3RepairPackage,
    *,
    data_root: str | Path | None = None,
) -> LoadedRepairPackage:
    """Validate a C3 CSV without training or changing its contents.

    Relative paths are resolved against C4_DATA_ROOT.
    Absolute filesystem paths are also supported.
    """

    # Reject failed packages before opening their data.
    if package.repair_package_status != "APPROVED":
        raise ValueError("C3 repair package is not APPROVED.")

    if package.privacy_status != "PASS":
        raise ValueError("C3 repair package failed its privacy check.")

    if package.n_synthetic_rows <= 0:
        raise ValueError("An approved repair package must contain rows.")

    uri = package.repair_dataset_uri

    if "://" in uri:
        raise ValueError(
            "This loader supports filesystem paths only."
        )

    path = Path(uri).expanduser()

    if not path.is_absolute():
        root = (
            data_root
            if data_root is not None
            else os.environ.get("C4_DATA_ROOT")
        )

        if not root:
            raise ValueError(
                "Set C4_DATA_ROOT or provide data_root "
                "for relative repair paths."
            )

        path = Path(root).expanduser() / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Repair dataset does not exist: {path}"
        )

    # Parse and hash the same bytes for a consistent audit record.
    contents = path.read_bytes()
    checksum = hashlib.sha256(contents).hexdigest()
    frame = pd.read_csv(io.BytesIO(contents))

    expected_columns = [*FEATURE_COLUMNS, "target"]

    if (
        len(frame.columns) != len(expected_columns)
        or set(frame.columns) != set(expected_columns)
    ):
        raise ValueError(
            "Repair CSV must contain exactly the Adult "
            "feature columns and target."
        )

    if frame.empty:
        raise ValueError("Repair dataset is empty.")

    if len(frame) != package.n_synthetic_rows:
        raise ValueError(
            "Repair CSV row count differs from n_synthetic_rows."
        )

    if not all(
        pd.api.types.is_numeric_dtype(frame[column])
        for column in expected_columns
    ):
        raise ValueError("Repair CSV columns must be numeric.")

    if not np.isfinite(
        frame[expected_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Repair CSV contains missing or infinite values."
        )

    if not frame["target"].isin([0, 1]).all():
        raise ValueError("Repair targets must be binary: 0 or 1.")

    # Match the champion's feature order without preprocessing again.
    frame = frame.loc[:, expected_columns].copy()

    return LoadedRepairPackage(
        package=package,
        dataset_path=path,
        dataset_sha256=checksum,
        frame=frame,
    )