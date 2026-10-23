"""Dataset Verification and Integrity Audit Engine for Component 3.

Performs deterministic schema verification, SHA-256 checksum tracking,
leakage-column detection, missingness analysis, and target distribution validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from component3.src.datasets.dataset_registry import DatasetMetadata, DatasetRegistry, get_dataset_registry


@dataclass
class DatasetVerificationReport:
    """Standardized report of dataset integrity and schema compliance."""
    dataset_id: str
    is_valid: bool
    row_count: int
    column_count: int
    sha256_hash: str
    target_column: str
    target_distribution: dict[int | str, float]
    class_imbalance_ratio: float
    duplicate_row_count: int
    duplicate_row_ratio: float
    missing_value_count: int
    missing_value_ratio: float
    missing_features: list[str]
    unexpected_features: list[str]
    leakage_columns_detected: list[str]
    identified_drop_columns: list[str]
    validation_errors: list[str]
    validation_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)


class DatasetVerifier:
    """Validates tabular datasets against registry metadata specifications."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self.registry = registry or get_dataset_registry()

    @staticmethod
    def compute_sha256(filepath: Path | str) -> str:
        """Compute the SHA-256 hash of a dataset file."""
        p = Path(filepath)
        if not p.is_file():
            return "FILE_NOT_FOUND"
        hasher = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def verify_file_integrity(self, filepath: Path | str, dataset_id: str | None = None, expected_hash: str | None = None) -> bool:
        """Verify whether a dataset file matches expected SHA-256 hash."""
        if expected_hash:
            exp = expected_hash
        elif dataset_id:
            meta = self.registry.get(dataset_id)
            exp = meta.sha256_hash
        else:
            return True
        if not exp:
            return True
        computed = self.compute_sha256(filepath)
        return computed.lower() == exp.lower()

    def verify(
        self,
        dataset_id: str | None = None,
        df: pd.DataFrame | None = None,
        filepath: Path | str | None = None,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetVerificationReport:
        """Execute rigorous integrity audit against dataset metadata."""
        meta = metadata or (self.registry.get(dataset_id) if dataset_id else None)
        if meta is None:
            raise ValueError("Must provide either a registered dataset_id or explicit metadata.")
        target_dataset_id = dataset_id or meta.dataset_id
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Resolve Data and Checksum
        resolved_hash = "IN_MEMORY_DATAFRAME"
        if df is None:
            if filepath is None:
                p = meta.get_absolute_path()
            else:
                p = Path(filepath)

            if p is None or not p.is_file():
                return DatasetVerificationReport(
                    dataset_id=dataset_id,
                    is_valid=False,
                    row_count=0,
                    column_count=0,
                    sha256_hash="FILE_NOT_FOUND",
                    target_column=meta.target_column,
                    target_distribution={},
                    class_imbalance_ratio=0.0,
                    duplicate_row_count=0,
                    duplicate_row_ratio=0.0,
                    missing_value_count=0,
                    missing_value_ratio=0.0,
                    missing_features=[],
                    unexpected_features=[],
                    leakage_columns_detected=[],
                    identified_drop_columns=[],
                    validation_errors=[f"Dataset file not found at: {p}"],
                    validation_warnings=[],
                )
            resolved_hash = self.compute_sha256(p)
            df = pd.read_csv(p)

        # 2. Basic Dimensions
        row_count, col_count = df.shape
        if row_count == 0:
            errors.append("Dataset is completely empty (0 rows).")

        # 3. Target Column and Distribution Verification
        target_dist: dict[int | str, float] = {}
        imbalance_ratio = 0.0

        if meta.target_column not in df.columns:
            errors.append(f"Target column '{meta.target_column}' is missing from DataFrame columns.")
        else:
            raw_target = df[meta.target_column]
            if meta.target_mapping:
                mapped_target = raw_target.map(meta.target_mapping)
                unmapped_count = mapped_target.isna().sum()
                if unmapped_count > 0:
                    warnings.append(f"{unmapped_count} target values could not be mapped using target_mapping.")
                val_counts = mapped_target.dropna().value_counts(normalize=True).to_dict()
                target_dist = {int(k): round(float(v), 4) for k, v in val_counts.items()}
            else:
                val_counts = raw_target.value_counts(normalize=True).to_dict()
                target_dist = {str(k): round(float(v), 4) for k, v in val_counts.items()}

            if len(target_dist) == 2:
                # Minority class ratio
                imbalance_ratio = round(min(target_dist.values()), 4)

        # 4. Feature Coverage & Unexpected Columns
        all_expected = set(meta.numerical_features + meta.categorical_features + [meta.target_column] + meta.id_columns + meta.drop_columns + meta.leakage_prone_columns)
        actual_cols = set(df.columns)

        missing_features = [c for c in (meta.numerical_features + meta.categorical_features) if c not in actual_cols and c not in meta.drop_columns]
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")

        unexpected_features = [c for c in actual_cols if c not in all_expected]
        if unexpected_features:
            warnings.append(f"Unexpected features found in dataset: {unexpected_features}")

        # 5. Leakage and Drop Columns Detection
        leakage_detected = [c for c in meta.leakage_prone_columns if c in actual_cols]
        if leakage_detected:
            warnings.append(f"Leakage-prone columns detected in raw data (must be dropped before model training): {leakage_detected}")

        drop_cols_present = [c for c in (meta.drop_columns + meta.id_columns) if c in actual_cols]

        # 6. Duplicate Rows Analysis
        dup_count = int(df.duplicated().sum())
        dup_ratio = round(dup_count / max(row_count, 1), 4)
        if dup_ratio > 0.05:
            warnings.append(f"High duplicate record ratio: {dup_ratio:.2%} ({dup_count} duplicate rows)")

        # 7. Missing Value Analysis
        # Check standard NaNs plus dataset-specific string missing representations
        missing_mask = df.isna()
        for repr_token in meta.missing_representations:
            if isinstance(repr_token, str) and repr_token != "":
                missing_mask = missing_mask | (df == repr_token)

        total_missing = int(missing_mask.sum().sum())
        total_cells = max(row_count * col_count, 1)
        missing_ratio = round(total_missing / total_cells, 4)

        is_valid = len(errors) == 0

        return DatasetVerificationReport(
            dataset_id=target_dataset_id,
            is_valid=is_valid,
            row_count=row_count,
            column_count=col_count,
            sha256_hash=resolved_hash,
            target_column=meta.target_column,
            target_distribution=target_dist,
            class_imbalance_ratio=imbalance_ratio,
            duplicate_row_count=dup_count,
            duplicate_row_ratio=dup_ratio,
            missing_value_count=total_missing,
            missing_value_ratio=missing_ratio,
            missing_features=missing_features,
            unexpected_features=unexpected_features,
            leakage_columns_detected=leakage_detected,
            identified_drop_columns=drop_cols_present,
            validation_errors=errors,
            validation_warnings=warnings,
        )
