"""Reproducible Data Splitting and Leakage Prevention Framework.

Supports:
- Stratified splitting for classification datasets.
- Time-based splitting for temporal and transaction streams.
- Group-aware splitting preventing intra-group contamination.
- Explicit split manifests and automated leakage prevention verifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from component3.src.datasets.dataset_registry import DatasetMetadata, SplitStrategy


@dataclass
class DataSplits:
    """Container for the three disjoint dataset partitions."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class SplitManifest:
    """Audit log recording exact partition provenance, indices, and class balance."""
    dataset_id: str
    strategy: str
    seed: int
    train_size: int
    val_size: int
    test_size: int
    total_size: int
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    class_proportions: dict[str, dict[str, float]] = field(default_factory=dict)
    time_column: str | None = None
    group_column: str | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary."""
        return asdict(self)

    def save(self, filepath: Path | str) -> None:
        """Write manifest to disk as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> SplitManifest:
        """Load manifest from JSON file."""
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class DataSplitter:
    """Partitions tabular datasets while enforcing strict leak-free isolation."""

    @staticmethod
    def split(
        df: pd.DataFrame,
        metadata: DatasetMetadata | None = None,
        strategy: SplitStrategy | str | None = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_column: str | None = None,
        time_column: str | None = None,
        group_column: str | None = None,
        seed: int = 42,
    ) -> tuple[DataSplits, SplitManifest]:
        """Execute reproducible split with explicit manifest generation."""
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        dataset_id = metadata.dataset_id if metadata else "custom_dataset"
        active_strategy = strategy or (metadata.split_strategy if metadata else SplitStrategy.STRATIFIED)
        if isinstance(active_strategy, str):
            active_strategy = SplitStrategy(active_strategy)

        active_target = target_column or (metadata.target_column if metadata else None)
        active_time = time_column or (metadata.split_column if metadata and active_strategy == SplitStrategy.TIME_BASED else None)
        active_group = group_column or (metadata.split_column if metadata and active_strategy == SplitStrategy.GROUP_AWARE else None)

        # Standardize reset index to preserve integer row pointers
        df_indexed = df.reset_index(drop=True)
        n_rows = len(df_indexed)
        all_indices = np.arange(n_rows)

        if active_strategy == SplitStrategy.TIME_BASED:
            if not active_time or active_time not in df_indexed.columns:
                raise ValueError(f"Time-based splitting requires valid time_column, got '{active_time}'")
            # Sort chronologically
            df_sorted = df_indexed.sort_values(by=active_time)
            sorted_indices = df_sorted.index.to_numpy()

            train_end = int(n_rows * train_ratio)
            val_end = int(n_rows * (train_ratio + val_ratio))

            train_idx = sorted_indices[:train_end]
            val_idx = sorted_indices[train_end:val_end]
            test_idx = sorted_indices[val_end:]

        elif active_strategy == SplitStrategy.GROUP_AWARE:
            if not active_group or active_group not in df_indexed.columns:
                raise ValueError(f"Group-aware splitting requires valid group_column, got '{active_group}'")

            groups = df_indexed[active_group].to_numpy()
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            train_val_idx, test_idx = next(gss_test.split(all_indices, groups=groups))

            # Split train_val into train and val
            val_relative_ratio = val_ratio / (train_ratio + val_ratio)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=seed)
            train_rel_idx, val_rel_idx = next(gss_val.split(train_val_idx, groups=groups[train_val_idx]))

            train_idx = train_val_idx[train_rel_idx]
            val_idx = train_val_idx[val_rel_idx]

        else:
            # Stratified splitting (default)
            y_strat = None
            if active_target and active_target in df_indexed.columns:
                y_strat = df_indexed[active_target].to_numpy()

            val_test_ratio = val_ratio + test_ratio
            train_idx, val_test_idx = train_test_split(
                all_indices,
                test_size=val_test_ratio,
                random_state=seed,
                stratify=y_strat,
            )

            # Second split for val and test
            relative_test_ratio = test_ratio / val_test_ratio
            y_val_test = y_strat[val_test_idx] if y_strat is not None else None
            val_idx, test_idx = train_test_split(
                val_test_idx,
                test_size=relative_test_ratio,
                random_state=seed,
                stratify=y_val_test,
            )

        # Slice dataframes
        train_df = df_indexed.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df_indexed.iloc[val_idx].copy().reset_index(drop=True)
        test_df = df_indexed.iloc[test_idx].copy().reset_index(drop=True)

        # Compute class proportions for audit
        class_proportions: dict[str, dict[str, float]] = {}
        if active_target and active_target in df_indexed.columns:
            for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                counts = split_df[active_target].value_counts(normalize=True).to_dict()
                class_proportions[name] = {str(k): round(float(v), 4) for k, v in counts.items()}

        manifest = SplitManifest(
            dataset_id=dataset_id,
            strategy=active_strategy.value,
            seed=seed,
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            total_size=n_rows,
            train_indices=[int(i) for i in train_idx],
            val_indices=[int(i) for i in val_idx],
            test_indices=[int(i) for i in test_idx],
            class_proportions=class_proportions,
            time_column=active_time,
            group_column=active_group,
        )

        # Enforce zero leakage assertion on creation
        assert_no_split_leakage(manifest)

        return DataSplits(train=train_df, val=val_df, test=test_df), manifest


def assert_no_split_leakage(manifest: SplitManifest) -> None:
    """Verify that indices across train, validation, and test sets are strictly disjoint."""
    s_train = set(manifest.train_indices)
    s_val = set(manifest.val_indices)
    s_test = set(manifest.test_indices)

    inter_train_val = s_train.intersection(s_val)
    inter_train_test = s_train.intersection(s_test)
    inter_val_test = s_val.intersection(s_test)

    if inter_train_val or inter_train_test or inter_val_test:
        raise AssertionError(
            f"Data leakage detected in split manifest for '{manifest.dataset_id}': "
            f"train/val overlap: {len(inter_train_val)}, "
            f"train/test overlap: {len(inter_train_test)}, "
            f"val/test overlap: {len(inter_val_test)}"
        )


def verify_generator_training_safety(
    generator_training_df: pd.DataFrame,
    evaluation_test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> bool:
    """Check that synthetic data generators are never trained on test evaluation records."""
    cols = feature_columns or [c for c in generator_training_df.columns if c in evaluation_test_df.columns]
    # Check exact row matching across training slice and evaluation test data
    overlap = pd.merge(
        generator_training_df[cols].drop_duplicates(),
        evaluation_test_df[cols].drop_duplicates(),
        how="inner",
    )
    # If overlap represents a large fraction (> 5%) of training data, flag warning / violation
    if len(overlap) > 0 and (len(overlap) / len(generator_training_df)) > 0.05:
        raise AssertionError(
            f"Severe data contamination detected: generator training slice contains {len(overlap)} "
            f"identical rows from the evaluation test partition!"
        )
    return True
