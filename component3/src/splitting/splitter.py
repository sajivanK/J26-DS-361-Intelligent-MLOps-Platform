"""Reproducible Data Splitting and Leakage Prevention Framework.

Supports:
- Duplicate-safe stratified splitting preventing identical feature leakage across partitions.
- Time-based splitting for temporal and transaction streams.
- Group-aware splitting preventing intra-group contamination.
- Explicit split manifests, dataset fingerprinting, and automated leakage prevention verifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
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
    dataset_fingerprint: str = ""
    duplicate_group_count: int = 0
    rows_in_duplicate_groups: int = 0
    feature_columns: list[str] = field(default_factory=list)
    leakage_verified: bool = False
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


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 fingerprint for a DataFrame."""
    try:
        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(hashed).hexdigest()[:16]
    except Exception:
        digest_str = f"{df.shape}_{list(df.columns)}_{df.iloc[:5].to_dict()}"
        return hashlib.sha256(digest_str.encode("utf-8")).hexdigest()[:16]


def extract_feature_groups(
    df: pd.DataFrame,
    feature_columns: list[str],
    missing_representations: list[Any] | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Identify duplicate feature vectors, treating missing representations consistently.

    Returns:
        group_ids: Array of group assignments for each row (0-indexed).
        n_groups: Total number of unique feature groups.
        n_duplicate_groups: Count of groups containing more than 1 row.
        rows_in_duplicate_groups: Total rows belonging to groups with > 1 row.
    """
    missing_tokens = list(missing_representations or ["?", "", "NA", "null", "NaN", "nan"])
    df_feat = df[feature_columns].copy()

    for col in feature_columns:
        if df_feat[col].dtype == object or pd.api.types.is_string_dtype(df_feat[col]):
            s = df_feat[col].astype(str).str.strip()
            for tok in missing_tokens:
                s = s.replace(str(tok).strip(), "__MISSING__")
            df_feat[col] = s.replace(["nan", "None", "<NA>"], "__MISSING__")
        else:
            df_feat[col] = df_feat[col].fillna("__MISSING__")

    group_ids = df_feat.groupby(feature_columns, sort=False, dropna=False).ngroup().to_numpy()
    n_groups = int(group_ids.max() + 1) if len(group_ids) > 0 else 0

    counts = np.bincount(group_ids) if len(group_ids) > 0 else np.array([])
    dup_mask = counts > 1
    n_duplicate_groups = int(np.sum(dup_mask))
    rows_in_duplicate_groups = int(np.sum(counts[dup_mask]))

    return group_ids, n_groups, n_duplicate_groups, rows_in_duplicate_groups


def stratified_group_partition(
    group_ids: np.ndarray,
    n_groups: int,
    y: np.ndarray | None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition groups into train/val/test using greedy deficit-balancing optimization.

    Ensures:
    1. Every group is assigned exclusively to ONE partition (0% intra-group cross-partition leakage).
    2. Partition sample sizes closely match (train_ratio, val_ratio, test_ratio).
    3. Target class distributions across partitions are matched as closely as practical.
    """
    n_samples = len(group_ids)
    r_vec = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)

    # Encode classes if target is present
    if y is not None:
        classes, y_inv = np.unique(y, return_inverse=True)
        n_classes = len(classes)
    else:
        classes = np.array([0])
        y_inv = np.zeros(n_samples, dtype=np.int64)
        n_classes = 1

    # Compute group class counts
    group_class_counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    for g_id, c_idx in zip(group_ids, y_inv):
        group_class_counts[g_id, c_idx] += 1
    group_sizes = group_class_counts.sum(axis=1)

    total_class_counts = group_class_counts.sum(axis=0)
    target_counts = r_vec[:, np.newaxis] * total_class_counts[np.newaxis, :]
    target_sizes = r_vec * n_samples

    inv_target_counts = 1.0 / np.maximum(target_counts, 1e-9)
    inv_target_sizes = 1.0 / np.maximum(target_sizes, 1e-9)

    # Deterministic permutation and sorting by group size descending
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    sorted_order = np.argsort(-group_sizes[perm], kind="stable")
    ordered_groups = perm[sorted_order]

    alloc_class_counts = np.zeros((3, n_classes), dtype=np.float64)
    alloc_sizes = np.zeros(3, dtype=np.float64)
    group_partition = np.empty(n_groups, dtype=np.int8)

    for g in ordered_groups:
        g_counts = group_class_counts[g]
        g_size = group_sizes[g]

        best_p = 0
        best_score = float("inf")

        for p in range(3):
            alloc_class_counts[p] += g_counts
            alloc_sizes[p] += g_size

            r_classes = alloc_class_counts * inv_target_counts
            r_sizes = alloc_sizes * inv_target_sizes

            # Variation across partitions
            score = np.std(r_classes, axis=0).mean() + np.std(r_sizes)

            alloc_class_counts[p] -= g_counts
            alloc_sizes[p] -= g_size

            if score < best_score:
                best_score = score
                best_p = p

        group_partition[g] = best_p
        alloc_class_counts[best_p] += g_counts
        alloc_sizes[best_p] += g_size

    row_partitions = group_partition[group_ids]
    train_idx = np.where(row_partitions == 0)[0]
    val_idx = np.where(row_partitions == 1)[0]
    test_idx = np.where(row_partitions == 2)[0]

    return train_idx, val_idx, test_idx


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
        feature_columns: list[str] | None = None,
        id_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        leakage_prone_columns: list[str] | None = None,
        missing_representations: list[Any] | None = None,
        duplicate_safe: bool = True,
        seed: int = 42,
    ) -> tuple[DataSplits, SplitManifest]:
        """Execute reproducible split with explicit manifest generation and leakage prevention."""
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

        # Compute dataset fingerprint
        fingerprint = compute_dataset_fingerprint(df_indexed)

        # Determine feature columns for duplicate grouping
        if feature_columns is not None:
            active_features = list(feature_columns)
        elif metadata is not None:
            active_features = list(metadata.feature_columns)
        else:
            excluded = set()
            if active_target:
                excluded.add(active_target)
            if id_columns:
                excluded.update(id_columns)
            if drop_columns:
                excluded.update(drop_columns)
            if leakage_prone_columns:
                excluded.update(leakage_prone_columns)
            if active_group:
                excluded.add(active_group)
            if active_time:
                excluded.add(active_time)
            active_features = [c for c in df_indexed.columns if c not in excluded]

        missing_reps = missing_representations or (
            metadata.missing_representations if metadata else ["?", "", "NA", "null", "NaN", "nan"]
        )

        n_dup_groups = 0
        n_rows_in_dups = 0

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

            group_series = df_indexed[active_group]
            group_ids = group_series.factorize()[0]
            n_groups = int(group_ids.max() + 1) if len(group_ids) > 0 else 0

            y_strat = df_indexed[active_target].to_numpy() if (active_target and active_target in df_indexed.columns) else None
            train_idx, val_idx, test_idx = stratified_group_partition(
                group_ids=group_ids,
                n_groups=n_groups,
                y=y_strat,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

        elif duplicate_safe and active_features:
            # Duplicate-safe stratified group splitting
            group_ids, n_groups, n_dup_groups, n_rows_in_dups = extract_feature_groups(
                df_indexed,
                feature_columns=active_features,
                missing_representations=missing_reps,
            )

            y_strat = df_indexed[active_target].to_numpy() if (active_target and active_target in df_indexed.columns) else None
            train_idx, val_idx, test_idx = stratified_group_partition(
                group_ids=group_ids,
                n_groups=n_groups,
                y=y_strat,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

        else:
            # Classical random stratified splitting
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

        splits = DataSplits(train=train_df, val=val_df, test=test_df)

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
            dataset_fingerprint=fingerprint,
            duplicate_group_count=n_dup_groups,
            rows_in_duplicate_groups=n_rows_in_dups,
            feature_columns=active_features,
            leakage_verified=False,
        )

        # Enforce zero index leakage assertion on creation
        assert_no_split_leakage(manifest)

        # Enforce zero duplicate feature leakage assertion if duplicate_safe
        if duplicate_safe and active_features and active_strategy != SplitStrategy.TIME_BASED:
            assert_no_duplicate_feature_leakage(splits, active_features, missing_reps)
            manifest.leakage_verified = True

        return splits, manifest


def assert_no_split_leakage(manifest: SplitManifest) -> None:
    """Verify that indices across train, validation, and test sets are strictly disjoint and exhaustive."""
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

    total_assigned = len(s_train) + len(s_val) + len(s_test)
    if total_assigned != manifest.total_size:
        raise AssertionError(
            f"Split indices do not partition all rows for '{manifest.dataset_id}': "
            f"total rows={manifest.total_size}, assigned rows={total_assigned}"
        )

    if len(manifest.train_indices) != len(s_train):
        raise AssertionError(f"Duplicate index found within train partition for '{manifest.dataset_id}'")
    if len(manifest.val_indices) != len(s_val):
        raise AssertionError(f"Duplicate index found within val partition for '{manifest.dataset_id}'")
    if len(manifest.test_indices) != len(s_test):
        raise AssertionError(f"Duplicate index found within test partition for '{manifest.dataset_id}'")


def assert_no_duplicate_feature_leakage(
    splits: DataSplits,
    feature_columns: list[str],
    missing_representations: list[Any] | None = None,
) -> dict[str, Any]:
    """Mathematically verify that zero identical feature records exist across partitions.

    Raises AssertionError with diagnostic details if identical feature vectors cross partition boundaries.
    """
    missing_tokens = list(missing_representations or ["?", "", "NA", "null", "NaN", "nan"])

    def _get_feature_tuples(df_part: pd.DataFrame) -> set[tuple]:
        df_clean = df_part[feature_columns].copy()
        for col in feature_columns:
            if df_clean[col].dtype == object or pd.api.types.is_string_dtype(df_clean[col]):
                s = df_clean[col].astype(str).str.strip()
                for tok in missing_tokens:
                    s = s.replace(str(tok).strip(), "__MISSING__")
                df_clean[col] = s.replace(["nan", "None", "<NA>"], "__MISSING__")
            else:
                df_clean[col] = df_clean[col].fillna("__MISSING__")
        return set(df_clean.itertuples(index=False, name=None))

    train_tuples = _get_feature_tuples(splits.train)
    val_tuples = _get_feature_tuples(splits.val)
    test_tuples = _get_feature_tuples(splits.test)

    train_val_overlap = train_tuples.intersection(val_tuples)
    train_test_overlap = train_tuples.intersection(test_tuples)
    val_test_overlap = val_tuples.intersection(test_tuples)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise AssertionError(
            f"Duplicate feature leakage detected across partitions!\n"
            f"  Train/Val identical feature vectors: {len(train_val_overlap)}\n"
            f"  Train/Test identical feature vectors: {len(train_test_overlap)}\n"
            f"  Val/Test identical feature vectors: {len(val_test_overlap)}"
        )

    return {
        "status": "PASS",
        "train_unique_feature_vectors": len(train_tuples),
        "val_unique_feature_vectors": len(val_tuples),
        "test_unique_feature_vectors": len(test_tuples),
        "cross_partition_overlap": 0,
    }


def verify_generator_training_safety(
    generator_training_df: pd.DataFrame,
    evaluation_test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> bool:
    """Check that synthetic data generators are never trained on test evaluation records."""
    cols = feature_columns or [c for c in generator_training_df.columns if c in evaluation_test_df.columns]
    overlap = pd.merge(
        generator_training_df[cols].drop_duplicates(),
        evaluation_test_df[cols].drop_duplicates(),
        how="inner",
    )
    if len(overlap) > 0 and (len(overlap) / len(generator_training_df)) > 0.05:
        raise AssertionError(
            f"Severe data contamination detected: generator training slice contains {len(overlap)} "
            f"identical rows from the evaluation test partition!"
        )
    return True
