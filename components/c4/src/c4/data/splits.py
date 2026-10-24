"""Leakage-safe role partitioning for the Adult holdout dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from c4.data.adult import AdultDataset, TARGET_COLUMN


@dataclass(frozen=True)
class AdultSplitConfig:
    """Frozen configuration for Adult role partitioning."""

    repair_fraction: float = 0.40
    release_validation_fraction: float = 0.30
    sealed_future_fraction: float = 0.30
    random_state: int = 42

    def __post_init__(self) -> None:
        fractions = (
            self.repair_fraction,
            self.release_validation_fraction,
            self.sealed_future_fraction,
        )

        if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
            raise ValueError("Every split fraction must be between 0 and 1.")

        if not math.isclose(
            sum(fractions),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Split fractions must sum to 1.0.")


@dataclass(frozen=True)
class HoldoutAudit:
    """Counts produced while creating the eligible holdout."""

    original_test_rows: int
    test_duplicate_extras: int
    cross_reference_rows: int
    post_reference_duplicate_extras: int
    eligible_rows: int


@dataclass(frozen=True)
class AdultRoleSplits:
    """Non-overlapping datasets assigned to C4 research roles."""

    reference: pd.DataFrame
    repair: pd.DataFrame
    release_validation: pd.DataFrame
    sealed_future: pd.DataFrame
    audit: HoldoutAudit


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame, index=False)


def _prepare_eligible_holdout(
    reference: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, HoldoutAudit]:
    """Exclude repeated processed representations conservatively."""

    if not test.index.is_unique:
        raise ValueError("Test dataframe index must be unique.")

    reference_hashes = set(_row_hashes(reference).tolist())
    test_hashes = _row_hashes(test)

    cross_reference_mask = test_hashes.isin(reference_hashes)
    without_reference_overlap = test.loc[~cross_reference_mask].copy()

    duplicate_mask = without_reference_overlap.duplicated(keep="first")
    eligible = without_reference_overlap.loc[~duplicate_mask].copy()

    audit = HoldoutAudit(
        original_test_rows=len(test),
        test_duplicate_extras=int(test.duplicated().sum()),
        cross_reference_rows=int(cross_reference_mask.sum()),
        post_reference_duplicate_extras=int(duplicate_mask.sum()),
        eligible_rows=len(eligible),
    )

    return eligible, audit


def _assert_partition_integrity(
    reference: pd.DataFrame,
    eligible: pd.DataFrame,
    repair: pd.DataFrame,
    release_validation: pd.DataFrame,
    sealed_future: pd.DataFrame,
) -> None:
    repair_indices = set(repair.index)
    release_indices = set(release_validation.index)
    future_indices = set(sealed_future.index)

    if repair_indices & release_indices:
        raise RuntimeError("Repair and release-validation rows overlap.")

    if repair_indices & future_indices:
        raise RuntimeError("Repair and sealed-future rows overlap.")

    if release_indices & future_indices:
        raise RuntimeError(
            "Release-validation and sealed-future rows overlap."
        )

    assigned_indices = (
        repair_indices | release_indices | future_indices
    )

    if assigned_indices != set(eligible.index):
        raise RuntimeError(
            "Role partitions do not exactly cover the eligible holdout."
        )

    reference_hashes = set(_row_hashes(reference).tolist())
    role_hash_sets = [
        set(_row_hashes(frame).tolist())
        for frame in (repair, release_validation, sealed_future)
    ]

    if any(reference_hashes & hashes for hashes in role_hash_sets):
        raise RuntimeError(
            "A holdout representation overlaps the reference set."
        )

    if role_hash_sets[0] & role_hash_sets[1]:
        raise RuntimeError(
            "Repair and release-validation representations overlap."
        )

    if role_hash_sets[0] & role_hash_sets[2]:
        raise RuntimeError(
            "Repair and sealed-future representations overlap."
        )

    if role_hash_sets[1] & role_hash_sets[2]:
        raise RuntimeError(
            "Release-validation and sealed-future representations overlap."
        )


def split_adult_roles(
    dataset: AdultDataset,
    config: AdultSplitConfig | None = None,
) -> AdultRoleSplits:
    """Create deterministic, stratified, leakage-safe Adult roles."""

    config = config or AdultSplitConfig()

    eligible, audit = _prepare_eligible_holdout(
        dataset.reference,
        dataset.test,
    )

    repair, remaining = train_test_split(
        eligible,
        train_size=config.repair_fraction,
        stratify=eligible[TARGET_COLUMN],
        random_state=config.random_state,
    )

    release_share_of_remaining = (
        config.release_validation_fraction
        / (
            config.release_validation_fraction
            + config.sealed_future_fraction
        )
    )

    release_validation, sealed_future = train_test_split(
        remaining,
        train_size=release_share_of_remaining,
        stratify=remaining[TARGET_COLUMN],
        random_state=config.random_state + 1,
    )

    repair = repair.sort_index().copy()
    release_validation = release_validation.sort_index().copy()
    sealed_future = sealed_future.sort_index().copy()

    _assert_partition_integrity(
        dataset.reference,
        eligible,
        repair,
        release_validation,
        sealed_future,
    )

    return AdultRoleSplits(
        reference=dataset.reference.copy(),
        repair=repair,
        release_validation=release_validation,
        sealed_future=sealed_future,
        audit=audit,
    )