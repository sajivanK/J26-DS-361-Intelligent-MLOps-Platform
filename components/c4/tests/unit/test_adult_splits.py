"""Tests for leakage-safe Adult role partitioning."""

import pandas as pd
import pytest

from c4.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    AdultDataset,
    AdultSplitConfig,
    split_adult_roles,
)


def make_frame(size: int, offset: int) -> pd.DataFrame:
    data = {}

    for column_number, column in enumerate(FEATURE_COLUMNS):
        data[column] = [
            float(offset + row_number) + column_number / 100
            for row_number in range(size)
        ]

    data[TARGET_COLUMN] = [
        row_number % 2
        for row_number in range(size)
    ]

    return pd.DataFrame(data)


def make_dataset_with_repeated_representations() -> AdultDataset:
    reference = make_frame(size=20, offset=0)
    unique_test = make_frame(size=100, offset=1000)

    test = pd.concat(
        [
            unique_test,
            reference.iloc[[0, 1]],
            unique_test.iloc[[0]],
        ],
        ignore_index=True,
    )

    return AdultDataset(reference=reference, test=test)


def frame_hashes(frame: pd.DataFrame) -> set[int]:
    return set(
        pd.util.hash_pandas_object(
            frame,
            index=False,
        ).tolist()
    )


def test_default_split_sizes_and_audit():
    dataset = make_dataset_with_repeated_representations()

    splits = split_adult_roles(dataset)

    assert splits.audit.original_test_rows == 103
    assert splits.audit.test_duplicate_extras == 1
    assert splits.audit.cross_reference_rows == 2
    assert splits.audit.post_reference_duplicate_extras == 1
    assert splits.audit.eligible_rows == 100

    assert len(splits.repair) == 40
    assert len(splits.release_validation) == 30
    assert len(splits.sealed_future) == 30


def test_role_indices_are_disjoint_and_complete():
    dataset = make_dataset_with_repeated_representations()

    splits = split_adult_roles(dataset)

    repair_indices = set(splits.repair.index)
    release_indices = set(splits.release_validation.index)
    future_indices = set(splits.sealed_future.index)

    assert repair_indices.isdisjoint(release_indices)
    assert repair_indices.isdisjoint(future_indices)
    assert release_indices.isdisjoint(future_indices)

    assert len(
        repair_indices | release_indices | future_indices
    ) == 100


def test_processed_representations_do_not_overlap():
    dataset = make_dataset_with_repeated_representations()

    splits = split_adult_roles(dataset)

    reference_hashes = frame_hashes(splits.reference)
    repair_hashes = frame_hashes(splits.repair)
    release_hashes = frame_hashes(splits.release_validation)
    future_hashes = frame_hashes(splits.sealed_future)

    assert reference_hashes.isdisjoint(repair_hashes)
    assert reference_hashes.isdisjoint(release_hashes)
    assert reference_hashes.isdisjoint(future_hashes)

    assert repair_hashes.isdisjoint(release_hashes)
    assert repair_hashes.isdisjoint(future_hashes)
    assert release_hashes.isdisjoint(future_hashes)


def test_split_is_reproducible_for_same_seed():
    dataset = make_dataset_with_repeated_representations()
    config = AdultSplitConfig(random_state=42)

    first = split_adult_roles(dataset, config)
    second = split_adult_roles(dataset, config)

    assert first.repair.equals(second.repair)
    assert first.release_validation.equals(
        second.release_validation
    )
    assert first.sealed_future.equals(second.sealed_future)


def test_invalid_fraction_total_is_rejected():
    with pytest.raises(ValueError, match="sum to 1.0"):
        AdultSplitConfig(
            repair_fraction=0.50,
            release_validation_fraction=0.30,
            sealed_future_fraction=0.30,
        )