"""Unit tests for Component 3 Data Splitter and Leakage Prevention."""

import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import SplitStrategy
from component3.src.splitting.splitter import (
    DataSplitter,
    SplitManifest,
    assert_no_split_leakage,
    verify_generator_training_safety,
)


@pytest.fixture
def synthetic_split_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "feat_num": rng.normal(size=n),
        "timestamp": range(1000, 1000 + n),
        "group_id": np.repeat(range(20), 5),  # 20 groups of 5
        "target": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
    })


def test_stratified_split(synthetic_split_df):
    splits, manifest = DataSplitter.split(
        synthetic_split_df,
        strategy=SplitStrategy.STRATIFIED,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_column="target",
        seed=42,
    )

    assert len(splits.train) == 70
    assert len(splits.val) == 15
    assert len(splits.test) == 15
    assert manifest.strategy == "STRATIFIED"

    # Zero index leakage check
    assert_no_split_leakage(manifest)

    # Class balance consistency across train and test
    train_prop = float(manifest.class_proportions["train"]["1"])
    test_prop = float(manifest.class_proportions["test"]["1"])
    assert abs(train_prop - test_prop) < 0.10


def test_time_based_split(synthetic_split_df):
    splits, manifest = DataSplitter.split(
        synthetic_split_df,
        strategy=SplitStrategy.TIME_BASED,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        time_column="timestamp",
        seed=42,
    )

    assert manifest.strategy == "TIME_BASED"
    # Verify temporal ordering: max train time < min val time <= max val time < min test time
    assert splits.train["timestamp"].max() < splits.val["timestamp"].min()
    assert splits.val["timestamp"].max() < splits.test["timestamp"].min()
    assert_no_split_leakage(manifest)


def test_group_aware_split(synthetic_split_df):
    splits, manifest = DataSplitter.split(
        synthetic_split_df,
        strategy=SplitStrategy.GROUP_AWARE,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column="group_id",
        seed=42,
    )

    assert manifest.strategy == "GROUP_AWARE"
    # Ensure no groups overlap across partitions
    train_groups = set(splits.train["group_id"].unique())
    val_groups = set(splits.val["group_id"].unique())
    test_groups = set(splits.test["group_id"].unique())

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert_no_split_leakage(manifest)


def test_leakage_assertion_raises_on_contamination():
    # Corrupt manifest with deliberate overlap
    corrupt_manifest = SplitManifest(
        dataset_id="leak_test",
        strategy="STRATIFIED",
        seed=42,
        train_size=3,
        val_size=2,
        test_size=2,
        total_size=5,
        train_indices=[0, 1, 2],
        val_indices=[2, 3],  # Index 2 leaked into validation!
        test_indices=[4],
    )

    with pytest.raises(AssertionError) as exc_info:
        assert_no_split_leakage(corrupt_manifest)
    assert "Data leakage detected" in str(exc_info.value)
    assert "train/val overlap: 1" in str(exc_info.value)


def test_generator_safety_check(synthetic_split_df):
    train_slice = synthetic_split_df.iloc[:20].copy()
    test_slice = synthetic_split_df.iloc[80:].copy()
    # Disjoint slices pass safety check
    assert verify_generator_training_safety(train_slice, test_slice) is True

    # Artificially contaminate generator training slice with test records
    contaminated_slice = pd.concat([train_slice, test_slice.iloc[:10]], ignore_index=True)
    with pytest.raises(AssertionError) as exc_info:
        verify_generator_training_safety(contaminated_slice, test_slice)
    assert "data contamination detected" in str(exc_info.value)


def test_manifest_serialization(synthetic_split_df, tmp_path):
    _, manifest = DataSplitter.split(
        synthetic_split_df,
        strategy=SplitStrategy.STRATIFIED,
        target_column="target",
        seed=42,
    )

    save_path = tmp_path / "split_manifest.json"
    manifest.save(save_path)
    assert save_path.is_file()

    loaded_manifest = SplitManifest.load(save_path)
    assert loaded_manifest.dataset_id == manifest.dataset_id
    assert loaded_manifest.train_size == manifest.train_size
    assert loaded_manifest.train_indices == manifest.train_indices
