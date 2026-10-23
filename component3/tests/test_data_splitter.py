"""Unit tests for Component 3 Data Splitter and Leakage Prevention."""

import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import SplitStrategy
from component3.src.splitting.splitter import (
    DataSplits,
    DataSplitter,
    SplitManifest,
    assert_no_duplicate_feature_leakage,
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
    assert loaded_manifest.dataset_fingerprint == manifest.dataset_fingerprint
    assert loaded_manifest.leakage_verified == manifest.leakage_verified


def test_duplicate_feature_records_isolated():
    """Verify that duplicate feature records are kept strictly in the same partition."""
    # 20 unique feature patterns, each repeated 5 times = 100 rows
    feat_vals = np.repeat(np.arange(20), 5)
    targets = np.repeat([0, 1], 50)
    df = pd.DataFrame({
        "feat_a": feat_vals,
        "feat_b": [f"cat_{v}" for v in feat_vals],
        "target": targets,
    })

    splits, manifest = DataSplitter.split(
        df,
        target_column="target",
        feature_columns=["feat_a", "feat_b"],
        seed=42,
    )

    assert manifest.duplicate_group_count == 20
    assert manifest.rows_in_duplicate_groups == 100
    assert manifest.leakage_verified is True

    # Assert no duplicate feature vectors cross partition boundaries
    report = assert_no_duplicate_feature_leakage(splits, ["feat_a", "feat_b"])
    assert report["status"] == "PASS"
    assert report["cross_partition_overlap"] == 0

    # Ensure all original rows are accounted for without any drops
    assert len(splits.train) + len(splits.val) + len(splits.test) == 100


def test_conflicting_target_labels_preserved_in_same_partition():
    """Identical feature records with conflicting labels must remain in the same partition."""
    # Row 0 and Row 1 have identical features ('X', 1.0) but opposite labels (0 and 1)
    # Row 2 and Row 3 have identical features ('Y', 2.0) but opposite labels (0 and 1)
    # Plus 96 distinct background rows to form 100 rows
    df = pd.DataFrame({
        "cat": ["X", "X", "Y", "Y"] + [f"C_{i}" for i in range(96)],
        "num": [1.0, 1.0, 2.0, 2.0] + [float(i + 10) for i in range(96)],
        "target": [0, 1, 0, 1] + [i % 2 for i in range(96)],
    })

    splits, manifest = DataSplitter.split(
        df,
        target_column="target",
        feature_columns=["cat", "num"],
        seed=42,
    )

    assert_no_split_leakage(manifest)
    assert_no_duplicate_feature_leakage(splits, ["cat", "num"])

    # Locate which partition holds ('X', 1.0)
    in_train = len(splits.train[(splits.train["cat"] == "X") & (splits.train["num"] == 1.0)])
    in_val = len(splits.val[(splits.val["cat"] == "X") & (splits.val["num"] == 1.0)])
    in_test = len(splits.test[(splits.test["cat"] == "X") & (splits.test["num"] == 1.0)])

    # All 2 conflicting rows must be in exactly ONE partition (2 in one, 0 in others)
    counts = [in_train, in_val, in_test]
    assert 2 in counts
    assert counts.count(0) == 2

    # Verify both conflicting target labels are preserved in that partition
    target_part = splits.train if in_train == 2 else (splits.val if in_val == 2 else splits.test)
    x_rows = target_part[(target_part["cat"] == "X") & (target_part["num"] == 1.0)]
    assert set(x_rows["target"].tolist()) == {0, 1}


def test_missing_values_consistent_grouping():
    """Missing value representations (?, NA, np.nan) must group identically and not leak."""
    df = pd.DataFrame({
        "cat": ["?", "NA", np.nan, "None", "valid", "valid"] + [f"cat_{i}" for i in range(94)],
        "num": [np.nan, np.nan, np.nan, np.nan, 5.0, 5.0] + [float(i) for i in range(94)],
        "target": [0, 1, 0, 1, 0, 1] + [i % 2 for i in range(94)],
    })

    splits, manifest = DataSplitter.split(
        df,
        target_column="target",
        feature_columns=["cat", "num"],
        missing_representations=["?", "NA", "None"],
        seed=42,
    )

    assert_no_split_leakage(manifest)
    assert_no_duplicate_feature_leakage(splits, ["cat", "num"], missing_representations=["?", "NA", "None"])

    # All 4 missing-representation rows belong to group ('__MISSING__', '__MISSING__')
    # and must be isolated in the exact same partition
    for part in [splits.train, splits.val, splits.test]:
        missing_rows = part[part["num"].isna()]
        if len(missing_rows) > 0:
            assert len(missing_rows) == 4


def test_duplicate_safe_split_reproducibility():
    """Identical seed must yield identical indices; different seed produces valid alternative."""
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "f1": rng.choice(np.arange(30), size=100),
        "f2": rng.choice(["A", "B", "C"], size=100),
        "target": rng.choice([0, 1], size=100),
    })

    _, m1 = DataSplitter.split(df, target_column="target", feature_columns=["f1", "f2"], seed=42)
    _, m2 = DataSplitter.split(df, target_column="target", feature_columns=["f1", "f2"], seed=42)
    _, m3 = DataSplitter.split(df, target_column="target", feature_columns=["f1", "f2"], seed=99)

    assert m1.train_indices == m2.train_indices
    assert m1.val_indices == m2.val_indices
    assert m1.test_indices == m2.test_indices
    assert m1.dataset_fingerprint == m2.dataset_fingerprint
    assert m1.train_indices != m3.train_indices


def test_class_imbalance_preserved_across_partitions():
    """Severe class imbalance (90/10) must be preserved across train, val, and test splits."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "f1": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n),
        "target": rng.choice([0, 1], size=n, p=[0.90, 0.10]),
    })

    splits, manifest = DataSplitter.split(
        df,
        target_column="target",
        feature_columns=["f1"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    overall_pos = (df["target"] == 1).mean()
    train_pos = (splits.train["target"] == 1).mean()
    val_pos = (splits.val["target"] == 1).mean()
    test_pos = (splits.test["target"] == 1).mean()

    # Preserved within reasonable group-quantization tolerance (< 5%)
    assert abs(train_pos - overall_pos) < 0.05
    assert abs(val_pos - overall_pos) < 0.05
    assert abs(test_pos - overall_pos) < 0.05


def test_duplicate_feature_leakage_assertion_raises_on_contamination():
    """Manually inserting a row from train into test must trigger assert_no_duplicate_feature_leakage."""
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": ["A", "B", "C", "D"],
        "target": [0, 1, 0, 1],
    })
    splits, _ = DataSplitter.split(df, target_column="target", feature_columns=["f1", "f2"], seed=42)

    # Intentionally contaminate test with a row from train
    contaminated_test = pd.concat([splits.test, splits.train.iloc[[0]]], ignore_index=True)
    bad_splits = DataSplits(train=splits.train, val=splits.val, test=contaminated_test)

    with pytest.raises(AssertionError) as exc_info:
        assert_no_duplicate_feature_leakage(bad_splits, ["f1", "f2"])
    assert "Duplicate feature leakage detected" in str(exc_info.value)


def test_real_dataset_duplicate_safe_preprocessing_compatibility():
    """Verify end-to-end compatibility between duplicate-safe splits and TabularDataPreprocessor."""
    from component3.src.datasets.dataset_acquisition import DatasetAcquisitionManager
    from component3.src.datasets.dataset_registry import get_dataset_registry
    from component3.src.preprocessing.tabular_preprocessor import get_preprocessor_for_dataset

    acq = DatasetAcquisitionManager()
    reg = get_dataset_registry()

    for did in ["uci_adult", "telco_churn"]:
        raw_path, _ = acq.acquire_dataset(did)
        meta = reg.get(did)
        df = pd.read_csv(raw_path)

        splits, manifest = DataSplitter.split(
            df=df,
            metadata=meta,
            seed=42,
        )

        assert manifest.leakage_verified is True
        assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)

        # Fit preprocessor strictly on train split
        prep = get_preprocessor_for_dataset(did)
        train_bundle = prep.fit_transform(splits.train)
        val_bundle = prep.transform(splits.val)
        test_bundle = prep.transform(splits.test)

        assert train_bundle.X.shape[0] == len(splits.train)
        assert val_bundle.X.shape[0] == len(splits.val)
        assert test_bundle.X.shape[0] == len(splits.test)
        assert train_bundle.X.shape[1] == val_bundle.X.shape[1] == test_bundle.X.shape[1]

