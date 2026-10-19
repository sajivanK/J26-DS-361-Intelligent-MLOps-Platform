"""Dataset splitting and leakage prevention package for Component 3."""

from component3.src.splitting.splitter import (
    DataSplits,
    DataSplitter,
    SplitManifest,
    assert_no_duplicate_feature_leakage,
    assert_no_split_leakage,
    compute_dataset_fingerprint,
    extract_feature_groups,
    verify_generator_training_safety,
)

__all__ = [
    "DataSplits",
    "DataSplitter",
    "SplitManifest",
    "assert_no_duplicate_feature_leakage",
    "assert_no_split_leakage",
    "compute_dataset_fingerprint",
    "extract_feature_groups",
    "verify_generator_training_safety",
]

