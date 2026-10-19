"""Dataset splitting and leakage prevention package for Component 3."""

from component3.src.splitting.splitter import (
    DataSplits,
    DataSplitter,
    SplitManifest,
    assert_no_split_leakage,
    verify_generator_training_safety,
)

__all__ = [
    "DataSplits",
    "DataSplitter",
    "SplitManifest",
    "assert_no_split_leakage",
    "verify_generator_training_safety",
]
