"""Dataset loading utilities for C4."""

from c4.data.adult import (
    DATA_ROOT_ENV,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    AdultDataset,
    load_adult_dataset,
    resolve_data_root,
)
from c4.data.splits import (
    AdultRoleSplits,
    AdultSplitConfig,
    HoldoutAudit,
    split_adult_roles,
)

__all__ = [
    "DATA_ROOT_ENV",
    "EXPECTED_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "AdultDataset",
    "load_adult_dataset",
    "resolve_data_root",
    "AdultRoleSplits",
    "AdultSplitConfig",
    "HoldoutAudit",
    "split_adult_roles",
]
