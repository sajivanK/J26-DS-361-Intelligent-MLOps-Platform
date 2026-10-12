"""Authoritative Adult feature types from the shared metadata JSON."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from c4.data import FEATURE_COLUMNS
from c4.scenarios.config import CONTINUOUS_FEATURES


@dataclass(frozen=True)
class AdultFeatureMetadata:
    feature_types: dict[str, str]
    source_path: Path
    source_sha256: str


def load_adult_feature_metadata(
    path: str | Path | None = None,
) -> AdultFeatureMetadata:
    """Load shared feature types and check C4 schema compatibility."""

    if path is None:
        repository = Path(__file__).resolve().parents[5]
        source_path = (
            repository / "shared" / "uci_adult"
            / "feature_importance_adult.json"
        )
    else:
        source_path = Path(path)

    contents = source_path.read_bytes()
    document = json.loads(contents)
    feature_types = document.get("feature_types")

    if not isinstance(feature_types, dict):
        raise ValueError("Shared metadata must contain feature_types.")

    if set(feature_types) != set(FEATURE_COLUMNS):
        raise ValueError("Shared metadata feature schema differs from C4.")

    if any(
        value not in {"continuous", "categorical"}
        for value in feature_types.values()
    ):
        raise ValueError("Shared metadata contains an unsupported feature type.")

    continuous = {
        feature
        for feature, kind in feature_types.items()
        if kind == "continuous"
    }

    # C4's slice validators also depend on feature types.
    # Reject disagreements instead of silently changing their meaning.
    if continuous != set(CONTINUOUS_FEATURES):
        raise ValueError(
            "Shared feature types conflict with C4 slice validation. "
            "Review and align the schema before continuing."
        )

    return AdultFeatureMetadata(
        feature_types={
            feature: feature_types[feature]
            for feature in FEATURE_COLUMNS
        },
        source_path=source_path.resolve(),
        source_sha256=hashlib.sha256(contents).hexdigest(),
    )