"""Offline drift transformations with preserved source-row identity."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule
from c4.scenarios.feature_metadata import load_adult_feature_metadata


@dataclass(frozen=True)
class TransformedRole:
    frame: pd.DataFrame
    true_membership: pd.Series
    provenance: dict[str, object]


def _load_shared_functions():
    # adapter.py -> scenarios -> c4 -> src -> c4 component
    # -> components -> repository root
    repository = Path(__file__).resolve().parents[5]
    path = repository / "shared" / "injection" / "drift_types.py"

    contents = path.read_bytes()
    specification = importlib.util.spec_from_file_location(
        "c4_shared_drift_types",
        path,
    )

    if specification is None or specification.loader is None:
        raise ImportError("Cannot load shared drift functions.")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    if path.read_bytes() != contents:
        raise ValueError("Shared drift code changed during loading.")

    return module, hashlib.sha256(contents).hexdigest()


def transform_role(
    clean_frame: pd.DataFrame,
    *,
    config: ScenarioConfig,
    true_slice: SliceRule,
    role: str,
) -> TransformedRole:
    """Transform repair or release rows without sampling or evaluation.

    Sealed-future transformation is deliberately unsupported here.
    Partition membership must be verified by the calling harness.
    """

    if role not in {"repair", "release_validation"}:
        raise ValueError(
            "Only repair and release_validation roles are supported."
        )

    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

    if clean_frame.columns.tolist() != expected_columns:
        raise ValueError("Scenario source schema is invalid.")

    if clean_frame.empty or not clean_frame.index.is_unique:
        raise ValueError("Source rows must be nonempty with unique IDs.")

    if (
        not pd.api.types.is_integer_dtype(clean_frame.index.dtype)
        or (clean_frame.index < 0).any()
    ):
        raise ValueError("Source row IDs must be nonnegative integers.")

    if not all(
        pd.api.types.is_numeric_dtype(clean_frame[column])
        for column in expected_columns
    ):
        raise ValueError("Scenario source columns must be numeric.")

    if not np.isfinite(
        clean_frame[expected_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Scenario source contains nonfinite values.")

    if not clean_frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Scenario targets must be binary.")

    membership = true_slice.mask(clean_frame).copy()
    membership.name = "true_slice_member"
    transformed = clean_frame.copy(deep=True)
    shared_checksum = None
    feature_metadata = load_adult_feature_metadata()

    if config.mechanism != "none":
        if int(membership.sum()) < 2:
            raise ValueError(
                "Drift injection requires at least two affected rows."
            )

        shared, shared_checksum = _load_shared_functions()
        rng = np.random.default_rng(config.seed)

        affected_features = clean_frame.loc[
            membership, list(FEATURE_COLUMNS)
        ].copy()
        feature_types = feature_metadata.feature_types

        if config.mechanism == "marginal_single":
            feature = config.features[0]
            drifted = shared.inject_marginal_single(
                affected_features,
                feature,
                config.magnitude,
                feature_types[feature],
                rng,
            )

        elif config.mechanism == "marginal_multi":
            drifted = shared.inject_marginal_multi(
                affected_features,
                list(config.features),
                config.magnitude,
                feature_types,
                rng,
            )

        elif config.mechanism == "interaction":
            drifted = shared.inject_interaction_multi_pair(
                affected_features,
                list(config.interaction_pairs),
                config.magnitude,
                rng,
            )

        elif config.mechanism == "combined":
            drifted = shared.inject_combined(
                affected_features,
                list(config.features),
                list(config.interaction_pairs),
                config.magnitude,
                feature_types,
                rng,
            )

        else:
            raise ValueError("Unsupported drift mechanism.")

        if (
            not drifted.index.equals(affected_features.index)
            or drifted.columns.tolist() != list(FEATURE_COLUMNS)
        ):
            raise ValueError("Shared transformation changed row identity.")

        for feature in FEATURE_COLUMNS:
            transformed.loc[membership, feature] = drifted[feature]

    if not np.isfinite(
        transformed[expected_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Transformation produced nonfinite values.")

    if not transformed[TARGET_COLUMN].equals(
        clean_frame[TARGET_COLUMN]
    ):
        raise ValueError("Transformation changed target labels.")

    if not transformed.loc[~membership].equals(
        clean_frame.loc[~membership]
    ):
        raise ValueError("Transformation changed non-affected rows.")

    changed_rows = (
        transformed.loc[:, list(FEATURE_COLUMNS)]
        != clean_frame.loc[:, list(FEATURE_COLUMNS)]
    ).any(axis=1)

    clean_bytes = clean_frame.to_csv(
        index=True, lineterminator="\n"
    ).encode("utf-8")
    transformed_bytes = transformed.to_csv(
        index=True, lineterminator="\n"
    ).encode("utf-8")

    return TransformedRole(
        frame=transformed,
        true_membership=membership,
        provenance={
            "adapter_version": "0.1",
            "role": role,
            "configuration": config.model_dump(mode="json"),
            "configuration_sha256": config.configuration_sha256(),
            "true_slice": true_slice.model_dump(mode="json"),
            "source_rows": len(clean_frame),
            "affected_rows": int(membership.sum()),
            "changed_feature_rows": int(changed_rows.sum()),
            "perturbation_statistics_population": (
                "clean_true_slice_of_current_role"
            ),
            "shared_drift_types_sha256": shared_checksum,
            "clean_frame_sha256": hashlib.sha256(
                clean_bytes
            ).hexdigest(),
            "transformed_frame_sha256": hashlib.sha256(
                transformed_bytes
            ).hexdigest(),
            "feature_metadata_sha256": feature_metadata.source_sha256,
            "feature_types": feature_metadata.feature_types.copy(),
        },
    )