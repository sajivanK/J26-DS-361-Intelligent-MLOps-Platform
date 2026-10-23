"""Structural and target coverage validator for synthetic tabular data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.generators.target_coverage import evaluate_target_coverage
from component3.src.validation.validation_models import (
    ValidationStageResult,
    ValidationStatus,
)


class StructuralValidator:
    """Validator for structural integrity, schema conformity, and target coverage."""

    def __init__(self, min_target_coverage: float = 0.8) -> None:
        if (
            isinstance(min_target_coverage, bool)
            or not isinstance(min_target_coverage, (int, float))
            or not math.isfinite(min_target_coverage)
            or not 0.0 <= min_target_coverage <= 1.0
        ):
            raise ValueError(
                "min_target_coverage must be a finite number between 0.0 and 1.0"
            )
        self.min_target_coverage = min_target_coverage

    def validate(
        self,
        synthetic_data: pd.DataFrame,
        prepared_data: PreparedGeneratorData,
        slice_rule: str | None = None,
    ) -> ValidationStageResult:
        """Validate synthetic data against reference schema and metadata."""
        return validate_structural(
            synthetic_data=synthetic_data,
            prepared_data=prepared_data,
            slice_rule=slice_rule,
            min_target_coverage=self.min_target_coverage,
        )


def validate_structural(
    synthetic_data: pd.DataFrame,
    prepared_data: PreparedGeneratorData,
    slice_rule: str | None = None,
    min_target_coverage: float = 0.8,
) -> ValidationStageResult:
    """Validate structural constraints, metadata consistency, and target coverage.

    Parameters
    ----------
    synthetic_data : pd.DataFrame
        The generated synthetic records to evaluate.
    prepared_data : PreparedGeneratorData
        The metadata and reference data used to prepare the generator.
    slice_rule : str | None, optional
        Target slice expression (e.g. 'race == 2'), by default None.
    min_target_coverage : float, optional
        Minimum acceptable target coverage threshold [0.0, 1.0], by default 0.8.

    Returns
    -------
    ValidationStageResult
        The structural validation stage outcome with status, metrics, and reasons.
    """
    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError("synthetic_data must be a pandas DataFrame")
    if not isinstance(prepared_data, PreparedGeneratorData):
        raise TypeError("prepared_data must be an instance of PreparedGeneratorData")

    # Reject invalid target coverage thresholds.
    if (
        isinstance(min_target_coverage, bool)
        or not isinstance(min_target_coverage, (int, float))
        or not math.isfinite(min_target_coverage)
        or not 0.0 <= min_target_coverage <= 1.0
    ):
        raise ValueError(
            "min_target_coverage must be a finite number between 0.0 and 1.0"
        )

    metrics: dict[str, Any] = {
        "total_rows": int(len(synthetic_data)),
        "total_columns": int(len(synthetic_data.columns)),
    }
    reasons: list[str] = []

    expected_columns = tuple(
        prepared_data.continuous_columns + prepared_data.discrete_columns
    )

    if synthetic_data.empty:
        reasons.append("Synthetic dataset is empty")
    if synthetic_data.columns.has_duplicates:
        reasons.append("Synthetic data contains duplicate columns")
    if prepared_data.training_data.columns.has_duplicates:
        reasons.append("Reference training data contains duplicate columns")
    if len(set(expected_columns)) != len(expected_columns):
        reasons.append("Expected feature names are not unique")
    if (
        set(synthetic_data.columns) != set(expected_columns)
        or len(synthetic_data.columns) != len(expected_columns)
    ):
        reasons.append("Synthetic columns do not match expected columns")

    if (
        synthetic_data.empty
        or synthetic_data.columns.has_duplicates
        or prepared_data.training_data.columns.has_duplicates
        or set(synthetic_data.columns) != set(expected_columns)
        or len(synthetic_data.columns) != len(expected_columns)
    ):
        return ValidationStageResult(
            stage="structural",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    # Validate metadata before checking feature values.
    continuous = tuple(prepared_data.continuous_columns)
    categorical = tuple(prepared_data.discrete_columns)

    # Cross-check the declared feature types with column metadata.
    declared_types = prepared_data.feature_types
    expected_feature_types = {
        **{column: "continuous" for column in continuous},
        **{column: "categorical" for column in categorical},
    }
    if (
        not isinstance(declared_types, dict)
        or declared_types != expected_feature_types
    ):
        reasons.append(
            "Feature_types metadata is inconsistent with "
            "continuous and categorical column definitions"
        )
        return ValidationStageResult(
            stage="structural",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    if (
        len(set(continuous + categorical)) != len(continuous + categorical)
        or set(continuous + categorical) != set(expected_columns)
    ):
        reasons.append("Continuous/categorical metadata is inconsistent")
        return ValidationStageResult(
            stage="structural",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    reference = prepared_data.training_data
    synthetic = synthetic_data

    metrics["missing_values"] = int(synthetic.isna().sum().sum())
    if metrics["missing_values"] > 0:
        reasons.append("Synthetic data contains missing values")

    # Validate continuous columns.
    for column in continuous:
        if not pd.api.types.is_numeric_dtype(synthetic[column]) or pd.api.types.is_bool_dtype(synthetic[column]):
            reasons.append(f"Continuous feature is not numeric: {column}")
            continue
        if not pd.api.types.is_numeric_dtype(reference[column]) or pd.api.types.is_bool_dtype(reference[column]):
            reasons.append(f"Reference continuous feature is not numeric: {column}")
            continue

        non_finite = int(
            np.isinf(
                synthetic[column].to_numpy(dtype=float, na_value=0.0)
            ).sum()
        )
        if non_finite > 0:
            reasons.append(
                f"Continuous feature '{column}' contains non-finite values"
            )

        ref_min = float(reference[column].min())
        ref_max = float(reference[column].max())
        syn_min = float(synthetic[column].min())
        syn_max = float(synthetic[column].max())

        out_of_range = int(
            ((synthetic[column] < ref_min) | (synthetic[column] > ref_max)).sum()
        )
        if "numerical_range_violations" not in metrics:
            metrics["numerical_range_violations"] = {}
        metrics["numerical_range_violations"][column] = {
            "ref_min": ref_min,
            "ref_max": ref_max,
            "syn_min": syn_min,
            "syn_max": syn_max,
            "out_of_range_count": out_of_range,
        }

    # Validate categorical columns.
    metrics["unseen_categories"] = {}
    for column in categorical:
        ref_categories = set(reference[column].dropna().unique())
        syn_categories = set(synthetic[column].dropna().unique())
        unseen = syn_categories - ref_categories
        if unseen:
            sorted_unseen = sorted(list(unseen), key=lambda x: str(x))
            reasons.append(
                f"Categorical feature '{column}' contains unseen categories: {sorted_unseen[:5]}"
            )
            metrics["unseen_categories"][column] = sorted_unseen

    # Evaluate target coverage if a slice rule is provided.
    if slice_rule:
        try:
            cov_result = evaluate_target_coverage(synthetic, slice_rule)
            metrics["target_coverage"] = cov_result.to_dict()
            if cov_result.coverage < min_target_coverage:
                reasons.append(
                    f"Target coverage {cov_result.coverage:.2%} is below threshold {min_target_coverage:.2%}"
                )
        except Exception as exc:
            reasons.append(f"Failed to evaluate target coverage: {exc}")

    status = ValidationStatus.FAIL if reasons else ValidationStatus.PASS
    return ValidationStageResult(
        stage="structural",
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
    )
