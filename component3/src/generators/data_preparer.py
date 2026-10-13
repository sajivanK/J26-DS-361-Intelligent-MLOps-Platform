"""Generator training data and metadata preparation for Component 3."""

from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

from component3.src.readiness.readiness_models import ReadinessDecision


class DataPreparationError(ValueError):
    """Raised when generator training data preparation fails."""


@dataclass(frozen=True)
class PreparedGeneratorData:
    """Structured metadata and training dataset for synthetic generators."""

    training_data: pd.DataFrame
    discrete_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    n_rows: int
    source: str
    target_included: bool
    slice_support: float
    generation_mode: str
    feature_types: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation of the metadata."""
        return {
            "n_rows": self.n_rows,
            "source": self.source,
            "target_included": self.target_included,
            "slice_support": self.slice_support,
            "generation_mode": self.generation_mode,
            "discrete_columns": list(self.discrete_columns),
            "continuous_columns": list(self.continuous_columns),
            "feature_types": self.feature_types.copy(),
            "column_count": len(self.training_data.columns),
        }


def prepare_generator_data(
    affected_slice: pd.DataFrame,
    readiness_decision: ReadinessDecision,
    config: dict[str, Any],
    source: str = "affected_slice",
) -> PreparedGeneratorData:
    """Prepare tabular training data and metadata for synthetic generators.

    Enforces readiness governance, schema consistency, feature ordering,
    and supervised vs. unsupervised label trust policies.
    """
    if not isinstance(affected_slice, pd.DataFrame):
        raise DataPreparationError("affected_slice must be a pandas DataFrame.")
    if affected_slice.empty:
        raise DataPreparationError("Training data cannot be empty.")
    if affected_slice.columns.has_duplicates:
        raise DataPreparationError("Training data contains duplicate column names.")
    if not isinstance(readiness_decision, ReadinessDecision):
        raise DataPreparationError(
            "readiness_decision must be a ReadinessDecision instance."
        )
    if not isinstance(config, dict):
        raise DataPreparationError("config must be a dictionary.")
    if not isinstance(source, str) or not source.strip():
        raise DataPreparationError("source must be a non-empty string.")

    if not readiness_decision.generation_ready:
        raise DataPreparationError(
            "Readiness decision does not permit synthetic generation "
            f"(status={readiness_decision.status.value})."
        )

    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise DataPreparationError("Missing 'dataset' section in config.")

    feature_types = dataset_cfg.get("feature_types")
    if not isinstance(feature_types, dict) or not feature_types:
        raise DataPreparationError(
            "Missing or invalid 'feature_types' in dataset config."
        )

    target_column = dataset_cfg.get("target_column", "target")

    feature_columns = list(feature_types.keys())
    missing_features = [
        col for col in feature_columns if col not in affected_slice.columns
    ]
    if missing_features:
        raise DataPreparationError(
            f"Training data is missing required feature columns: {missing_features}"
        )

    # Check for missing values in feature columns
    if affected_slice[feature_columns].isna().any().any():
        raise DataPreparationError(
            "Training data contains missing values in required feature columns."
        )

    # Determine if target column is included based on supervised retraining readiness
    target_included = bool(
        readiness_decision.supervised_retraining_ready
        and target_column in affected_slice.columns
    )

    if target_included and affected_slice[target_column].isna().any():
        raise DataPreparationError(
            f"Target column '{target_column}' contains missing values."
        )

    columns_to_keep = feature_columns + ([target_column] if target_included else [])
    training_data = affected_slice[columns_to_keep].copy(deep=True)

    for column in feature_columns:
        if feature_types[column] == "continuous":
            if not pd.api.types.is_numeric_dtype(training_data[column]):
                raise DataPreparationError(
                    f"Continuous feature is not numeric: {column}"
                )
            if pd.api.types.is_bool_dtype(training_data[column]):
                raise DataPreparationError(
                    f"Continuous feature cannot be boolean: {column}"
                )

    discrete_columns = tuple(
        column
        for column in feature_columns
        if feature_types[column] == "categorical"
    )
    continuous_columns = tuple(
        column
        for column in feature_columns
        if feature_types[column] == "continuous"
    )

    if target_included:
        # Target metadata is separate from the configured feature mapping.
        discrete_columns = (*discrete_columns, target_column)

    if readiness_decision.current_rows <= 0:
        raise DataPreparationError(
            "Readiness decision has no valid current-row count."
        )
    if len(training_data) > readiness_decision.current_rows:
        raise DataPreparationError(
            "Training rows exceed the assessed current dataset size."
        )

    if source.strip() == "affected_slice":
        if len(training_data) != readiness_decision.affected_slice_rows:
            raise DataPreparationError(
                "Training row count does not match the "
                "readiness decision's affected-slice count."
            )

    slice_support = len(training_data) / readiness_decision.current_rows
    generation_mode = (
        "supervised_repair"
        if target_included
        else "unsupervised_exploratory"
    )

    return PreparedGeneratorData(
        training_data=training_data,
        discrete_columns=discrete_columns,
        continuous_columns=continuous_columns,
        n_rows=len(training_data),
        source=source.strip(),
        target_included=target_included,
        slice_support=round(slice_support, 6),
        generation_mode=generation_mode,
        feature_types=feature_types.copy(),
    )
