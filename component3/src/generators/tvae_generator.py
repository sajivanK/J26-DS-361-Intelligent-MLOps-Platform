"""TVAE synthetic tabular data generator for Component 3."""

from __future__ import annotations

import pandas as pd
from ctgan import TVAE

from .base import BaseGenerator


class TVAEGenerator(BaseGenerator):
    """TVAE synthetic tabular data generator with optional categorical metadata."""

    def __init__(
        self,
        epochs: int = 300,
        discrete_columns: list[str] | tuple[str, ...] | None = None,
        batch_size: int = 128,
    ):
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise ValueError("epochs must be a positive integer")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        if discrete_columns is not None:
            if not isinstance(discrete_columns, (list, tuple)):
                raise ValueError("discrete_columns must be a list or tuple")
            if (
                any(
                    not isinstance(column, str) or not column
                    for column in discrete_columns
                )
                or len(set(discrete_columns)) != len(discrete_columns)
            ):
                raise ValueError(
                    "discrete_columns must contain unique, non-empty string column names"
                )
            self.discrete_columns = (
                tuple(discrete_columns)
                if isinstance(discrete_columns, tuple)
                else list(discrete_columns)
            )
        else:
            self.discrete_columns = None

        self.epochs = epochs
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.model = TVAE(
            epochs=epochs,
            batch_size=batch_size,
            verbose=False,
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Train TVAE using real tabular data."""
        self.is_fitted = False

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Training data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Training data cannot be empty")

        if data.isna().any().any():
            raise ValueError("Training data cannot contain missing values")

        auto_discrete = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        if self.discrete_columns is not None:
            missing_columns = [
                column
                for column in self.discrete_columns
                if column not in data.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Discrete columns not found: {missing_columns}"
                )
            discrete_columns = list(dict.fromkeys(list(self.discrete_columns) + auto_discrete))
        else:
            discrete_columns = auto_discrete

        training_data = data.copy(deep=True)
        self.model.fit(
            training_data,
            discrete_columns=discrete_columns,
        )
        self.is_fitted = True

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic records after successful training."""
        if not self.is_fitted:
            raise RuntimeError("Train TVAE before sampling")

        if (
            isinstance(n_rows, bool)
            or not isinstance(n_rows, int)
            or n_rows <= 0
        ):
            raise ValueError("n_rows must be a positive integer")

        return self.model.sample(n_rows)
