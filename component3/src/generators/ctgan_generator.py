"""CTGAN synthetic tabular data generator for Component 3."""

from __future__ import annotations

import pandas as pd
from ctgan import CTGAN

from .base import BaseGenerator


class CTGANGenerator(BaseGenerator):
    """CTGAN generator with optional categorical feature metadata."""

    def __init__(
        self,
        epochs: int = 300,
        discrete_columns: list[str] | tuple[str, ...] | None = None,
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
            self.discrete_columns = list(discrete_columns)
        else:
            self.discrete_columns = None

        self.epochs = epochs
        self.model = CTGAN(
            epochs=epochs,
            verbose=False,
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Train CTGAN using real tabular data."""
        self.is_fitted = False

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Training data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Training data cannot be empty")

        training_data = data.copy(deep=True)

        if self.discrete_columns is not None:
            missing_columns = [
                column
                for column in self.discrete_columns
                if column not in training_data.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Configured discrete columns missing from data: {missing_columns}"
                )
            discrete_columns = list(self.discrete_columns)
        else:
            discrete_columns = training_data.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        self.model.fit(
            training_data,
            discrete_columns=discrete_columns,
        )

        self.is_fitted = True

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic tabular records."""
        if not self.is_fitted:
            raise RuntimeError("Train CTGAN before sampling")

        if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
            raise ValueError("n_rows must be positive")

        return self.model.sample(n_rows)