"""Reference-derived slice definitions for offline experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from c4.data import FEATURE_COLUMNS
from c4.scenarios.config import CONTINUOUS_FEATURES


class SliceRule(BaseModel):
    """A frozen predicate on processed Adult features."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    name: str = Field(min_length=1)
    feature: str
    operator: Literal["le", "gt", "eq"]
    value: float

    @model_validator(mode="after")
    def validate_feature(self) -> SliceRule:
        if self.feature not in FEATURE_COLUMNS:
            raise ValueError("Slice feature must be an Adult input feature.")

        if self.operator != "eq" and self.feature not in CONTINUOUS_FEATURES:
            raise ValueError(
                "Ordered slice comparisons require a continuous feature."
            )

        return self

    def mask(self, clean_frame: pd.DataFrame) -> pd.Series:
        """Compute membership before perturbation, preserving row indices."""

        if self.feature not in clean_frame.columns:
            raise ValueError(f"Slice feature is missing: {self.feature}")

        values = clean_frame[self.feature]

        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError("Slice feature must be numeric.")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Slice feature contains missing or infinite values.")

        if self.operator == "le":
            return values <= self.value

        if self.operator == "gt":
            return values > self.value

        return values == self.value


def reference_quantile_slice(
    reference: pd.DataFrame,
    *,
    name: str,
    feature: str,
    quantile: float,
    operator: Literal["le", "gt"] = "le",
) -> SliceRule:
    """Derive a threshold from champion reference features only."""

    if feature not in CONTINUOUS_FEATURES:
        raise ValueError("Quantile slices require a continuous feature.")

    if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("Quantile must be finite and between zero and one.")

    # Validate the reference column before deriving a threshold.
    probe = SliceRule(
        name=name,
        feature=feature,
        operator=operator,
        value=0.0,
    )
    probe.mask(reference)

    if reference.empty:
        raise ValueError("Reference dataset is empty.")

    threshold = float(reference[feature].quantile(quantile))

    return SliceRule(
        name=name,
        feature=feature,
        operator=operator,
        value=threshold,
    )