"""Validated, immutable specifications for offline Adult experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from c4.data import FEATURE_COLUMNS


CONTINUOUS_FEATURES = frozenset(
    {
        "age",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    }
)

CATEGORICAL_FEATURES = frozenset(FEATURE_COLUMNS) - CONTINUOUS_FEATURES

DriftMechanism = Literal[
    "none",
    "marginal_single",
    "marginal_multi",
    "interaction",
    "combined",
]


class ScenarioConfig(BaseModel):
    """Injection settings, distinct from measured harm or C1 evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["1.0"] = "1.0"
    dataset: Literal["uci_adult"] = "uci_adult"
    champion_version: str = Field(default="v1", min_length=1)

    mechanism: DriftMechanism
    magnitude: float = Field(ge=0.0, le=1.0)
    features: tuple[str, ...] = ()
    interaction_pairs: tuple[tuple[str, str], ...] = ()

    seed: int = Field(ge=0, le=2**32 - 1, strict=True)

    @model_validator(mode="after")
    def validate_injection_settings(self) -> ScenarioConfig:
        known_features = set(FEATURE_COLUMNS)

        if len(set(self.features)) != len(self.features):
            raise ValueError("Marginal features must be unique.")

        for feature in self.features:
            if feature not in known_features:
                raise ValueError(f"Unknown Adult feature: {feature}")

        seen_pairs: set[frozenset[str]] = set()

        for feature_a, feature_b in self.interaction_pairs:
            if feature_a not in known_features or feature_b not in known_features:
                raise ValueError("Interaction pairs contain unknown features.")

            if feature_a == feature_b:
                raise ValueError("Interaction pairs require distinct features.")

            pair_key = frozenset((feature_a, feature_b))

            if pair_key in seen_pairs:
                raise ValueError("Interaction pairs must be unique.")

            seen_pairs.add(pair_key)

        if self.mechanism == "none":
            if self.magnitude != 0.0:
                raise ValueError("No-drift controls require magnitude zero.")

            if self.features or self.interaction_pairs:
                raise ValueError("No-drift controls cannot specify features.")

            return self

        if self.magnitude == 0.0:
            raise ValueError(
                "Use mechanism='none' for zero-magnitude controls."
            )

        if self.mechanism == "marginal_single":
            if len(self.features) != 1 or self.interaction_pairs:
                raise ValueError(
                    "Marginal-single requires exactly one feature and no pairs."
                )

        elif self.mechanism == "marginal_multi":
            if len(self.features) < 2 or self.interaction_pairs:
                raise ValueError(
                    "Marginal-multi requires at least two features and no pairs."
                )

        elif self.mechanism == "interaction":
            if self.features or not self.interaction_pairs:
                raise ValueError(
                    "Interaction requires pairs and no marginal features."
                )

        elif self.mechanism == "combined":
            if not self.features or not self.interaction_pairs:
                raise ValueError(
                    "Combined requires marginal features and interaction pairs."
                )

            pair_features = {
                feature
                for pair in self.interaction_pairs
                for feature in pair
            }

            if set(self.features) & pair_features:
                raise ValueError(
                    "Combined marginal and interaction features must be disjoint."
                )

        return self

    def configuration_sha256(self) -> str:
        """Identify this specification using canonical JSON."""

        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def feature_types(self) -> dict[str, str]:
        """Return explicit types for all Adult model features."""

        return {
            feature: (
                "continuous"
                if feature in CONTINUOUS_FEATURES
                else "categorical"
            )
            for feature in FEATURE_COLUMNS
        }