"""Resource-aware generator selection, execution, and fallback orchestration.

Integrates TVAE, CTGAN, and TabDDPM via GeneratorPluginManager with schema
compatibility checks, deterministic seeding, bounded retries, and fallback handling.
"""

from __future__ import annotations

import logging
import time
from typing import Any
import numpy as np
import pandas as pd

from component3.src.generators.base import BaseGenerator
from component3.src.generators.plugin_manager import (
    GeneratorPluginError,
    GeneratorPluginManager,
)

logger = logging.getLogger(__name__)


class GeneratorExecutionError(RuntimeError):
    """Raised when generator training or sampling fails across retries and fallbacks."""


class GeneratorSelector:
    """Selects, configures, and executes tabular data generators with fallback."""

    def __init__(self, plugin_manager: GeneratorPluginManager | None = None) -> None:
        self.plugin_manager = plugin_manager or GeneratorPluginManager()

    def check_compatibility(
        self,
        generator_name: str,
        df: pd.DataFrame,
        categorical_columns: list[str] | tuple[str, ...],
        numerical_columns: list[str] | tuple[str, ...],
    ) -> tuple[bool, str]:
        """Check compatibility between requested generator and dataset schema."""
        norm_name = generator_name.strip().lower()
        if not self.plugin_manager.is_registered(norm_name):
            return False, f"Generator '{norm_name}' is not registered."

        n_rows = len(df)
        if n_rows < 30:
            return False, f"Dataset has {n_rows} rows; generators require at least 30 rows."

        # TabDDPM requires numericals to be standard scaled and can be slow on large schemas
        if norm_name == "tabddpm":
            if len(df.columns) > 50:
                return False, f"TabDDPM does not support schemas with >50 features (got {len(df.columns)})."

        return True, f"Generator '{norm_name}' is compatible with dataset schema."

    def select_fallback_generator(
        self,
        primary_generator: str,
        df: pd.DataFrame,
        categorical_columns: list[str] | tuple[str, ...],
        numerical_columns: list[str] | tuple[str, ...],
    ) -> str | None:
        """Deterministically select a secondary fallback generator if primary fails."""
        norm_primary = primary_generator.strip().lower()
        candidates = ["tvae", "ctgan", "tabddpm"]
        for cand in candidates:
            if cand != norm_primary:
                compat, _ = self.check_compatibility(cand, df, categorical_columns, numerical_columns)
                if compat:
                    return cand
        return None

    def fit_and_sample(
        self,
        generator_name: str,
        training_df: pd.DataFrame,
        num_samples: int,
        categorical_columns: list[str] | tuple[str, ...] | None = None,
        numerical_columns: list[str] | tuple[str, ...] | None = None,
        seed: int = 42,
        max_retries: int = 2,
        enable_fallback: bool = True,
        generator_kwargs: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit generator and sample synthetic candidates with bounded retries and fallback."""
        cat_cols = list(categorical_columns) if categorical_columns else []
        num_cols = list(numerical_columns) if numerical_columns else []
        kwargs = dict(generator_kwargs or {})

        # Check compatibility of primary generator
        is_compat, reason = self.check_compatibility(generator_name, training_df, cat_cols, num_cols)
        if not is_compat:
            if not enable_fallback:
                raise GeneratorExecutionError(f"Primary generator '{generator_name}' incompatible: {reason}")
            fallback = self.select_fallback_generator(generator_name, training_df, cat_cols, num_cols)
            if not fallback:
                raise GeneratorExecutionError(f"No compatible generator found. Primary '{generator_name}' failed: {reason}")
            logger.warning("Primary generator '%s' incompatible (%s). Falling back to '%s'.", generator_name, reason, fallback)
            generator_name = fallback

        active_generator = generator_name.strip().lower()
        attempt = 0
        provenance: dict[str, Any] = {
            "requested_generator": generator_name,
            "executed_generator": active_generator,
            "seed": seed,
            "num_requested_samples": num_samples,
            "retries_attempted": 0,
            "fallback_used": False,
            "fit_duration_seconds": 0.0,
            "sample_duration_seconds": 0.0,
        }

        while attempt < max_retries:
            attempt += 1
            provenance["retries_attempted"] = attempt - 1
            try:
                import inspect
                gen_cls = self.plugin_manager.get_generator_class(active_generator)
                sig = inspect.signature(gen_cls.__init__)
                accepted = set(sig.parameters.keys()) - {"self"}

                gen_params = dict(kwargs)
                if "discrete_columns" in accepted and "discrete_columns" not in gen_params:
                    auto_cat = training_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                    combined_discrete = list(dict.fromkeys([c for c in cat_cols if c in training_df.columns] + auto_cat))
                    gen_params["discrete_columns"] = combined_discrete
                if "epochs" in accepted and "epochs" not in gen_params:
                    gen_params["epochs"] = 10

                filtered_params = {k: v for k, v in gen_params.items() if k in accepted}
                generator_inst = gen_cls(**filtered_params)
                if hasattr(getattr(generator_inst, "model", None), "set_random_state"):
                    generator_inst.model.set_random_state(seed)

                # Fit generator on training data
                t0_fit = time.perf_counter()
                generator_inst.fit(training_df)
                dur_fit = round(time.perf_counter() - t0_fit, 3)
                provenance["fit_duration_seconds"] = dur_fit

                # Sample synthetic records
                t0_sample = time.perf_counter()
                synthetic_df = generator_inst.sample(num_samples)
                dur_sample = round(time.perf_counter() - t0_sample, 3)
                provenance["sample_duration_seconds"] = dur_sample

                if len(synthetic_df) == 0:
                    raise RuntimeError("Generator produced 0 samples.")

                provenance["executed_generator"] = active_generator
                provenance["num_produced_samples"] = len(synthetic_df)
                return synthetic_df, provenance

            except Exception as exc:
                logger.error("Attempt %d with generator '%s' failed: %s", attempt, active_generator, exc)
                if attempt >= max_retries:
                    # Attempt fallback if enabled
                    if enable_fallback and not provenance["fallback_used"]:
                        fallback = self.select_fallback_generator(active_generator, training_df, cat_cols, num_cols)
                        if fallback and fallback != active_generator:
                            logger.warning("Switching to fallback generator '%s' after failure of '%s'.", fallback, active_generator)
                            active_generator = fallback
                            provenance["fallback_used"] = True
                            provenance["primary_failure_error"] = str(exc)
                            attempt = 0  # Reset retries for fallback
                            continue
                    raise GeneratorExecutionError(
                        f"Generator execution failed for '{active_generator}' after {attempt} attempts: {exc}"
                    ) from exc

        raise GeneratorExecutionError(f"Failed to generate synthetic data with generator '{active_generator}'.")
