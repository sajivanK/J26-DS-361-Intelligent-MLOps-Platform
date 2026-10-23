"""Registry and factory for Component 3 synthetic data generators."""

from __future__ import annotations

import inspect
from typing import Any

from component3.src.generators.base import BaseGenerator
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.tvae_generator import TVAEGenerator


class GeneratorPluginError(ValueError):
    """Raised when a generator cannot be selected or configured."""


class GeneratorPluginManager:
    """Manage generator registration and instance creation."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseGenerator]] = {}
        self.register("ctgan", CTGANGenerator)
        self.register("tvae", TVAEGenerator)
        self.register("tabddpm", TabDDPMGenerator)

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise GeneratorPluginError(
                "Generator name must be a non-empty string."
            )
        return name.strip().lower()

    def register(
        self,
        name: str,
        generator_cls: type[BaseGenerator],
        override: bool = False,
    ) -> None:
        """Register a generator class under the specified name."""
        norm_name = self._normalize_name(name)

        if not isinstance(generator_cls, type) or not issubclass(
            generator_cls, BaseGenerator
        ):
            raise GeneratorPluginError(
                f"Registered generator class for '{norm_name}' must be a subclass of BaseGenerator."
            )

        if norm_name in self._registry and not override:
            raise GeneratorPluginError(
                f"Generator '{norm_name}' is already registered. Set override=True to replace."
            )

        self._registry[norm_name] = generator_cls

    def unregister(self, name: str) -> None:
        """Unregister a generator class by name."""
        norm_name = self._normalize_name(name)
        if norm_name not in self._registry:
            raise GeneratorPluginError(
                f"Cannot unregister '{norm_name}': generator is not registered."
            )
        del self._registry[norm_name]

    def is_registered(self, name: str) -> bool:
        """Check if a generator name is registered."""
        try:
            norm_name = self._normalize_name(name)
            return norm_name in self._registry
        except GeneratorPluginError:
            return False

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)

    def available_generators(self) -> tuple[str, ...]:
        """Return a sorted tuple of registered generator names."""
        return tuple(sorted(self._registry.keys()))

    def get_generator_class(self, name: str) -> type[BaseGenerator]:
        """Retrieve the generator class for the given name."""
        norm_name = self._normalize_name(name)
        if norm_name not in self._registry:
            raise GeneratorPluginError(
                f"Unknown generator '{norm_name}'. Available: {self.available_generators()}"
            )
        return self._registry[norm_name]

    def create(self, name: str, **kwargs: Any) -> BaseGenerator:
        """Validate parameters and instantiate the requested generator."""
        norm_name = self._normalize_name(name)
        generator_cls = self.get_generator_class(norm_name)

        sig = inspect.signature(generator_cls.__init__)
        has_var_keyword = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        valid_params = {
            param.name
            for param in sig.parameters.values()
            if param.name != "self"
            and param.kind != inspect.Parameter.VAR_POSITIONAL
        }

        if not has_var_keyword:
            unsupported = sorted(set(kwargs.keys()) - valid_params)
            if unsupported:
                raise GeneratorPluginError(
                    f"Generator '{norm_name}' does not support parameter(s): {unsupported}. "
                    f"Supported parameters: {sorted(valid_params)}"
                )

        try:
            return generator_cls(**kwargs)
        except Exception as exc:
            if isinstance(exc, GeneratorPluginError):
                raise
            raise GeneratorPluginError(
                f"Failed to instantiate generator '{norm_name}': {exc}"
            ) from exc
