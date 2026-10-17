"""Validate platform inputs without training, reading data, or authorizing release."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from c4.contracts.models import (
    C1Evidence, C2Diagnosis, C3RepairPackage, StrictContractModel, UpstreamInput,
)


class PlatformContext(StrictContractModel):
    dataset: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    current_dataset_uri: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Upstream timestamp must include a timezone.")
    return value


class PlatformDiagnosis(C2Diagnosis):
    diagnosis_timestamp: datetime
    _timestamp = field_validator("diagnosis_timestamp")(_aware)


class PlatformRepair(C3RepairPackage):
    repair_timestamp: datetime
    _timestamp = field_validator("repair_timestamp")(_aware)


def _finite_json(value: Any) -> None:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Platform object keys must be strings.")
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Platform inputs must contain finite numbers.")
    elif value is not None and not isinstance(value, (str, bool, int)):
        raise ValueError("Platform inputs must be JSON-compatible values.")


@dataclass(frozen=True)
class AdaptedPlatformInput:
    context: PlatformContext
    inputs: UpstreamInput
    audit_envelope: dict[str, Any]
    prerequisite_failures: tuple[str, ...]


def adapt_platform_upstream(upstream: dict[str, Any]) -> AdaptedPlatformInput:
    """Map platform names to research contracts; never infer missing evidence.

    Additional context/output records are preserved for audit. Only the four
    context fields consumed here are validated. Runtime checks, data lineage,
    model/preprocessing compatibility and forecast applicability remain separate.
    """
    if not isinstance(upstream, dict):
        raise ValueError("Platform upstream must be an object.")
    _finite_json(upstream)
    for key in ("context", "c1", "c2_to_c4", "c3"):
        if not isinstance(upstream.get(key), dict):
            raise ValueError(f"Platform upstream requires object: {key}")
    context = PlatformContext.model_validate({
        key: upstream["context"][key]
        for key in PlatformContext.model_fields
        if key in upstream["context"]
    })
    c1 = C1Evidence.model_validate(upstream["c1"])
    c2 = PlatformDiagnosis.model_validate(upstream["c2_to_c4"])
    c3 = PlatformRepair.model_validate(upstream["c3"])
    if context.model_id != c1.model_id:
        raise ValueError("Platform and C1 champion model IDs differ.")
    if context.current_dataset_uri != c1.current_dataset_uri:
        raise ValueError("Platform and C1 current dataset URIs differ.")
    inputs = UpstreamInput(
        c1_evidence=c1,
        c2_diagnosis=C2Diagnosis.model_validate(
            c2.model_dump(exclude={"diagnosis_timestamp"})),
        c3_repair_package=C3RepairPackage.model_validate(
            c3.model_dump(exclude={"repair_timestamp"})),
    )
    failures = []
    if c2.diagnosis_stability < 0.5:
        failures.append("diagnosis_stability")
    if c3.repair_package_status != "APPROVED":
        failures.append("repair_approved")
    if c3.privacy_status != "PASS":
        failures.append("privacy_check")
    return AdaptedPlatformInput(context, inputs, deepcopy(upstream), tuple(failures))
