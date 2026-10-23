"""Versioned, validated RepairRequest contract for Component 3.

Provides strict schema validation, path safety guarantees, and authorized
dataset reference checks for incoming repair requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

from component3.src.contracts.targeted_repair import DiagnosticSliceDefinition
from component3.src.datasets.dataset_registry import get_dataset_registry

REPO_ROOT = Path(__file__).resolve().parents[3]


class DriftType(str, Enum):
    """Supported drift types."""
    NONE = "NONE"
    CONCEPT_DRIFT = "CONCEPT_DRIFT"
    COVARIATE_SHIFT = "COVARIATE_SHIFT"
    SUBPOPULATION_SHIFT = "SUBPOPULATION_SHIFT"
    INTERACTION = "INTERACTION"
    UNKNOWN = "UNKNOWN"


class DriftSeverity(str, Enum):
    """Categorical severity of detected drift."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RepairObjective(str, Enum):
    """Optimization objectives for repair strategy selection."""
    MAXIMIZE_TARGET_ACCURACY = "MAXIMIZE_TARGET_ACCURACY"
    PRESERVE_BACKWARD_COMPATIBILITY = "PRESERVE_BACKWARD_COMPATIBILITY"
    BALANCED_RETENTION = "BALANCED_RETENTION"


class ExecutionMode(str, Enum):
    """Execution modes for repair orchestrator."""
    DRY_RUN = "DRY_RUN"
    EXECUTE = "EXECUTE"
    EVALUATE_ONLY = "EVALUATE_ONLY"


class InvalidRepairRequestError(ValueError):
    """Raised when a RepairRequest fails validation or security checks."""


def _validate_safe_path(p: Path | str | None, allow_none: bool = False) -> Path | None:
    """Validate that path does not escape repository root or execute unsafe traversal."""
    if p is None:
        if allow_none:
            return None
        raise InvalidRepairRequestError("Path cannot be None.")

    path_obj = Path(p)
    # Check for obvious traversal sequences
    parts = path_obj.parts
    if ".." in parts:
        raise InvalidRepairRequestError(f"Path traversal ('..') is prohibited: {p}")

    # Resolve against REPO_ROOT if relative
    if not path_obj.is_absolute():
        resolved = (REPO_ROOT / path_obj).resolve()
    else:
        resolved = path_obj.resolve()

    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise InvalidRepairRequestError(
            f"Path '{resolved}' is outside repository root '{REPO_ROOT.resolve()}'."
        )

    return resolved


@dataclass(frozen=True)
class RepairRequest:
    """Versioned, validated repair request contract."""
    # Identification & Governance
    request_id: str = field(default_factory=lambda: f"REQ-{uuid.uuid4().hex[:12].upper()}")
    schema_version: str = "1.0.0"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Dataset and Model References
    dataset_id: str = "telco_churn"
    model_id: str = "telco_rf_experimental_source"
    source_model_path: str | Path = "component3/models/experimental/telco_rf_experimental_source.joblib"
    source_preprocessor_path: str | Path = "component3/models/experimental/telco_preprocessor_experimental_source.joblib"

    # Drift Context
    drift_detected: bool = True
    drift_type: DriftType = DriftType.CONCEPT_DRIFT
    drift_severity: DriftSeverity = DriftSeverity.HIGH
    drift_severity_score: float = 0.50
    affected_features: tuple[str, ...] = ("tenure",)

    # Diagnostic Payload (Optional)
    diagnostic_slice: DiagnosticSliceDefinition | None = None

    # Available Labeled Adaptation Data
    has_authentic_adaptation_labels: bool = True
    adaptation_data_path: str | Path | None = None
    adaptation_budget_size: int = 200

    # Repair Objective & Constraints
    repair_objective: RepairObjective = RepairObjective.BALANCED_RETENTION
    source_retention_tolerance: float = 0.05
    target_f1_improvement_threshold: float = 0.02
    max_candidate_search_budget: int = 200

    # Operational Resource Limits
    max_cpu_cores: int = 2
    max_memory_mb: float = 4096.0
    max_runtime_seconds: float = 300.0

    # Validation Thresholds
    min_dcr_p05: float = 0.01
    min_structural_validity_rate: float = 0.95
    max_exact_duplicate_ratio: float = 0.0

    # Execution Mode
    execution_mode: ExecutionMode = ExecutionMode.EXECUTE

    def __post_init__(self) -> None:
        """Enforce strict validation across all request fields."""
        if not self.request_id or not isinstance(self.request_id, str):
            raise InvalidRepairRequestError("request_id must be a non-empty string.")

        if self.schema_version != "1.0.0":
            raise InvalidRepairRequestError(f"Unsupported schema version: {self.schema_version}")

        # Check authorized dataset registration
        reg = get_dataset_registry()
        if self.dataset_id not in reg.list_datasets():
            raise InvalidRepairRequestError(
                f"Unauthorized dataset '{self.dataset_id}'. Must be registered in DatasetRegistry: {reg.list_datasets()}"
            )

        # Check path safety
        _validate_safe_path(self.source_model_path)
        _validate_safe_path(self.source_preprocessor_path)
        if self.adaptation_data_path is not None:
            _validate_safe_path(self.adaptation_data_path)

        # Value bounds
        if not (0.0 <= self.drift_severity_score <= 1.0):
            raise InvalidRepairRequestError(
                f"drift_severity_score must be between 0.0 and 1.0, got {self.drift_severity_score}"
            )

        if self.source_retention_tolerance < 0.0:
            raise InvalidRepairRequestError("source_retention_tolerance must be non-negative.")

        if self.max_cpu_cores < 1:
            raise InvalidRepairRequestError("max_cpu_cores must be at least 1.")

        if self.max_memory_mb < 256.0:
            raise InvalidRepairRequestError("max_memory_mb must be at least 256 MB.")

        if self.max_runtime_seconds < 5.0:
            raise InvalidRepairRequestError("max_runtime_seconds must be at least 5.0 seconds.")

        if self.adaptation_budget_size < 0:
            raise InvalidRepairRequestError("adaptation_budget_size must be non-negative.")

        if not self.has_authentic_adaptation_labels and self.adaptation_budget_size > 0:
            raise InvalidRepairRequestError(
                "Inconsistent request: adaptation_budget_size > 0 but has_authentic_adaptation_labels is False."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairRequest:
        """Parse dictionary payload into validated RepairRequest."""
        d = dict(data)

        # Convert enums if strings
        if "drift_type" in d and isinstance(d["drift_type"], str):
            d["drift_type"] = DriftType(d["drift_type"])
        if "drift_severity" in d and isinstance(d["drift_severity"], str):
            d["drift_severity"] = DriftSeverity(d["drift_severity"])
        if "repair_objective" in d and isinstance(d["repair_objective"], str):
            d["repair_objective"] = RepairObjective(d["repair_objective"])
        if "execution_mode" in d and isinstance(d["execution_mode"], str):
            d["execution_mode"] = ExecutionMode(d["execution_mode"])

        if "affected_features" in d and isinstance(d["affected_features"], (list, set)):
            d["affected_features"] = tuple(d["affected_features"])

        if "diagnostic_slice" in d and isinstance(d["diagnostic_slice"], dict):
            d["diagnostic_slice"] = DiagnosticSliceDefinition(**d["diagnostic_slice"])

        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Convert request to JSON-serializable dictionary."""
        d = asdict(self)
        d["drift_type"] = self.drift_type.value
        d["drift_severity"] = self.drift_severity.value
        d["repair_objective"] = self.repair_objective.value
        d["execution_mode"] = self.execution_mode.value
        d["source_model_path"] = str(self.source_model_path)
        d["source_preprocessor_path"] = str(self.source_preprocessor_path)
        if self.adaptation_data_path:
            d["adaptation_data_path"] = str(self.adaptation_data_path)
        if self.diagnostic_slice:
            d["diagnostic_slice"] = self.diagnostic_slice.to_dict()
        return d
