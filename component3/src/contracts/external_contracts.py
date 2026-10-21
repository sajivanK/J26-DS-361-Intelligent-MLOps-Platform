"""External Data Contracts and Schema Validation for Component 3.

Implements strict validation for the agreed multi-component payload:
1. c1_evidence (from Component 1 - Drift Detection)
2. c2_diagnosis (from Component 2 - Root Cause Diagnosis)
3. data_context (from Platform Orchestrator)

Also provides safe serialization and validation for the exact 12-field C4 output contract
defined in shared/contracts/c3_output_schema.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


class ContractValidationError(ValueError):
    """Raised when an incoming external contract fails schema, type, or range validation."""


class DependencyResolutionError(FileNotFoundError):
    """Raised when data_context references datasets or model artifacts that do not exist."""


# -----------------------------------------------------------------------------
# Allowlist and Enum Validations
# -----------------------------------------------------------------------------

VALID_C1_DRIFT_TYPES = {
    "marginal_single",
    "marginal_multi",
    "marginal",
    "interaction",
    "combined",
    "concept_drift",
    "covariate_shift",
    "subpopulation_shift",
    "none",
    "no_drift",
}

VALID_C1_DRIFT_SEVERITIES = {
    "none",
    "low",
    "moderate",
    "medium",
    "high",
    "critical",
}

VALID_C2_ROOT_CAUSE_CATEGORIES = {
    "marginal",
    "interaction",
    "combined",
    "unknown",
    "concept_shift",
    "covariate_shift",
    "sampling_bias",
    "data_quality",
    "external_factor",
}

VALID_LABEL_STATUSES = {"AVAILABLE", "UNAVAILABLE"}

VALID_C4_PACKAGE_STATUSES = {"APPROVED", "REJECTED"}
VALID_C4_GENERATORS = {"CTGAN", "TVAE", "TabDDPM"}
VALID_C4_PRIVACY_STATUSES = {"PASS", "FAIL"}


# -----------------------------------------------------------------------------
# Dataclasses for Structured Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class C1EvidenceContract:
    """Validated representation of c1_evidence."""
    predicted_accuracy_drop: float
    confidence_interval: tuple[float, float]
    drift_type: str
    drift_severity: str
    affected_features: tuple[str, ...]
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_harm_proportion(self) -> float:
        """Return drop as a proportion [0.0, 1.0]."""
        if self.predicted_accuracy_drop > 1.0:
            return round(self.predicted_accuracy_drop / 100.0, 4)
        return round(self.predicted_accuracy_drop, 4)

    @property
    def confidence_interval_proportions(self) -> tuple[float, float]:
        """Return confidence interval as proportions [0.0, 1.0]."""
        low, high = self.confidence_interval
        if high > 1.0 or low > 1.0:
            return (round(low / 100.0, 4), round(high / 100.0, 4))
        return (round(low, 4), round(high, 4))

    @property
    def is_drift_detected(self) -> bool:
        """Derive boolean drift detection flag based on contract semantics."""
        return (
            self.drift_type.lower() not in ("none", "no_drift")
            and self.drift_severity.lower() != "none"
            and self.predicted_harm_proportion > 0.01
        )


@dataclass(frozen=True)
class C2DiagnosisContract:
    """Validated representation of c2_diagnosis."""
    root_cause_feature: str
    root_cause_category: str
    affected_slice_rule: str
    affected_features_ranked: tuple[str, ...]
    affected_value_ranges: dict[str, Any]
    slice_support: float
    diagnosis_stability: float
    drift_impact_score: float
    temporal_window: int
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_slice_rule(self) -> str:
        """Normalize boolean operator casing for safe AST evaluation."""
        rule = self.affected_slice_rule.strip()
        rule = re.sub(r"\bAND\b", "and", rule)
        rule = re.sub(r"\bOR\b", "or", rule)
        rule = re.sub(r"\bNOT\b", "not", rule)
        return rule


@dataclass(frozen=True)
class DataContextContract:
    """Validated representation of data_context from platform orchestrator."""
    model_id: str
    dataset_id: str
    reference_dataset_uri: str
    current_dataset_uri: str
    target_column: str
    schema_version: str
    label_status: str
    preprocessing_uri: str | None = None
    adaptation_budget_size: int = 200
    source_retention_tolerance: float = 0.05
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_authentic_labels(self) -> bool:
        return self.label_status.upper() == "AVAILABLE"


@dataclass(frozen=True)
class ThreeSectionRepairInput:
    """Complete, validated 3-section input container."""
    c1_evidence: C1EvidenceContract
    c2_diagnosis: C2DiagnosisContract
    data_context: DataContextContract
    raw_payload: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Input Validation Functions
# -----------------------------------------------------------------------------

def validate_c1_evidence(data: dict[str, Any]) -> tuple[bool, list[str], C1EvidenceContract | None]:
    """Strictly validate c1_evidence dictionary against contract specification."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["c1_evidence must be a dictionary."], None

    # Required fields
    # Accept predicted_accuracy_drop OR predicted_harm
    raw_drop = data.get("predicted_accuracy_drop", data.get("predicted_harm"))
    if raw_drop is None:
        errors.append("c1_evidence missing required field: 'predicted_accuracy_drop' (or 'predicted_harm').")
    elif not isinstance(raw_drop, (int, float)) or raw_drop < 0:
        errors.append(f"predicted_accuracy_drop must be a non-negative number, got {raw_drop}.")

    # Confidence interval (accept confidence_interval or harm_interval)
    raw_ci = data.get("confidence_interval", data.get("harm_interval"))
    if raw_ci is None:
        errors.append("c1_evidence missing required field: 'confidence_interval'.")
    elif not isinstance(raw_ci, (list, tuple)) or len(raw_ci) != 2:
        errors.append("confidence_interval must be a list/tuple of two floats [lower, upper].")
    else:
        low, high = raw_ci
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            errors.append(f"confidence_interval bounds must be numbers, got {raw_ci}.")
        elif low > high:
            errors.append(f"confidence_interval lower bound ({low}) exceeds upper bound ({high}).")

    # Drift type
    raw_type = data.get("drift_type")
    if not raw_type or not isinstance(raw_type, str):
        errors.append("c1_evidence missing required string field: 'drift_type'.")
    elif raw_type.lower() not in VALID_C1_DRIFT_TYPES:
        errors.append(f"drift_type '{raw_type}' not in allowed values: {sorted(VALID_C1_DRIFT_TYPES)}.")

    # Drift severity
    raw_sev = data.get("drift_severity")
    if not raw_sev or not isinstance(raw_sev, str):
        errors.append("c1_evidence missing required string field: 'drift_severity'.")
    elif raw_sev.lower() not in VALID_C1_DRIFT_SEVERITIES:
        errors.append(f"drift_severity '{raw_sev}' not in allowed values: {sorted(VALID_C1_DRIFT_SEVERITIES)}.")

    # Affected features
    raw_features = data.get("affected_features", data.get("features_drifted"))
    if raw_features is None:
        errors.append("c1_evidence missing required field: 'affected_features'.")
    elif not isinstance(raw_features, (list, tuple)):
        errors.append("affected_features must be a list of feature name strings.")
    elif not all(isinstance(f, str) and f.strip() for f in raw_features):
        errors.append("All items in affected_features must be non-empty strings.")

    if errors:
        return False, errors, None

    contract = C1EvidenceContract(
        predicted_accuracy_drop=float(raw_drop),
        confidence_interval=(float(raw_ci[0]), float(raw_ci[1])),
        drift_type=str(raw_type),
        drift_severity=str(raw_sev),
        affected_features=tuple(str(f).strip() for f in raw_features),
        raw_metadata={k: v for k, v in data.items() if k not in {
            "predicted_accuracy_drop", "predicted_harm", "confidence_interval",
            "harm_interval", "drift_type", "drift_severity", "affected_features", "features_drifted"
        }},
    )
    return True, [], contract


def validate_c2_diagnosis(data: dict[str, Any]) -> tuple[bool, list[str], C2DiagnosisContract | None]:
    """Strictly validate c2_diagnosis dictionary against contract specification."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["c2_diagnosis must be a dictionary."], None

    # Required fields
    rc_feature = data.get("root_cause_feature")
    if not rc_feature or not isinstance(rc_feature, str):
        errors.append("c2_diagnosis missing required string field: 'root_cause_feature'.")

    rc_cat = data.get("root_cause_category")
    if not rc_cat or not isinstance(rc_cat, str):
        errors.append("c2_diagnosis missing required string field: 'root_cause_category'.")
    elif rc_cat.lower() not in VALID_C2_ROOT_CAUSE_CATEGORIES:
        errors.append(f"root_cause_category '{rc_cat}' not in allowed values: {sorted(VALID_C2_ROOT_CAUSE_CATEGORIES)}.")

    slice_rule = data.get("affected_slice_rule", data.get("slice_predicate", data.get("diagnosed_problematic_slice")))
    if not slice_rule or not isinstance(slice_rule, str) or not slice_rule.strip():
        errors.append("c2_diagnosis missing required non-empty string field: 'affected_slice_rule'.")

    ranked_feats = data.get("affected_features_ranked", data.get("affected_features"))
    if ranked_feats is None:
        errors.append("c2_diagnosis missing required field: 'affected_features_ranked'.")
    elif not isinstance(ranked_feats, (list, tuple)):
        errors.append("affected_features_ranked must be a list of strings.")

    val_ranges = data.get("affected_value_ranges")
    if val_ranges is not None and not isinstance(val_ranges, dict):
        errors.append("affected_value_ranges must be a dictionary.")

    # Numeric score ranges [0.0, 1.0]
    for field_name in ("slice_support", "diagnosis_stability", "drift_impact_score"):
        val = data.get(field_name)
        if val is None:
            errors.append(f"c2_diagnosis missing required float field: '{field_name}'.")
        elif not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
            errors.append(f"{field_name} must be a float between 0.0 and 1.0, got {val}.")

    temp_win = data.get("temporal_window", 1)
    if not isinstance(temp_win, int) or temp_win < 1:
        errors.append(f"temporal_window must be an integer >= 1, got {temp_win}.")

    if errors:
        return False, errors, None

    contract = C2DiagnosisContract(
        root_cause_feature=str(rc_feature).strip(),
        root_cause_category=str(rc_cat).strip(),
        affected_slice_rule=str(slice_rule).strip(),
        affected_features_ranked=tuple(str(f).strip() for f in ranked_feats),
        affected_value_ranges=dict(val_ranges or {}),
        slice_support=float(data["slice_support"]),
        diagnosis_stability=float(data["diagnosis_stability"]),
        drift_impact_score=float(data["drift_impact_score"]),
        temporal_window=int(temp_win),
        raw_metadata={k: v for k, v in data.items() if k not in {
            "root_cause_feature", "root_cause_category", "affected_slice_rule",
            "slice_predicate", "diagnosed_problematic_slice", "affected_features_ranked",
            "affected_value_ranges", "slice_support", "diagnosis_stability",
            "drift_impact_score", "temporal_window"
        }},
    )
    return True, [], contract


def validate_data_context(data: dict[str, Any]) -> tuple[bool, list[str], DataContextContract | None]:
    """Strictly validate data_context dictionary against contract specification."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["data_context must be a dictionary."], None

    for req_field in ("model_id", "dataset_id", "reference_dataset_uri", "current_dataset_uri", "target_column"):
        val = data.get(req_field)
        if not val or not isinstance(val, str) or not val.strip():
            errors.append(f"data_context missing required non-empty string field: '{req_field}'.")

    lbl_status = data.get("label_status", "AVAILABLE")
    if not isinstance(lbl_status, str) or lbl_status.upper() not in VALID_LABEL_STATUSES:
        errors.append(f"label_status must be one of {sorted(VALID_LABEL_STATUSES)}, got '{lbl_status}'.")

    # Traversal safety checks on URIs
    for uri_field in ("reference_dataset_uri", "current_dataset_uri", "preprocessing_uri"):
        val = data.get(uri_field)
        if val is not None and isinstance(val, str):
            if ".." in Path(val).parts:
                errors.append(f"Path traversal ('..') is prohibited in {uri_field}: '{val}'.")

    if errors:
        return False, errors, None

    budget = int(data.get("adaptation_budget_size", 200))
    tolerance = float(data.get("source_retention_tolerance", 0.05))

    contract = DataContextContract(
        model_id=str(data["model_id"]).strip(),
        dataset_id=str(data["dataset_id"]).strip(),
        reference_dataset_uri=str(data["reference_dataset_uri"]).strip(),
        current_dataset_uri=str(data["current_dataset_uri"]).strip(),
        target_column=str(data["target_column"]).strip(),
        schema_version=str(data.get("schema_version", "1.0.0")).strip(),
        label_status=str(lbl_status).upper().strip(),
        preprocessing_uri=str(data.get("preprocessing_uri", "")).strip() or None,
        adaptation_budget_size=budget,
        source_retention_tolerance=tolerance,
        raw_metadata={k: v for k, v in data.items() if k not in {
            "model_id", "dataset_id", "reference_dataset_uri", "current_dataset_uri",
            "target_column", "schema_version", "label_status", "preprocessing_uri",
            "adaptation_budget_size", "source_retention_tolerance"
        }},
    )
    return True, [], contract


def validate_three_section_contract(
    raw_payload: dict[str, Any] | str | Path,
) -> tuple[bool, list[str], ThreeSectionRepairInput | None]:
    """Parse and strictly validate complete 3-section input payload."""
    if isinstance(raw_payload, (str, Path)):
        p = Path(raw_payload)
        if not p.is_file():
            return False, [f"Payload file not found: {p}"], None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            return False, [f"JSON decoding error: {exc}"], None
    elif isinstance(raw_payload, dict):
        data = dict(raw_payload)
    else:
        return False, [f"Unsupported payload type: {type(raw_payload).__name__}"], None

    all_errors: list[str] = []

    # Check top-level sections
    for sec in ("c1_evidence", "c2_diagnosis", "data_context"):
        if sec not in data:
            all_errors.append(f"Missing required top-level section: '{sec}'.")

    if all_errors:
        return False, all_errors, None

    c1_ok, c1_errs, c1_contract = validate_c1_evidence(data["c1_evidence"])
    c2_ok, c2_errs, c2_contract = validate_c2_diagnosis(data["c2_diagnosis"])
    dc_ok, dc_errs, dc_contract = validate_data_context(data["data_context"])

    all_errors.extend(c1_errs)
    all_errors.extend(c2_errs)
    all_errors.extend(dc_errs)

    if not (c1_ok and c2_ok and dc_ok):
        return False, all_errors, None

    container = ThreeSectionRepairInput(
        c1_evidence=c1_contract,  # type: ignore[arg-type]
        c2_diagnosis=c2_contract,  # type: ignore[arg-type]
        data_context=dc_contract,  # type: ignore[arg-type]
        raw_payload=data,
    )
    return True, [], container


# -----------------------------------------------------------------------------
# Dependency Resolution & Artifact Availability
# -----------------------------------------------------------------------------

def resolve_data_context_artifacts(
    data_context: DataContextContract,
    repo_root: Path = REPO_ROOT,
) -> tuple[bool, list[str], dict[str, Path]]:
    """Verify physical presence of referenced dataset and preprocessor paths on disk.
    
    Returns (success, missing_errors, resolved_paths_dict).
    """
    errors: list[str] = []
    resolved: dict[str, Path] = {}

    # 1. Reference dataset
    ref_p = Path(data_context.reference_dataset_uri)
    if not ref_p.is_absolute():
        ref_p = repo_root / ref_p
    if not ref_p.is_file():
        errors.append(
            f"Reference dataset not found at URI '{data_context.reference_dataset_uri}' "
            f"(resolved: {ref_p})."
        )
    else:
        resolved["reference_dataset"] = ref_p

    # 2. Current dataset
    cur_p = Path(data_context.current_dataset_uri)
    if not cur_p.is_absolute():
        cur_p = repo_root / cur_p
    if not cur_p.is_file():
        errors.append(
            f"Current drifted dataset not found at URI '{data_context.current_dataset_uri}' "
            f"(resolved: {cur_p})."
        )
    else:
        resolved["current_dataset"] = cur_p

    # 3. Preprocessor / script (optional but if provided, must exist)
    if data_context.preprocessing_uri:
        prep_p = Path(data_context.preprocessing_uri)
        if not prep_p.is_absolute():
            prep_p = repo_root / prep_p
        if not prep_p.is_file():
            errors.append(
                f"Preprocessing artifact not found at URI '{data_context.preprocessing_uri}' "
                f"(resolved: {prep_p})."
            )
        else:
            resolved["preprocessing"] = prep_p

    return (len(errors) == 0), errors, resolved


# -----------------------------------------------------------------------------
# Output C4 Contract Schema & Helper
# -----------------------------------------------------------------------------

C4_FLAT_CONTRACT_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "C3OutputSchema",
    "type": "object",
    "required": [
        "repair_dataset_uri",
        "repair_package_status",
        "generator",
        "n_synthetic_rows",
        "augmentation_ratio",
        "fidelity_score",
        "utility_score",
        "privacy_status",
        "target_coverage",
        "prescription_confidence",
        "repair_version",
        "repair_timestamp",
    ],
    "properties": {
        "repair_dataset_uri": {"type": "string"},
        "repair_package_status": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
        "generator": {"type": "string", "enum": ["CTGAN", "TVAE", "TabDDPM", "NONE"]},
        "n_synthetic_rows": {"type": "integer", "minimum": 0},
        "augmentation_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "fidelity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "utility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "privacy_status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "target_coverage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "prescription_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "repair_version": {"type": "string"},
        "repair_timestamp": {"type": "string"},
    },
    "additionalProperties": False,
}


def validate_c4_flat_contract(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate flat 12-field dictionary against C4 schema requirements."""
    errors: list[str] = []
    required_keys = C4_FLAT_CONTRACT_JSON_SCHEMA["required"]

    missing = [k for k in required_keys if k not in payload]
    if missing:
        errors.append(f"Missing required fields: {missing}")
        return False, errors

    # Check field types and value constraints
    status = payload.get("repair_package_status")
    if status not in ("APPROVED", "REJECTED"):
        errors.append(f"repair_package_status must be 'APPROVED' or 'REJECTED', got '{status}'.")

    gen = payload.get("generator")
    if gen not in ("CTGAN", "TVAE", "TabDDPM", "NONE"):
        errors.append(f"generator must be CTGAN, TVAE, TabDDPM, or NONE, got '{gen}'.")

    n_rows = payload.get("n_synthetic_rows")
    if not isinstance(n_rows, int) or n_rows < 0:
        errors.append(f"n_synthetic_rows must be integer >= 0, got {n_rows}.")

    for flt_field in ("augmentation_ratio", "fidelity_score", "utility_score", "target_coverage", "prescription_confidence"):
        val = payload.get(flt_field)
        if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
            errors.append(f"{flt_field} must be float between 0.0 and 1.0, got {val}.")

    priv = payload.get("privacy_status")
    if priv not in ("PASS", "FAIL"):
        errors.append(f"privacy_status must be 'PASS' or 'FAIL', got '{priv}'.")

    for str_field in ("repair_dataset_uri", "repair_version", "repair_timestamp"):
        s_val = payload.get(str_field)
        if not isinstance(s_val, str) or not s_val.strip():
            errors.append(f"{str_field} must be a non-empty string.")

    return (len(errors) == 0), errors
