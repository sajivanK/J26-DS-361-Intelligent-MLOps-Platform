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
import math
from pathlib import Path
import re
import tempfile
import os
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[3]


class ContractValidationError(ValueError):
    """Raised when an incoming external contract fails schema, type, or range validation."""


class DependencyResolutionError(FileNotFoundError):
    """Raised when data_context references datasets or model artifacts that do not exist or are unauthorized."""


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
VALID_C4_GENERATORS = {"CTGAN", "TVAE", "TabDDPM", "NONE"}
VALID_C4_PRIVACY_STATUSES = {"PASS", "FAIL"}


# -----------------------------------------------------------------------------
# Numeric and Path Security Validators
# -----------------------------------------------------------------------------

def _check_finite_number(
    val: Any,
    field_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> tuple[bool, str | None, float | None]:
    """Validate that val is a finite number, strictly rejecting booleans, NaN, Inf, and non-numerics."""
    if isinstance(val, bool):
        return False, f"'{field_name}' must be a numeric value, but got boolean '{val}'.", None
    if not isinstance(val, (int, float)):
        return False, f"'{field_name}' must be a numeric value, got {type(val).__name__}.", None
    if not math.isfinite(val):
        return False, f"'{field_name}' must be a finite number, got non-finite value '{val}'.", None
    f_val = float(val)
    if min_val is not None and f_val < min_val:
        return False, f"'{field_name}' must be >= {min_val}, got {f_val}.", None
    if max_val is not None and f_val > max_val:
        return False, f"'{field_name}' must be <= {max_val}, got {f_val}.", None
    return True, None, f_val


def _check_finite_int(
    val: Any,
    field_name: str,
    min_val: int | None = None,
    max_val: int | None = None,
) -> tuple[bool, str | None, int | None]:
    """Validate that val is a finite integer, strictly rejecting booleans and floats."""
    if isinstance(val, bool):
        return False, f"'{field_name}' must be an integer, but got boolean '{val}'.", None
    if not isinstance(val, int):
        return False, f"'{field_name}' must be an integer, got {type(val).__name__}.", None
    if min_val is not None and val < min_val:
        return False, f"'{field_name}' must be >= {min_val}, got {val}.", None
    if max_val is not None and val > max_val:
        return False, f"'{field_name}' must be <= {max_val}, got {val}.", None
    return True, None, val


def get_authorized_directory_roots() -> dict[str, list[Path]]:
    """Return canonical paths for authorized dataset, model, and output directories."""
    roots = {
        "datasets": [
            (REPO_ROOT / "component3" / "data").resolve(),
            (REPO_ROOT / "shared").resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ],
        "models": [
            (REPO_ROOT / "component3" / "models").resolve(),
            (REPO_ROOT / "component3" / "results" / "trained_models").resolve(),
            (REPO_ROOT / "component3" / "results").resolve(),
            (REPO_ROOT / "shared").resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ],
        "outputs": [
            (REPO_ROOT / "component3" / "results").resolve(),
            (REPO_ROOT / "shared").resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ],
    }
    from component3.src.integration_storage import data_root, upload_root, run_root
    roots["datasets"].extend([data_root().resolve(), upload_root().resolve()])
    roots["models"].extend([data_root().resolve(), upload_root().resolve()])
    roots["outputs"].append(run_root().resolve())
    for category, variable in (("datasets", "C3_DATA_ROOTS"), ("models", "C3_MODEL_ROOTS")):
        roots[category].extend(Path(value).resolve() for value in os.environ.get(variable, "").split(os.pathsep) if value)
    return roots


def verify_path_authorization(path: Path | str, allowed_category: str = "datasets") -> tuple[bool, str | None, Path | None]:
    """Check that the resolved canonical path is strictly inside an authorized platform root.
    
    Guards against directory traversal ('..'), unauthorized absolute paths, and symlink escapes.
    """
    raw_p = Path(path)
    if ".." in raw_p.parts:
        return False, f"Path traversal ('..') is prohibited in path '{path}'.", None

    if not raw_p.is_absolute():
        resolved_p = (REPO_ROOT / raw_p).resolve()
    else:
        resolved_p = raw_p.resolve()

    allowed_roots = get_authorized_directory_roots().get(allowed_category, [])
    authorized = False
    for root in allowed_roots:
        try:
            resolved_p.relative_to(root)
            authorized = True
            break
        except ValueError:
            continue

    if not authorized:
        allowed_str = ", ".join(str(r) for r in allowed_roots)
        return False, f"Path '{path}' resolves to '{resolved_p}', which is outside authorized {allowed_category} directories: [{allowed_str}].", None

    return True, None, resolved_p


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
        """Return drop as a proportion [0.0, 1.0].
        
        Measurement Unit Convention:
        - Values in [0.0, 1.0] are interpreted as proportions (e.g., 0.18 = 18% accuracy drop).
        - Values in (1.0, 100.0] are interpreted as percentages and converted to proportions.
        """
        unit = self.raw_metadata.get("harm_unit")
        if unit == "percentage_points" or (unit is None and self.predicted_accuracy_drop > 1.0):
            return round(self.predicted_accuracy_drop / 100.0, 4)
        return round(self.predicted_accuracy_drop, 4)

    @property
    def confidence_interval_proportions(self) -> tuple[float, float]:
        """Return confidence interval as proportions [0.0, 1.0]."""
        low, high = self.confidence_interval
        unit = self.raw_metadata.get("harm_unit")
        if unit == "percentage_points" or (unit is None and (high > 1.0 or low > 1.0)):
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

    # Required fields: predicted_accuracy_drop OR predicted_harm
    raw_drop = data.get("predicted_accuracy_drop", data.get("predicted_harm"))
    drop_val: float | None = None
    if raw_drop is None:
        errors.append("c1_evidence missing required field: 'predicted_accuracy_drop' (or 'predicted_harm').")
    else:
        ok_drop, err_drop, drop_val = _check_finite_number(
            raw_drop, "predicted_accuracy_drop", min_val=-100.0, max_val=100.0
        )
        if not ok_drop and err_drop:
            errors.append(err_drop)

    # Confidence interval (accept confidence_interval or harm_interval)
    raw_ci = data.get("confidence_interval", data.get("harm_interval"))
    ci_tuple: tuple[float, float] | None = None
    if raw_ci is None:
        errors.append("c1_evidence missing required field: 'confidence_interval'.")
    elif not isinstance(raw_ci, (list, tuple)) or len(raw_ci) != 2:
        errors.append("confidence_interval must be a list or tuple of exactly two numbers [lower, upper].")
    else:
        low, high = raw_ci
        ok_low, err_low, low_val = _check_finite_number(low, "confidence_interval[0]", min_val=-100.0, max_val=100.0)
        ok_high, err_high, high_val = _check_finite_number(high, "confidence_interval[1]", min_val=-100.0, max_val=100.0)
        if not ok_low and err_low:
            errors.append(err_low)
        if not ok_high and err_high:
            errors.append(err_high)
        if ok_low and ok_high and low_val is not None and high_val is not None:
            if low_val > high_val:
                errors.append(f"confidence_interval lower bound ({low_val}) exceeds upper bound ({high_val}).")
            else:
                ci_tuple = (low_val, high_val)

    # Drift type
    raw_type = data.get("drift_type")
    if "marginal_extent" in data:
        extent = data["marginal_extent"]
        interaction = data.get("interaction_suspected", 0.0)
        valid_i, error_i, interaction = _check_finite_number(interaction, "interaction_suspected", 0.0, 1.0)
        if not valid_i:
            errors.append(error_i)
        if extent not in ("none", "single", "multi"):
            errors.append("marginal_extent must be none, single, or multi.")
        elif raw_type is None:
            has_interaction = valid_i and interaction >= 0.4
            raw_type = ("combined" if has_interaction else f"marginal_{extent}") if extent != "none" else ("interaction" if has_interaction else "none")
    elif "interaction_suspected" in data:
        ok_interaction, error_interaction, _ = _check_finite_number(data["interaction_suspected"], "interaction_suspected", 0.0, 1.0)
        if not ok_interaction:
            errors.append(error_interaction)
    if not raw_type or not isinstance(raw_type, str):
        errors.append("c1_evidence missing required string field: 'drift_type'.")
    elif raw_type.lower() not in VALID_C1_DRIFT_TYPES:
        errors.append(f"drift_type '{raw_type}' not in allowed values: {sorted(VALID_C1_DRIFT_TYPES)}.")

    # Drift severity
    raw_sev = data.get("harm_severity", data.get("drift_severity"))
    if not raw_sev or not isinstance(raw_sev, str):
        errors.append("c1_evidence missing required string field: 'drift_severity'.")
    elif raw_sev.lower() not in VALID_C1_DRIFT_SEVERITIES:
        errors.append(f"drift_severity '{raw_sev}' not in allowed values: {sorted(VALID_C1_DRIFT_SEVERITIES)}.")

    # Affected features
    raw_features = data.get("affected_features", data.get("features_drifted"))
    cleaned_features: list[str] = []
    if raw_features is None:
        errors.append("c1_evidence missing required field: 'affected_features'.")
    elif not isinstance(raw_features, (list, tuple)):
        errors.append("affected_features must be a list of feature name strings.")
    elif not all(isinstance(f, str) and f.strip() for f in raw_features):
        errors.append("All items in affected_features must be non-empty strings.")
    else:
        cleaned_features = [str(f).strip() for f in raw_features]
        if len(cleaned_features) != len(set(cleaned_features)):
            errors.append("affected_features cannot contain duplicate feature names.")

    # C1 v1.1 uses proportions, including signed uncertainty intervals.
    if "predicted_harm" in data:
        if drop_val is not None and not -1.0 <= drop_val <= 1.0:
            errors.append("predicted_harm must be a proportion between -1 and 1.")
        if ci_tuple is not None and not all(-1.0 <= v <= 1.0 for v in ci_tuple):
            errors.append("harm_interval must use proportions between -1 and 1.")
    if "drift_detected" in data and not isinstance(data["drift_detected"], bool):
        errors.append("drift_detected must be a boolean.")
    harm_unit = "proportion" if "predicted_harm" in data else data.get("harm_unit", "percentage_points")
    if harm_unit not in ("proportion", "percentage_points"):
        errors.append("harm_unit must be proportion or percentage_points.")

    if errors:
        return False, errors, None

    contract = C1EvidenceContract(
        predicted_accuracy_drop=float(drop_val),  # type: ignore[arg-type]
        confidence_interval=ci_tuple,  # type: ignore[arg-type]
        drift_type=str(raw_type).lower(),
        drift_severity=str(raw_sev).lower(),
        affected_features=tuple(cleaned_features),
        raw_metadata={**{k: v for k, v in data.items() if k not in {
            "predicted_accuracy_drop", "predicted_harm", "confidence_interval",
            "harm_interval", "drift_type", "drift_severity", "affected_features", "features_drifted"
        }}, "harm_unit": harm_unit},
    )
    return True, [], contract


def validate_c2_diagnosis(data: dict[str, Any]) -> tuple[bool, list[str], C2DiagnosisContract | None]:
    """Strictly validate c2_diagnosis dictionary against contract specification."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["c2_diagnosis must be a dictionary."], None

    # Required string fields
    rc_feature = data.get("root_cause_feature")
    if not rc_feature or not isinstance(rc_feature, str) or not rc_feature.strip():
        errors.append("c2_diagnosis missing required non-empty string field: 'root_cause_feature'.")

    rc_cat = data.get("root_cause_category")
    if not rc_cat or not isinstance(rc_cat, str):
        errors.append("c2_diagnosis missing required string field: 'root_cause_category'.")
    elif rc_cat.lower() not in VALID_C2_ROOT_CAUSE_CATEGORIES:
        errors.append(f"root_cause_category '{rc_cat}' not in allowed values: {sorted(VALID_C2_ROOT_CAUSE_CATEGORIES)}.")

    slice_rule = data.get("affected_slice_rule", data.get("slice_predicate", data.get("diagnosed_problematic_slice")))
    if not slice_rule or not isinstance(slice_rule, str) or not slice_rule.strip():
        errors.append("c2_diagnosis missing required non-empty string field: 'affected_slice_rule'.")

    ranked_feats = data.get("affected_features_ranked", data.get("affected_features"))
    cleaned_ranked: list[str] = []
    if ranked_feats is None:
        errors.append("c2_diagnosis missing required field: 'affected_features_ranked'.")
    elif not isinstance(ranked_feats, (list, tuple)):
        errors.append("affected_features_ranked must be a list of strings.")
    elif not all(isinstance(f, str) and f.strip() for f in ranked_feats):
        errors.append("All items in affected_features_ranked must be non-empty strings.")
    else:
        cleaned_ranked = [str(f).strip() for f in ranked_feats]
        if len(cleaned_ranked) != len(set(cleaned_ranked)):
            errors.append("affected_features_ranked cannot contain duplicate feature names.")

    val_ranges = data.get("affected_value_ranges")
    if val_ranges is not None and not isinstance(val_ranges, dict):
        errors.append("affected_value_ranges must be a dictionary.")

    # Strict numeric range validation [0.0, 1.0]
    ok_supp, err_supp, val_supp = _check_finite_number(data.get("slice_support"), "slice_support", min_val=0.0, max_val=1.0)
    if not ok_supp and err_supp:
        errors.append(err_supp)

    ok_stab, err_stab, val_stab = _check_finite_number(data.get("diagnosis_stability"), "diagnosis_stability", min_val=0.0, max_val=1.0)
    if not ok_stab and err_stab:
        errors.append(err_stab)

    ok_imp, err_imp, val_imp = _check_finite_number(data.get("drift_impact_score"), "drift_impact_score", min_val=0.0, max_val=1.0)
    if not ok_imp and err_imp:
        errors.append(err_imp)

    raw_temp = data.get("temporal_window", 1)
    ok_temp, err_temp, val_temp = _check_finite_int(raw_temp, "temporal_window", min_val=1, max_val=1000)
    if not ok_temp and err_temp:
        errors.append(err_temp)

    if errors:
        return False, errors, None

    contract = C2DiagnosisContract(
        root_cause_feature=str(rc_feature).strip(),
        root_cause_category=str(rc_cat).strip().lower(),
        affected_slice_rule=str(slice_rule).strip(),
        affected_features_ranked=tuple(cleaned_ranked),
        affected_value_ranges=dict(val_ranges or {}),
        slice_support=float(val_supp),  # type: ignore[arg-type]
        diagnosis_stability=float(val_stab),  # type: ignore[arg-type]
        drift_impact_score=float(val_imp),  # type: ignore[arg-type]
        temporal_window=int(val_temp),  # type: ignore[arg-type]
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

    lbl_status = data.get("label_status", "UNAVAILABLE")
    if not isinstance(lbl_status, str) or lbl_status.upper() not in VALID_LABEL_STATUSES:
        errors.append(f"label_status must be one of {sorted(VALID_LABEL_STATUSES)}, got '{lbl_status}'.")

    # Path traversal and authorization checks on URIs
    for uri_field, cat in (
        ("reference_dataset_uri", "datasets"),
        ("current_dataset_uri", "datasets"),
        ("preprocessing_uri", "models"),
    ):
        val = data.get(uri_field)
        if val is not None and isinstance(val, str) and val.strip():
            ok_auth, err_auth, _ = verify_path_authorization(val, cat)
            if not ok_auth and err_auth:
                errors.append(f"{uri_field}: {err_auth}")

    # Numeric budget and tolerance validation
    raw_budget = data.get("adaptation_budget_size", 200)
    ok_b, err_b, val_budget = _check_finite_int(raw_budget, "adaptation_budget_size", min_val=1, max_val=50000)
    if not ok_b and err_b:
        errors.append(err_b)

    raw_tol = data.get("source_retention_tolerance", 0.05)
    ok_t, err_t, val_tol = _check_finite_number(raw_tol, "source_retention_tolerance", min_val=0.0, max_val=0.50)
    if not ok_t and err_t:
        errors.append(err_t)

    if errors:
        return False, errors, None

    contract = DataContextContract(
        model_id=str(data["model_id"]).strip(),
        dataset_id=str(data["dataset_id"]).strip(),
        reference_dataset_uri=str(data["reference_dataset_uri"]).strip(),
        current_dataset_uri=str(data["current_dataset_uri"]).strip(),
        target_column=str(data["target_column"]).strip(),
        schema_version=str(data.get("schema_version", "1.0.0")).strip(),
        label_status=str(lbl_status).upper().strip(),
        preprocessing_uri=str(data.get("preprocessing_uri") or "").strip() or None,
        adaptation_budget_size=int(val_budget),  # type: ignore[arg-type]
        source_retention_tolerance=float(val_tol),  # type: ignore[arg-type]
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

    # Support upstream orchestration wrapper and section alias names
    norm_data = dict(data)
    if "upstream" in norm_data and isinstance(norm_data["upstream"], dict):
        for k, v in norm_data["upstream"].items():
            if k not in norm_data:
                norm_data[k] = v

    c1_raw = norm_data.get("c1_evidence") or norm_data.get("c1")
    c2_raw = norm_data.get("c2_diagnosis") or norm_data.get("c2_to_c3") or norm_data.get("c2")
    dc_raw = norm_data.get("data_context") or norm_data.get("context")

    c1_contract = None
    c2_contract = None
    dc_contract = None

    if c1_raw is None:
        all_errors.append("Missing required top-level section: 'c1_evidence' (or 'c1').")
    else:
        _, c1_errs, c1_contract = validate_c1_evidence(c1_raw)
        all_errors.extend(c1_errs)

    if c2_raw is None:
        all_errors.append("Missing required top-level section: 'c2_diagnosis' (or 'c2_to_c3').")
    else:
        _, c2_errs, c2_contract = validate_c2_diagnosis(c2_raw)
        all_errors.extend(c2_errs)

    if dc_raw is None:
        all_errors.append("Missing required top-level section: 'data_context' (or 'context').")
    else:
        # Allow data_context to inherit pass-through fields from c1 if omitted in context
        if isinstance(dc_raw, dict) and isinstance(c1_raw, dict):
            dc_copy = dict(dc_raw)
            for passthrough_key in ("model_id", "dataset_id", "reference_dataset_uri", "current_dataset_uri"):
                if passthrough_key not in dc_copy and passthrough_key in c1_raw:
                    dc_copy[passthrough_key] = c1_raw[passthrough_key]
            dc_raw = dc_copy

        _, dc_errs, dc_contract = validate_data_context(dc_raw)
        all_errors.extend(dc_errs)

    if all_errors:
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
    
    Guarantees that paths are strictly located in authorized platform directories
    and cannot escape via symlinks or directory traversal.
    
    Returns (success, missing_errors, resolved_paths_dict).
    """
    errors: list[str] = []
    resolved: dict[str, Path] = {}

    # 1. Reference dataset
    ok_ref, err_ref, ref_p = verify_path_authorization(data_context.reference_dataset_uri, "datasets")
    if not ok_ref and err_ref:
        errors.append(err_ref)
    elif ref_p is not None:
        if not ref_p.is_file():
            errors.append(
                f"Reference dataset not found at URI '{data_context.reference_dataset_uri}' "
                f"(resolved: {ref_p}). Component 3 will not silently substitute another dataset."
            )
        else:
            resolved["reference_dataset"] = ref_p

    # 2. Current dataset
    ok_cur, err_cur, cur_p = verify_path_authorization(data_context.current_dataset_uri, "datasets")
    if not ok_cur and err_cur:
        errors.append(err_cur)
    elif cur_p is not None:
        if not cur_p.is_file():
            errors.append(
                f"Current drifted dataset not found at URI '{data_context.current_dataset_uri}' "
                f"(resolved: {cur_p}). Component 3 will not silently substitute another dataset."
            )
        else:
            resolved["current_dataset"] = cur_p

    # 3. Preprocessor / script (optional but if provided, must exist and be authorized)
    if data_context.preprocessing_uri:
        ok_prep, err_prep, prep_p = verify_path_authorization(data_context.preprocessing_uri, "models")
        if not ok_prep and err_prep:
            errors.append(err_prep)
        elif prep_p is not None:
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
        "repair_strategy": {"type": "string", "enum": ["TARGETED_SYNTHETIC_REPAIR", "AUTHENTIC_WEIGHTED_ADAPTATION", "NO_REPAIR"]},
        "fidelity_score": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "utility_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "privacy_status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "target_coverage": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "prescription_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "repair_version": {"type": "string"},
        "repair_timestamp": {"type": "string"},
    },
    "additionalProperties": False,
    "allOf": [
        {"if": {"required": ["repair_strategy"],
                "properties": {"repair_strategy": {"const": "AUTHENTIC_WEIGHTED_ADAPTATION"}}},
         "then": {"properties": {"repair_package_status": {"const": "APPROVED"},
                                  "generator": {"const": "NONE"}, "n_synthetic_rows": {"const": 0},
                                  "augmentation_ratio": {"const": 0.0}, "fidelity_score": {"type": "null"},
                                  "target_coverage": {"type": "null"}, "privacy_status": {"const": "PASS"}}}},
        {"if": {"required": ["repair_strategy"],
                "properties": {"repair_strategy": {"const": "TARGETED_SYNTHETIC_REPAIR"}}},
         "then": {"properties": {"repair_package_status": {"const": "APPROVED"}},
                  "not": {"properties": {"generator": {"const": "NONE"}}}}},
        {"if": {"properties": {"fidelity_score": {"type": "null"}}},
         "then": {"required": ["repair_strategy"],
                  "properties": {"repair_strategy": {"const": "AUTHENTIC_WEIGHTED_ADAPTATION"}}}},
        {"if": {"properties": {"target_coverage": {"type": "null"}}},
         "then": {"required": ["repair_strategy"],
                  "properties": {"repair_strategy": {"const": "AUTHENTIC_WEIGHTED_ADAPTATION"}}}},
        {"if": {"properties": {"repair_package_status": {"const": "REJECTED"}}},
         "then": {"properties": {"n_synthetic_rows": {"const": 0}, "repair_dataset_uri": {"const": "NONE"},
                                  "repair_strategy": {"enum": ["NO_REPAIR"]}}}},
    ],
}


def validate_c4_flat_contract(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate flat 12-field dictionary against C4 schema requirements.
    
    Performs full Draft-7 JSON Schema validation against shared/contracts/c3_output_schema.json specification.
    """
    errors: list[str] = []
    
    # 1. Official JSON Schema validation
    try:
        jsonschema.validate(instance=payload, schema=C4_FLAT_CONTRACT_JSON_SCHEMA)
    except jsonschema.ValidationError as ve:
        prop = list(ve.path)[0] if ve.path else "payload"
        errors.append(f"JSONSchema validation error for '{prop}': {ve.message}")
    except Exception as e:
        errors.append(f"JSONSchema validator failure: {e}")

    # 2. Strict field existence check
    required_keys = C4_FLAT_CONTRACT_JSON_SCHEMA["required"]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        errors.append(f"Missing required fields: {missing}")

    return (len(errors) == 0), errors
