"""Read-only artifact binding checks before operational candidate training.

Configuration is trusted operator input, never copied from an HTTP request.
This module does not deserialize models, fit preprocessors, train candidates,
or certify that producer-declared source-row identities are truthful.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from c4.contracts.platform_adapter import AdaptedPlatformInput


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str

    def read(self) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Expected a lowercase SHA256 digest.")
        contents = Path(self.path).read_bytes()
        if hashlib.sha256(contents).hexdigest() != self.sha256:
            raise ValueError(f"Artifact checksum mismatch: {self.path}")
        return contents


@dataclass(frozen=True)
class RuntimeBindings:
    dataset: str
    champion_model_id: str
    champion_model: PinnedFile
    champion_metadata: PinnedFile
    preprocessing: PinnedFile
    reference: PinnedFile
    release_validation: PinnedFile
    repair: PinnedFile
    lineage: PinnedFile
    feature_columns: tuple[str, ...]
    target_column: str
    rule_feature_space: str


@dataclass(frozen=True)
class CompatibilityResult:
    verified_bytes: dict[str, bytes]
    row_counts: dict[str, int]
    artifact_bindings_verified: bool
    declared_lineage_checked: bool
    model_runtime_validated: bool = False
    forecast_applicable: bool = False
    forecast_reason: str = (
        "Frozen pilot targets true injected-slice accuracy margins; "
        "operational diagnosed-slice forecasting is not validated."
    )
    production_release_authorized: bool = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(contents):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")
    def unique_object(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON object key.")
            result[key] = value
        return result
    value = json.loads(contents.decode("utf-8-sig"),
                       parse_constant=reject_constant, object_pairs_hook=unique_object)
    _require(isinstance(value, dict), "Metadata must be a JSON object.")
    return value


def _csv_rows(contents, columns, target):
    reader = csv.reader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
    header = next(reader, [])
    _require(len(header) == len(columns) and len(set(header)) == len(header)
             and set(header) == set(columns), "Invalid CSV feature/target schema.")
    rows = list(reader)
    _require(bool(rows), "Labelled data must contain rows.")
    target_index = header.index(target)
    labels = set()
    for row in rows:
        _require(len(row) == len(header), "Malformed CSV row.")
        try:
            numbers = [float(value) for value in row]
        except ValueError as error:
            raise ValueError("CSV values must be numeric.") from error
        _require(all(math.isfinite(value) for value in numbers), "Nonfinite CSV value.")
        _require(numbers[target_index] in (0, 1), "Targets must be binary.")
        labels.add(numbers[target_index])
    return len(rows), labels


def verify_runtime_compatibility(
    adapted: AdaptedPlatformInput, bindings: RuntimeBindings,
) -> CompatibilityResult:
    """Check pinned bytes and lineage declarations without executing models.

    The pinned lineage manifest must contain:
      preprocessing_sha256; reference/release_validation/repair_csv_sha256;
      reference_row_ids; release_validation_row_ids; repair_source_row_ids;
      repair_source_role='dedicated_repair';
      repair_excluded_release_validation=true; repair_excluded_future=true.
    IDs must be globally source-qualified, e.g. 'adult-source-v1:row:123'.
    These declarations must be established upstream from real provenance;
    inventing them to pass this check does not establish data separation.
    """
    _require(not adapted.prerequisite_failures, "Upstream prerequisites failed.")
    _require(bindings.dataset == "uci" and adapted.context.dataset == bindings.dataset,
             "Only the explicitly bound Adult dataset is supported.")
    _require(adapted.context.model_id == bindings.champion_model_id,
             "Configured champion model ID differs from platform input.")
    _require(bindings.rule_feature_space == "processed", "Verified rule translation required.")
    features = bindings.feature_columns
    _require(isinstance(features, tuple) and bool(features)
             and all(isinstance(c, str) and bool(c) for c in features)
             and len(set(features)) == len(features)
             and isinstance(bindings.target_column, str) and bool(bindings.target_column)
             and bindings.target_column not in features, "Invalid configured schema.")
    # Keep only bytes read and hashed together; later stages should consume these
    # snapshots instead of reopening mutable paths after verification.
    names = ("champion_model", "champion_metadata", "preprocessing", "reference",
             "release_validation", "repair", "lineage")
    contents = {name: getattr(bindings, name).read() for name in names}
    metadata = _json(contents["champion_metadata"])
    _require(metadata.get("role") == "champion" and metadata.get("dataset") == "uci_adult",
             "Wrong champion role/dataset.")
    _require(metadata.get("feature_columns") == list(features)
             and metadata.get("target_column") == bindings.target_column,
             "Champion metadata schema differs.")
    _require(metadata.get("model_sha256") == bindings.champion_model.sha256,
             "Champion model link differs.")
    _require(metadata.get("reference_csv_sha256") == bindings.reference.sha256,
             "Champion reference training link differs.")
    package = adapted.inputs.c3_repair_package
    _require(Path(package.repair_dataset_uri).is_absolute(),
             "Use an explicitly resolved absolute C3 repair path for this check.")
    _require(Path(package.repair_dataset_uri).resolve() == Path(bindings.repair.path).resolve(),
             "C3 repair dataset path differs from configured binding.")
    counts = {}
    for name in ("reference", "release_validation", "repair"):
        counts[name], labels = _csv_rows(contents[name], (*features, bindings.target_column),
                                        bindings.target_column)
        if name == "reference":
            _require(labels == {0, 1}, "Reference training requires both classes.")
    _require(counts["repair"] == package.n_synthetic_rows and counts["repair"] > 0,
             "C3 repair row count differs.")
    actual_ratio = counts["repair"] / counts["reference"]
    _require(math.isclose(package.augmentation_ratio, actual_ratio,
                         rel_tol=1e-6, abs_tol=1e-8), "C3 augmentation ratio differs.")
    lineage = _json(contents["lineage"])
    for name in ("reference", "release_validation", "repair"):
        _require(lineage.get(name + "_csv_sha256") == getattr(bindings, name).sha256,
                 "Lineage data checksum link differs.")
    _require(lineage.get("preprocessing_sha256") == bindings.preprocessing.sha256,
             "Preprocessing lineage link differs.")
    groups = []
    for name in ("reference", "release_validation", "repair_source"):
        ids = lineage.get(name + "_row_ids")
        _require(isinstance(ids, list) and bool(ids)
                 and all(isinstance(i, str) and bool(i.strip()) for i in ids),
                 "Missing source-qualified lineage row IDs.")
        _require(len(set(ids)) == len(ids), "Duplicate lineage row IDs.")
        if name in counts:
            _require(len(ids) == counts[name], "Lineage row count differs.")
        groups.append(set(ids))
    _require(all(not groups[i] & groups[j] for i in range(3) for j in range(i)),
             "Training, repair-source and release-validation lineage overlaps.")
    _require(lineage.get("repair_source_role") == "dedicated_repair"
             and lineage.get("repair_excluded_release_validation") is True
             and lineage.get("repair_excluded_future") is True,
             "Repair generation exclusion declarations missing.")
    return CompatibilityResult(contents, counts, True, True)
