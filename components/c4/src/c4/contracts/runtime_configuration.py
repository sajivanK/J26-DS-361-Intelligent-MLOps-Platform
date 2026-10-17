"""Load trusted operator configuration; never infer missing runtime bindings."""
from dataclasses import dataclass, fields
from pathlib import Path
import re

from c4.contracts.runtime_compatibility import (
    PinnedFile, RuntimeBindings, _json, verify_runtime_compatibility,
)
from c4.contracts.platform_adapter import adapt_platform_upstream

FILE_FIELDS = (
    "champion_model", "champion_metadata", "preprocessing", "reference",
    "release_validation", "repair", "lineage",
)
BINDING_FIELDS = {field.name for field in fields(RuntimeBindings)}


@dataclass(frozen=True)
class RuntimeConfiguration:
    bindings: RuntimeBindings | None
    missing: tuple[str, ...]


def load_runtime_configuration(path: str | Path) -> RuntimeConfiguration:
    """Resolve relative paths against the config directory, not the shell cwd.

    Paths/hashes must be reviewed by the operator. Reading a file and inserting
    its current hash automatically is not verification against trusted evidence.
    """
    path = Path(path).resolve()
    config = _json(path.read_bytes())
    if set(config) != {"schema_version", "bindings"} or config["schema_version"] != "1.0":
        raise ValueError("Expected runtime configuration schema 1.0 and bindings only.")
    raw = config["bindings"]
    if not isinstance(raw, dict) or set(raw) - BINDING_FIELDS:
        raise ValueError("Unknown or invalid runtime binding fields.")
    values, missing = {}, []
    for name in sorted(BINDING_FIELDS):
        value = raw.get(name)
        if name in FILE_FIELDS:
            if value is None:
                missing.append(name)
                continue
            if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
                raise ValueError(f"{name}: expected path and sha256 only.")
            for key in ("path", "sha256"):
                if value[key] is None or value[key] == "":
                    missing.append(f"{name}.{key}")
                elif not isinstance(value[key], str) or not value[key].strip():
                    raise ValueError(f"{name}.{key}: expected nonempty string.")
            if value["sha256"] not in (None, "") and not re.fullmatch(
                    r"[0-9a-f]{64}", value["sha256"]):
                raise ValueError(f"{name}: invalid SHA256 digest.")
            if any(item.startswith(name + ".") for item in missing):
                continue
            resolved = Path(value["path"]).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            values[name] = PinnedFile(resolved.resolve(), value["sha256"])
        elif value is None or value == "" or value == []:
            missing.append(name)
        elif name == "feature_columns":
            if not isinstance(value, list) or not all(
                    isinstance(item, str) and item.strip() for item in value):
                raise ValueError("feature_columns: expected nonempty column names.")
            if len(set(value)) != len(value):
                raise ValueError("Duplicate feature columns.")
            values[name] = tuple(value)
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}: expected nonempty string.")
        else:
            values[name] = value
    return RuntimeConfiguration(
        None if missing else RuntimeBindings(**values), tuple(missing))


def preflight_runtime(config_path, upstream_path=None):
    """Return a readiness report, never a release-controller decision."""
    config = load_runtime_configuration(config_path)
    report = {
        "status": "CONFIGURATION_INCOMPLETE",
        "missing_bindings": list(config.missing),
        "training_executed": False,
        "future_outcomes_read": False,
        "model_runtime_validated": False,
        "forecast_applicable": False,
        "production_release_authorized": False,
    }
    if config.missing:
        return report
    if upstream_path is None:
        report["status"] = "UPSTREAM_INPUT_REQUIRED"
        return report
    upstream = _json(Path(upstream_path).read_bytes())
    adapted = adapt_platform_upstream(upstream)
    checked = verify_runtime_compatibility(adapted, config.bindings)
    report.update(
        status="BINDINGS_VERIFIED_OPERATIONAL_READINESS_PENDING",
        artifact_bindings_verified=checked.artifact_bindings_verified,
        declared_lineage_checked=checked.declared_lineage_checked,
        row_counts=checked.row_counts,
        forecast_reason=checked.forecast_reason,
    )
    return report
