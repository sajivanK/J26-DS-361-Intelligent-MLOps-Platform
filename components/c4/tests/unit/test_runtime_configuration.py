"""Configuration tests use temporary files, not benchmark artifacts."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from c4.contracts.runtime_configuration import (
    FILE_FIELDS, load_runtime_configuration, preflight_runtime,
)


def configuration():
    return {"schema_version": "1.0", "bindings": {
        "dataset": "uci", "champion_model_id": "operator-bound-model",
        **{name: {"path": name + ".bin", "sha256": "a" * 64}
           for name in FILE_FIELDS},
        "feature_columns": ["age"], "target_column": "target",
        "rule_feature_space": "processed"}}


def write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_paths_resolved_from_configuration_directory(tmp_path):
    result = load_runtime_configuration(write(tmp_path, configuration()))
    assert not result.missing
    assert result.bindings.repair.path == (tmp_path / "repair.bin").resolve()
    # Loading configuration does not open its declared data/model files.
    assert not result.bindings.repair.path.exists()


def test_incomplete_preflight_does_not_read_upstream(tmp_path):
    data = configuration(); data["bindings"]["repair"] = None
    result = preflight_runtime(write(tmp_path, data), tmp_path / "nonexistent.json")
    assert result["status"] == "CONFIGURATION_INCOMPLETE"
    assert result["missing_bindings"] == ["repair"]
    assert not result["training_executed"]
    assert not result["production_release_authorized"]


def test_upstream_required(tmp_path):
    assert preflight_runtime(write(tmp_path, configuration()))["status"] == "UPSTREAM_INPUT_REQUIRED"


@pytest.mark.parametrize("field", list(FILE_FIELDS))
def test_missing_pinned_file_fields(tmp_path, field):
    data = configuration(); data["bindings"][field] = {"path": None, "sha256": None}
    assert set(load_runtime_configuration(write(tmp_path, data)).missing) == {
        field + ".path", field + ".sha256"}


@pytest.mark.parametrize("change", ["version", "unknown", "bad_hash", "bad_columns", "extra_pin"])
def test_invalid_configuration_rejected(tmp_path, change):
    data = configuration()
    if change == "version": data["schema_version"] = "2"
    if change == "unknown": data["bindings"]["override_safety"] = True
    if change == "bad_hash": data["bindings"]["repair"]["sha256"] = "invented"
    if change == "bad_columns": data["bindings"]["feature_columns"] = ["age", "age"]
    if change == "extra_pin": data["bindings"]["repair"]["trust"] = True
    with pytest.raises(ValueError): load_runtime_configuration(write(tmp_path, data))


def test_preflight_calls_adapter_and_checker(monkeypatch, tmp_path):
    import c4.contracts.runtime_configuration as module
    from types import SimpleNamespace
    upstream = tmp_path / "upstream.json"; upstream.write_text('{"source":"fixture"}')
    adapted = object(); seen = []
    def adapt(raw):
        assert raw == {"source": "fixture"}; seen.append("adapt"); return adapted
    def check(actual, bindings):
        assert actual is adapted
        assert bindings.champion_model_id == "operator-bound-model"
        seen.append("check")
        return SimpleNamespace(artifact_bindings_verified=True, declared_lineage_checked=True,
                               row_counts={"repair": 1}, forecast_reason="unsupported target")
    monkeypatch.setattr(module, "adapt_platform_upstream", adapt)
    monkeypatch.setattr(module, "verify_runtime_compatibility", check)
    out = preflight_runtime(write(tmp_path, configuration()), upstream)
    assert seen == ["adapt", "check"]
    assert out["status"] == "BINDINGS_VERIFIED_OPERATIONAL_READINESS_PENDING"
    assert not out["forecast_applicable"]
    assert not out["production_release_authorized"]


def test_cli_incomplete_exit_code(tmp_path):
    data = configuration(); data["bindings"]["champion_model_id"] = None
    script = Path(__file__).resolve().parents[2] / "scripts/check_runtime_preflight.py"
    result = subprocess.run([sys.executable, str(script), "--config",
        str(write(tmp_path, data))], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["status"] == "CONFIGURATION_INCOMPLETE"
