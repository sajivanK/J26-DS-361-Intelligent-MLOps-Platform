"""Service sequencing tests with injected adapters/configuration/evaluation/storage."""
from types import SimpleNamespace
import pytest
import c4.api.runtime_service as module


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    calls = []
    adapted = SimpleNamespace(context=SimpleNamespace(run_id="run1"), prerequisite_failures=())
    bindings = object()
    config = SimpleNamespace(missing=(), bindings=bindings)
    def adapt(raw): calls.append("adapt"); return adapted
    def load(path): calls.append("config"); return config
    def evaluate(actual, bound):
        assert actual is adapted and bound is bindings
        calls.append("evaluate")
        return SimpleNamespace(report={"candidate_model_sha256": "a" * 64,
            "pre_release_evaluation": {"comparisons": {}}, "forecast_reason": "unsupported"})
    def persist(result, root, version):
        calls.append("persist")
        return SimpleNamespace(directory=tmp_path / version, manifest_sha256="b" * 64)
    monkeypatch.setattr(module, "adapt_platform_upstream", adapt)
    monkeypatch.setattr(module, "load_runtime_configuration", load)
    monkeypatch.setattr(module, "_evaluate", evaluate)
    monkeypatch.setattr(module, "_persist", persist)
    return adapted, config, calls


def invoke(tmp_path, **kwargs):
    return module.prepare_runtime_candidate(config_path="operator.json", upstream={},
        data_root=tmp_path, version="new-1", **kwargs)


def test_plan_never_trains_or_writes(prepared, tmp_path):
    _, _, calls = prepared
    out = invoke(tmp_path)
    assert calls == ["adapt", "config"]
    assert out["status"] == "PLAN_ONLY"
    assert not out["candidate_training_executed"]
    assert not list(tmp_path.iterdir())


def test_success_preserves_no_release_state(prepared, tmp_path):
    _, _, calls = prepared
    out = invoke(tmp_path, execute_training=True)
    assert calls == ["adapt", "config", "evaluate", "persist"]
    assert out["status"] == "CANDIDATE_SAVED_RELEASE_PENDING"
    assert out["artifact_persistence_completed"]
    assert out["manifest_sha256"] == "b" * 64
    assert out["release_decision"] is None
    assert not out["production_release_authorized"]
    assert not out["release_controller_executed"]
    assert not out["N1_executed"] and not out["N2_executed"]


def test_incomplete_configuration_blocks_execution(prepared, tmp_path):
    _, config, calls = prepared; config.missing = ("preprocessing",)
    out = invoke(tmp_path, execute_training=True)
    assert out["status"] == "CONFIGURATION_INCOMPLETE"
    assert calls == ["adapt", "config"]


def test_failed_upstream_blocks_execution(prepared, tmp_path):
    adapted, _, calls = prepared; adapted.prerequisite_failures = ("privacy_check",)
    out = invoke(tmp_path, execute_training=True)
    assert out["status"] == "UPSTREAM_PREREQUISITES_FAILED"
    assert calls == ["adapt", "config"]


@pytest.mark.parametrize("stage", ["_evaluate", "_persist"])
def test_failures_propagate_without_fake_decision(prepared, tmp_path, monkeypatch, stage):
    def fail(*a): raise ValueError("injected failure")
    monkeypatch.setattr(module, stage, fail)
    with pytest.raises(ValueError, match="injected failure"):
        invoke(tmp_path, execute_training=True)


@pytest.mark.parametrize("version", ["../escape", "CON", "bad.", "a/b"])
def test_bad_version_rejected_before_training(prepared, tmp_path, version):
    _, _, calls = prepared
    with pytest.raises(ValueError):
        module.prepare_runtime_candidate(config_path="x", upstream={}, execute_training=True,
            data_root=tmp_path, version=version)
    assert "evaluate" not in calls


def test_existing_version_rejected_before_training(prepared, tmp_path):
    _, _, calls = prepared
    (tmp_path / "c4_artifacts/operational_candidates/adult/new-1").mkdir(parents=True)
    with pytest.raises(FileExistsError): invoke(tmp_path, execute_training=True)
    assert "evaluate" not in calls


def test_boolean_opt_in_required(prepared, tmp_path):
    _, _, calls = prepared
    with pytest.raises(ValueError): invoke(tmp_path, execute_training="true")
    assert not calls
