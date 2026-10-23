"""C3-owned regression coverage for the team's integration guide."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from component3.tests.test_live_repair import supplied, IndependentSampler
from component3.src import live_repair as live
from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule
from component3.src.contracts.external_contracts import ContractValidationError, verify_path_authorization

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp/backend"))
from app.components.c3 import C3Component
from app import contracts, orchestrator


def test_backend_imports_without_c3_libraries():
    code = '''
import importlib.abc, sys
class WithoutC3(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ctgan', 'torch', 'jsonschema', 'matplotlib', 'psutil'}:
            raise ImportError('simulated absent C3 dependency: ' + fullname)
sys.meta_path.insert(0, WithoutC3())
sys.path.insert(0, 'webapp/backend')
from app.main import app
from app.components import BY_NAME
component = BY_NAME['c3']
component.safe_load()
assert component.loaded and component.source == 'fake'
assert component.details()['dependency_error']
assert component.run(None, {})['c3']['generator'] == 'NONE'
print('startup fallback verified')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("upstream, reason", [
    ({"context": {"dataset": "ieee"}}, "C2 diagnosis"),
    ({"component_sources": {"c2": "fake"}}, "C2 diagnosis"),
    ({"context": {"model_id": "unknown"}}, "onboarding context"),
])
def test_expected_cases_are_contract_valid_rejections(upstream, reason):
    component = C3Component()
    output = component.run(None, upstream)["c3"]
    assert not contracts.validate("c3", output)
    assert output["generator"] == "NONE"
    assert output["repair_dataset_uri"] == "NONE"
    assert output["n_synthetic_rows"] == 0
    assert reason in output["reason"]
    assert component.details()["fallback_runs"] == 1


def test_missing_labels_blocks_platform(supplied):
    payload, root = supplied
    payload["data_context"]["label_status"] = "UNAVAILABLE"
    output = C3Component().run(None, {"context": payload["data_context"]})["c3"]
    assert output["repair_package_status"] == "REJECTED"
    assert "waiting" in output["reason"]


def test_scaled_and_encoded_slice_values_are_used_unchanged():
    frame = pd.DataFrame({"age": [-1., -.5, .1, .8], "marital_status": [2, 2, 1, 2]})
    before = frame.copy(deep=True)
    selected = evaluate_slice_rule(frame, "age > -0.703 and marital_status == 2")
    assert selected.tolist() == [False, True, False, True]
    pd.testing.assert_frame_equal(frame, before)


def test_model_ready_rejects_double_preprocessing(supplied):
    payload, root = supplied
    payload["data_context"]["preprocessing_uri"] = payload["data_context"]["model_uri"]
    with pytest.raises(ContractValidationError, match="already transformed"):
        live.execute_live_repair(payload, root / "out")


def test_budget_targets_narrow_affected_group(supplied):
    payload, root = supplied
    current = pd.read_csv(root / "current.csv")
    payload["c2_diagnosis"]["affected_slice_rule"] = "f0 > 1.0"
    payload["c2_diagnosis"]["slice_support"] = float(current.f0.gt(1.).mean())
    payload["data_context"]["adaptation_budget_size"] = 40
    sampler = IndependentSampler()
    package, _ = live.execute_live_repair(payload, root / "out", generator_selector=sampler)
    assert sampler.seen is not None, package.rejection_reasons
    assert len(sampler.seen["training_df"]) == 40
    assert sampler.seen["training_df"].f0.gt(1.).all()


def test_unlabelled_batch_uses_only_supplied_labelled_samples(supplied):
    payload, root = supplied
    current = pd.read_csv(root / "current.csv")
    adaptation, evaluation = live._split(current, "outcome", 43)
    adaptation = adaptation[adaptation.f0 > -.5].head(100)
    adaptation.to_csv(root / "labels.csv", index=False)
    evaluation.to_csv(root / "eval.csv", index=False)
    current.drop(columns="outcome").to_csv(root / "current.csv", index=False)
    context = payload["data_context"]
    context.update(adaptation_dataset_uri=str(root / "labels.csv"), evaluation_dataset_uri=str(root / "eval.csv"), adaptation_budget_size=100)
    sampler = IndependentSampler()
    package, _ = live.execute_live_repair(payload, root / "out", generator_selector=sampler)
    assert sampler.seen is not None, package.rejection_reasons
    assert len(sampler.seen["training_df"]) == 100
    assert package.downstream_handoff_metadata["labelled_current_adaptation_rows_used"] == 100
    assert package.downstream_handoff_metadata["labelled_target_evaluation_rows_used"] == len(evaluation)


def test_unlabelled_sample_requires_independent_evaluation(supplied):
    payload, root = supplied
    payload["data_context"]["adaptation_dataset_uri"] = str(root / "current.csv")
    with pytest.raises(ContractValidationError, match="independent evaluation"):
        live.execute_live_repair(payload, root / "out")


@pytest.mark.parametrize("problem", ["overlap", "outside_batch"])
def test_sample_labels_cannot_leak_or_substitute_the_current_batch(supplied, problem):
    payload, root = supplied
    current = pd.read_csv(root / "current.csv")
    current.head(100).to_csv(root / "labels.csv", index=False)
    holdout = current.head(100) if problem == "overlap" else current.tail(150)
    holdout.to_csv(root / "eval.csv", index=False)
    if problem == "outside_batch":
        labels = current.head(100).copy()
        labels.f0 += 100
        labels.to_csv(root / "labels.csv", index=False)
    payload["data_context"].update(adaptation_dataset_uri=str(root / "labels.csv"), evaluation_dataset_uri=str(root / "eval.csv"))
    with pytest.raises(ContractValidationError, match="Overlapping|actual monitored"):
        live.execute_live_repair(payload, root / "out")


def test_external_storage_roots_work_without_extra_allowlists(tmp_path, monkeypatch):
    monkeypatch.setenv("J26_DATA_DIR", str(tmp_path / "team-data"))
    monkeypatch.setenv("C3_UPLOAD_DIR", str(tmp_path / "batches"))
    monkeypatch.delenv("C3_DATA_ROOTS", raising=False)
    monkeypatch.delenv("C3_MODEL_ROOTS", raising=False)
    monkeypatch.delenv("C3_RUNS_DIR", raising=False)
    assert verify_path_authorization(tmp_path / "batches/new.csv", "datasets")[0]
    assert verify_path_authorization(tmp_path / "team-data/c2/model.pkl", "models")[0]
    assert verify_path_authorization(tmp_path / "team-data/platform_runs/c3_repairs", "outputs")[0]


def test_ieee_rejection_allows_pipeline_to_reach_c4(monkeypatch):
    class Upstream:
        name, title, source, loaded, produces = "c2", "fixture", "fake", True, ()
        def run(self, batch, upstream): return {}
    class Downstream:
        name, title, source, loaded, produces = "c4", "fixture", "fake", True, ()
        def run(self, batch, upstream):
            assert upstream["c3"]["repair_package_status"] == "REJECTED"
            return {}
    component = C3Component()
    component.loaded = True
    monkeypatch.setattr(orchestrator, "PIPELINE", [Upstream(), component, Downstream()])
    result = orchestrator.run_pipeline(None, {"dataset": "ieee"})
    assert result["status"] == "completed"
    assert len(result["steps"]) == 3


def test_configuration_does_not_overwrite_existing_models(tmp_path):
    from component3.src.configure_team_context import configure
    template = json.loads((ROOT / "component3/docs/integration_review/c3_contexts.example.json").read_text())
    columns = template["adult-xgb-v1"]["feature_columns"] + ["target"]
    (tmp_path / "uci").mkdir()
    (tmp_path / "c2").mkdir()
    model = tmp_path / "c2/base_model_adult.pkl"
    model.write_bytes(b"existing team artifact")
    for name in ("reference.csv", "test.csv"):
        pd.DataFrame(columns=columns).to_csv(tmp_path / "uci" / name, index=False)
    destination = configure(tmp_path, tmp_path / "c3/context.json")
    value = json.loads(destination.read_text())["adult-xgb-v1"]
    assert value["input_representation"] == "model_ready"
    assert value["source_retention_tolerance"] == .05
    assert "preprocessing_uri" not in value
    assert model.read_bytes() == b"existing team artifact"


def test_invalid_c3_onboarding_cannot_break_shared_ingestion(tmp_path, monkeypatch):
    from app.settings import c3_data_context
    path = tmp_path / "invalid-context.json"
    path.write_text("{not valid json}")
    monkeypatch.setenv("C3_CONTEXTS_FILE", str(path))
    assert c3_data_context("adult-xgb-v1") == {}


def test_team_model_identity_has_no_experimental_default():
    from component3.src.models.model_registry import get_model_registry
    assert not get_model_registry().has("adult-xgb-v1")
