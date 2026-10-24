"""Orchestration boundary tests with synthetic frames and injected model doubles.

These tests do not constitute an actual upstream or trained-model integration run.
"""
from dataclasses import replace
import io
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
import c4.models.runtime_candidate as module


class Model:
    classes_ = np.array([0, 1])
    def get_booster(self): return SimpleNamespace(feature_names=list(FEATURE_COLUMNS))
    def predict(self, features): return (features["age"].to_numpy() > .5).astype(int)
    def predict_proba(self, features):
        positive = np.where(features["age"].to_numpy() > .5, .8, .2)
        return np.column_stack([1 - positive, positive])


@pytest.fixture
def prepared(monkeypatch):
    calls = []
    def csv(ages, targets):
        frame = pd.DataFrame({column: ages for column in FEATURE_COLUMNS})
        frame[TARGET_COLUMN] = targets
        return frame.to_csv(index=False).encode()
    snapshots = {
        "champion_metadata": json.dumps({"training_config": {"n_estimators": 2}}).encode(),
        "champion_model": b"champion",
        "reference": csv([.1, .8], [0, 1]),
        "repair": csv([.7], [1]),
        "release_validation": csv([.2, .9], [0, 1]),
        "lineage": json.dumps({"release_validation_row_ids": ["source:20", "source:21"]}).encode(),
    }
    bindings = SimpleNamespace(feature_columns=tuple(FEATURE_COLUMNS),
                               target_column=TARGET_COLUMN, rule_feature_space="processed")
    adapted = SimpleNamespace(context=SimpleNamespace(run_id="r1", dataset="uci", model_id="m1"),
        inputs=SimpleNamespace(c2_diagnosis=object()), audit_envelope={"source": "test"})
    def verify(actual, bound):
        assert actual is adapted and bound is bindings
        calls.append("verify")
        return SimpleNamespace(verified_bytes=snapshots, forecast_reason="pilot target incompatible")
    def load(contents):
        calls.append("load:" + contents.decode()); return Model()
    def train(frame, config):
        calls.append("train")
        expected = pd.concat([pd.read_csv(io.BytesIO(snapshots["reference"])),
                              pd.read_csv(io.BytesIO(snapshots["repair"]))], ignore_index=True)
        pd.testing.assert_frame_equal(frame, expected)
        assert len(frame) == 3 and config.n_estimators == 2
        return Model()
    monkeypatch.setattr(module, "verify_runtime_compatibility", verify)
    monkeypatch.setattr(module, "_load_model", load)
    monkeypatch.setattr(module, "_model_bytes", lambda model: b"candidate")
    monkeypatch.setattr(module, "train_champion", train)
    monkeypatch.setattr(module, "diagnosed_slice_mask", lambda frame, *args, **kwargs:
                        pd.Series([False, True], index=frame.index, dtype=bool))
    return adapted, bindings, calls, snapshots


def test_evaluation_uses_training_rows_only_and_never_authorizes_release(prepared):
    adapted, bindings, calls, _ = prepared
    result = module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
    assert calls == ["verify", "load:champion", "train", "load:candidate"]
    assert result.candidate_model_bytes == b"candidate"
    report = result.report
    assert report["candidate_reload_verified"]
    assert report["training_rows"] == 3 and report["validation_rows"] == 2
    assert report["pre_release_evaluation"]["comparisons"]["affected_slice"]["rows"] == 1
    for key in ["production_release_authorized", "forecast_applicable", "N1_executed",
                "N2_executed", "future_outcomes_read", "validation_used_for_training",
                "artifact_persistence_completed"]:
        assert report[key] is False
    assert "decision" not in report


def test_requires_explicit_training_opt_in(prepared):
    adapted, bindings, calls, _ = prepared
    with pytest.raises(ValueError, match="execute_training"):
        module.evaluate_runtime_candidate(adapted, bindings)
    assert calls == []


def test_schema_mismatch_stops_before_artifact_access(prepared):
    adapted, bindings, calls, _ = prepared
    bindings.feature_columns = ("wrong",)
    with pytest.raises(ValueError, match="schema"):
        module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
    assert calls == []


def test_compatibility_failure_stops_training(prepared, monkeypatch):
    adapted, bindings, calls, _ = prepared
    def fail(*args): raise ValueError("checksum mismatch")
    monkeypatch.setattr(module, "verify_runtime_compatibility", fail)
    with pytest.raises(ValueError, match="checksum"):
        module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
    assert calls == []


@pytest.mark.parametrize("values", [[True, True], [False, False], [True, None], [0, 1]])
def test_invalid_population_mask_stops_before_training(prepared, monkeypatch, values):
    adapted, bindings, calls, _ = prepared
    monkeypatch.setattr(module, "diagnosed_slice_mask", lambda frame, *a, **k:
                        pd.Series(values, index=frame.index))
    with pytest.raises(ValueError):
        module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
    assert "train" not in calls


def test_unsupported_slice_rule_stops_before_training(prepared, monkeypatch):
    adapted, bindings, calls, _ = prepared
    def fail(*a, **k): raise ValueError("Unsupported rule")
    monkeypatch.setattr(module, "diagnosed_slice_mask", fail)
    with pytest.raises(ValueError, match="Unsupported"):
        module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
    assert "train" not in calls


def test_invalid_champion_outputs_stop_before_training(prepared, monkeypatch):
    adapted, bindings, calls, _ = prepared
    class Invalid(Model):
        def predict_proba(self, features): return np.ones((len(features), 2))
    monkeypatch.setattr(module, "_load_model", lambda contents: Invalid())
    with pytest.raises(ValueError, match="probabilities"):
        module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
    assert "train" not in calls


def test_reload_difference_rejected(prepared, monkeypatch):
    adapted, bindings, calls, _ = prepared
    class Changed(Model):
        def predict_proba(self, features): return np.full((len(features), 2), .5)
    monkeypatch.setattr(module, "_load_model", lambda contents:
                        Changed() if contents == b"candidate" else Model())
    with pytest.raises(ValueError, match="replay"):
        module.evaluate_runtime_candidate(adapted, bindings, execute_training=True)
