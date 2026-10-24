"""Technical forecast/evidence freeze checks, not empirical safety evidence."""
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import _hybrid_policy_forecasts as q
import run_hybrid_policy_forecast_freeze as r


@pytest.fixture
def tmp_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("c4forecast")
    if os.name == "nt":
        raw = str(directory.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\UNC\\"+raw[2:] if raw.startswith("\\\\") else "\\\\?\\"+raw
        directory = Path(raw)
    yield directory
    if os.name == "nt" and directory.exists():
        shutil.rmtree(directory)


def save_doc(directory, stem, doc):
    directory.mkdir(parents=True, exist_ok=True)
    content = q.e.encoded(doc)
    (directory / (stem+".json")).write_bytes(content)
    (directory / (stem+".sha256")).write_text(q.d.sha(content)+"\n")
    return q.d.sha(content)


@pytest.fixture
def row():
    original = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    manifest = q.v.create_design(original)
    return next(s for s in manifest["scenarios"] if s["split_role"] == "policy_calibration")


class FixtureModel:
    features = ["a", "b"]

    def predict(self, x, raw=False):
        values = np.array([[-.01, .02, .01]*3]*len(x))
        if not raw:
            values = np.column_stack([q.e.ordered(values[:, j*3:(j+1)*3]) for j in range(3)])
        names = [t+"__"+label for t in q.d.TARGETS for _, label in q.d.QUANTILES]
        return pd.DataFrame(values, index=x.index, columns=names)


def test_evidence_freezer_exact_scientific_reuse():
    original = (C4 / "scripts/freeze_portable_training_case.py").read_text(encoding="utf-8-sig")
    adapted = (C4 / "scripts/freeze_hybrid_policy_case.py").read_text(encoding="utf-8-sig")
    assert q.expected_freezer_source(original) == adapted
    assert q.SCOPE in adapted
    assert "adult-simulation-pp1-v2" not in adapted
    assert "upstream = UpstreamInput.model_validate" in adapted


def test_feature_order_and_forecast_postprocessing():
    x = q.make_feature_frame({"b": .2, "a": .1}, ["a", "b"], "s1")
    assert list(x) == ["a", "b"] and x.iloc[0].tolist() == [.1, .2]
    output = q.forecast_values({q.PRIMARY: FixtureModel()}, x, states({q.PRIMARY: None}))
    target = q.d.TARGETS[0]
    assert output[q.PRIMARY]["raw"][target+"__median"] == .02
    assert output[q.PRIMARY]["processed"][target+"__median"] == .01


@pytest.mark.parametrize("change", ["missing", "extra", "nan", "string", "duplicate_column"])
def test_invalid_features_rejected(change):
    values, columns = {"a": .1, "b": .2}, ["a", "b"]
    if change == "missing":
        values.pop("b")
    elif change == "extra":
        values["future_target"] = .3
    elif change == "nan":
        values["a"] = np.nan
    elif change == "string":
        values["a"] = "bad"
    else:
        columns = ["a", "a"]
    with pytest.raises(ValueError):
        q.make_feature_frame(values, columns, "s1")


@pytest.mark.parametrize("change", ["order", "nonfinite", "postprocessing", "schema"])
def test_invalid_forecast_rejected(change):
    class BadModel(FixtureModel):
        def predict(self, x, raw=False):
            p = super().predict(x, raw)
            if change == "nonfinite":
                p.iloc[0, 0] = np.nan
            elif change == "postprocessing":
                p.iloc[0, 0] += 1 if not raw else 0
            elif change == "schema":
                p = p.iloc[:, :-1]
            return p
    model = BadModel()
    if change == "order":
        model.features = ["b", "a"]
    x = q.make_feature_frame({"a": .1, "b": .2}, ["a", "b"], "s1")
    with pytest.raises(ValueError):
        q.forecast_values({q.PRIMARY: model}, x, states({q.PRIMARY: None}))


def test_actual_four_forecasters_inference():
    rng = np.random.default_rng(63)
    ids = pd.Index([f"s{i}" for i in range(40)], name="scenario_id")
    x = pd.DataFrame(rng.normal(size=(40, 3)), index=ids, columns=list("abc"))
    y = pd.DataFrame({t: .01*x.a+rng.normal(0, .002, 40) for t in q.d.TARGETS}, index=ids)
    groups = pd.Series(np.repeat(range(8), 5), index=ids)
    models = {q.PRIMARY: q.b.h.HybridForecaster().fit(x, y, groups)}
    models.update({name: q.b.FixedBaseline(kind).fit(x, y, groups) for name, kind in q.b.METHODS.items()})
    result = q.forecast_values(models, x.iloc[:1], states(models))
    assert len(result) == 4
    assert all(len(p["raw"]) == len(p["processed"]) == 9 for p in result.values())


def decision_fixture(tmp_path, row, monkeypatch):
    run = tmp_path / "c4_artifacts/benchmark_runs/adult" / q.v.VERSION / row["scenario_id"]
    evidence_dir = tmp_path / "c4_artifacts/pre_release/adult" / row["scenario_id"]
    envelope = dict(scope="controlled_offline_simulation", production_package_approved=False, upstream={})
    envelope_hash = save_doc(run / "pre_release_freeze-v1", "upstream_simulation", envelope)
    evidence = dict(scenario_id=row["scenario_id"], slice_basis="c2_diagnosed_slice")
    evidence_hash = save_doc(evidence_dir, "evidence", evidence)
    source = tmp_path / "decision-input"
    source.write_bytes(b"frozen")
    freeze = dict(scenario=row, scope=q.SCOPE, complete_decision_evidence_frozen=True,
        future_evaluation_executed=False, production_package_approved=False, release_authorized=False,
        runtime_checks=dict(schema_compatible=True, rollback_ready=True),
        dependencies={str(source): q.d.sha(source.read_bytes())},
        upstream_simulation_sha256=envelope_hash, pre_release_evidence_sha256=evidence_hash)
    save_doc(run / "pre_release_freeze-v1", "freeze", freeze)
    monkeypatch.setattr(q.c, "completed_case", lambda *_: {"pre_release_evidence_sha256": evidence_hash})
    return run, freeze, source


def test_decision_dependency_integrity(row, tmp_path, monkeypatch):
    run, freeze, source = decision_fixture(tmp_path, row, monkeypatch)
    q.verify_decision(tmp_path, row)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        q.verify_decision(tmp_path, row)


@pytest.mark.parametrize("key", ["future_evaluation_executed", "release_authorized", "production_package_approved"])
def test_forbidden_decision_state(row, tmp_path, monkeypatch, key):
    run, freeze, _ = decision_fixture(tmp_path, row, monkeypatch)
    freeze[key] = True
    save_doc(run / "pre_release_freeze-v1", "freeze", freeze)
    with pytest.raises(ValueError, match="scope"):
        q.verify_decision(tmp_path, row)


def test_forecast_checkpoint_reproduces_and_detects_changes(row, tmp_path, monkeypatch):
    run, _, _ = decision_fixture(tmp_path, row, monkeypatch)
    import _scenario_meta
    monkeypatch.setattr(_scenario_meta, "feature_row", lambda *_: {"a": .1, "b": .2})
    models = {q.PRIMARY: FixtureModel()}
    bindings = {q.PRIMARY: {"metadata_sha256": "fixture"}}
    assert q.completed_forecast(tmp_path, row, models, bindings) is None
    record = q.freeze_forecasts(tmp_path, row, models, bindings)
    assert record == q.completed_forecast(tmp_path, row, models, bindings)
    state, _ = q.d.checked_json(run / "forecast_freeze-v1", "freeze")
    assert state["forecasts_frozen_before_future"] is True
    assert state["future_evaluation_executed"] is False
    with pytest.raises(FileExistsError):
        q.freeze_forecasts(tmp_path, row, models, bindings)
    with pytest.raises(ValueError, match="provenance"):
        q.completed_forecast(tmp_path, row, models, {q.PRIMARY: {"metadata_sha256": "changed"}})
    monkeypatch.setattr(_scenario_meta, "feature_row", lambda *_: {"a": .4, "b": .2})
    with pytest.raises(ValueError, match="Features do not match"):
        q.completed_forecast(tmp_path, row, models, bindings)
    (run / "forecast_freeze-v1/forecasts.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        q.completed_forecast(tmp_path, row, models, bindings)


def test_bounded_resume_and_final_gate(row, tmp_path, monkeypatch):
    second = dict(row, scenario_id=row["scenario_id"]+"-fixture-second")
    rows, records = [row, second], {}
    claim = dict(inputs={}, model_bindings={}, nonempty_admission_groups=1, blocked_cases_retained=35, N2_calibration_sha256="fixture-n2")
    monkeypatch.setattr(q, "completed_forecast", lambda root, item, *args: records.get(item["scenario_id"]))
    monkeypatch.setattr(q.c, "verify_scope", lambda *_: None)
    monkeypatch.setattr(q, "verify_decision", lambda *_: None)
    calls = []
    def runner(command, **kwargs):
        sid = command[command.index("--scenario-id")+1]
        assert "freeze_hybrid_policy_case.py" in command[1] and "--freeze" in command
        calls.append(sid)
        return SimpleNamespace(returncode=0)
    def freeze(root, item, *args):
        records[item["scenario_id"]] = {"scenario_id": item["scenario_id"]}
        return records[item["scenario_id"]]
    monkeypatch.setattr(q, "freeze_forecasts", freeze)
    output = q.f.bundle_at(tmp_path) / "policy-forecast-freezer-v1/completion-v1"
    r.run_batch(tmp_path, C4, claim, rows, {}, 1, runner)
    assert not output.exists() and len(calls) == 1
    r.run_batch(tmp_path, C4, claim, rows, {}, 2, runner)
    report, _ = q.d.checked_json(output, "forecast_batch")
    assert report["cases"] == 2 and report["forecasts_frozen_before_future"] is True
    assert report["future_evaluation_executed"] is False
    r.run_batch(tmp_path, C4, claim, rows, {}, 2, runner)
    assert len(calls) == 2


def test_preflight_model_bindings_and_future_stage_guard(row, tmp_path, monkeypatch):
    directory = q.f.bundle_at(tmp_path)
    candidate_claim = dict(inputs={}, blocked_cases=396, nonempty_admission_groups=12)
    candidate_dir = directory / "policy-candidate-executor-v1"
    save_doc(candidate_dir, "claim", candidate_claim)
    record = dict(scenario_id=row["scenario_id"])
    save_doc(candidate_dir / "completion-v1", "candidate_batch", dict(candidates=1, records=[record],
        future_evaluation_executed=False, complete_decision_evidence_freeze=False, N1_forecasts_frozen=False))
    monkeypatch.setattr(q.c, "prepare", lambda *_: (candidate_claim, [row]))
    monkeypatch.setattr(q.c, "completed_case", lambda *_: record)
    monkeypatch.setattr(q.c, "environment", lambda: {"fixture": True})
    run = tmp_path / "c4_artifacts/benchmark_runs/adult" / q.v.VERSION / row["scenario_id"]
    save_doc(run / "scoped_candidate_validation-v1", "validation", {})
    save_doc(run / "scoped_candidate_validation-v1", "metadata", {})
    hybrid_dir = tmp_path / "c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1/fit-v1"
    digest = save_doc(hybrid_dir, "metadata", {})
    bindings = {q.PRIMARY: dict(directory=str(hybrid_dir), metadata_sha256=digest)}
    columns = [f"feature_{i}" for i in range(75)]
    monkeypatch.setattr(q.b.h.HybridForecaster, "load", lambda *_: SimpleNamespace(features=columns))
    baseline_dir = directory / "baseline-fit-v1"
    for name, kind in q.b.METHODS.items():
        path = baseline_dir / name
        digest = save_doc(path, "model", {"kind": kind})
        bindings[name] = dict(directory=str(path), model_sha256=digest)
    monkeypatch.setattr(q.b.FixedBaseline, "load", lambda path:
        SimpleNamespace(features=columns, kind=q.b.METHODS[Path(path).name]))
    save_doc(baseline_dir, "metadata", dict(models=bindings))
    claim, rows, models = q.prepare(tmp_path, C4)
    assert rows == [row] and len(models) == 4
    assert claim["future_evaluation_authorized"] is False
    assert claim["model_refitting_authorized"] is False
    assert claim["model_bindings"] == bindings
    assert not (directory / "policy-forecast-freezer-v1").exists()
    (run / "external_future-v1").mkdir()
    with pytest.raises(ValueError, match="Future-stage artifacts"):
        q.prepare(tmp_path, C4)


REAL_N2_BINDING = q.n2_binding


def states(models, adjustment=.003):
    state = q.JointCalibration(.05,
        "replication_group_max_over_admitted_variants_and_three_targets",
        20, 20, adjustment, False, True, True).to_dict()
    return {name: dict(state) for name in models}


@pytest.fixture(autouse=True)
def fixed_n2_for_technical_fixtures(monkeypatch):
    monkeypatch.setattr(q, "n2_binding", lambda root, bindings:
        (states(bindings), "fixture-n2", {}))


def test_n2_uses_each_methods_fixed_adjustment_and_preserves_median():
    models = {q.PRIMARY: FixtureModel(), "comparator": FixtureModel()}
    x = q.make_feature_frame({"a": .1, "b": .2}, ["a", "b"], "case")
    calibration = states(models)
    calibration["comparator"]["adjustment"] = .007
    result = q.forecast_values(models, x, calibration)
    for method, adjustment in ((q.PRIMARY, .003), ("comparator", .007)):
        for target in q.d.TARGETS:
            raw = result[method]["processed"]
            calibrated = result[method]["calibrated"]
            assert calibrated[target+"__median"] == raw[target+"__median"]
            assert calibrated[target+"__lower"] == pytest.approx(raw[target+"__lower"]-adjustment)
            assert calibrated[target+"__upper"] == pytest.approx(raw[target+"__upper"]+adjustment)


def test_n2_requires_exact_frozen_model_bindings(tmp_path, monkeypatch):
    bindings = {q.PRIMARY: {"metadata_sha256": "fixed"}}
    path = q.f.bundle_at(tmp_path) / "n2-calibration-v1/completion-v1/calibration.json"
    monkeypatch.setattr(q.f, "verify_n2", lambda root:
        ({}, {"model_bindings": bindings, "methods": states(bindings)}, {str(path): "fixed-calibration-sha"}))
    _, digest, _ = REAL_N2_BINDING(tmp_path, bindings)
    assert digest == "fixed-calibration-sha"
    with pytest.raises(ValueError, match="exactly"):
        REAL_N2_BINDING(tmp_path, {q.PRIMARY: {"metadata_sha256": "changed"}})


def test_changed_n2_checkpoint_cannot_be_reused(row, tmp_path, monkeypatch):
    decision_fixture(tmp_path, row, monkeypatch)
    import _scenario_meta
    monkeypatch.setattr(_scenario_meta, "feature_row", lambda *_: {"a": .1, "b": .2})
    models = {q.PRIMARY: FixtureModel()}
    bindings = {q.PRIMARY: {"metadata_sha256": "fixture"}}
    q.freeze_forecasts(tmp_path, row, models, bindings)
    monkeypatch.setattr(q, "n2_binding", lambda root, bindings: (states(bindings), "changed-n2-sha", {}))
    with pytest.raises(ValueError, match="provenance"):
        q.completed_forecast(tmp_path, row, models, bindings)
