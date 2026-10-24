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
import _hybrid_outer_forecasts as q
import run_hybrid_outer_forecast_freeze as r


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
    return next(s for s in manifest["scenarios"] if s["split_role"] == "outer_test")


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
    adapted = (C4 / "scripts/freeze_hybrid_outer_case.py").read_text(encoding="utf-8-sig")
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
    monkeypatch.setattr(q, "controller_values", lambda *a: {"technical_fixture": True})
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
    claim = dict(inputs={}, model_bindings={}, nonempty_admission_groups=1, blocked_cases_retained=35, N2_calibration_sha256="fixture-n2", policy_selection_sha256="fixture-policy")
    monkeypatch.setattr(q, "completed_forecast", lambda root, item, *args: records.get(item["scenario_id"]))
    monkeypatch.setattr(q.c, "verify_scope", lambda *_: None)
    monkeypatch.setattr(q, "verify_decision", lambda *_: None)
    calls = []
    def runner(command, **kwargs):
        sid = command[command.index("--scenario-id")+1]
        assert "freeze_hybrid_outer_case.py" in command[1] and "--freeze" in command
        calls.append(sid)
        return SimpleNamespace(returncode=0)
    def freeze(root, item, *args):
        records[item["scenario_id"]] = {"scenario_id": item["scenario_id"]}
        return records[item["scenario_id"]]
    monkeypatch.setattr(q, "freeze_forecasts", freeze)
    output = q.f.bundle_at(tmp_path) / "outer-forecast-freezer-v1/completion-v1"
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
    candidate_claim = dict(inputs={}, blocked_cases=536, nonempty_admission_groups=11, policy_selection_sha256="fixture-policy")
    candidate_dir = directory / "outer-candidate-executor-v1"
    save_doc(candidate_dir, "claim", candidate_claim)
    record = dict(scenario_id=row["scenario_id"])
    save_doc(candidate_dir / "completion-v1", "candidate_batch", dict(purpose="completed_fresh_outer_candidate_pre_release_batch",assigned_groups=16,blocked_cases_retained=536,policy_selection_sha256="fixture-policy",production_release_authorized=False,candidates=1, records=[record],
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
    assert not (directory / "outer-forecast-freezer-v1").exists()
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
    monkeypatch.setattr(q, "policy_binding", lambda root, bindings:
        (dict(model_bindings=bindings,N2_calibration_sha256="fixture-n2"),"fixture-policy",{}))


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
    monkeypatch.setattr(q.f.base, "verify_n2", lambda root:
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


REAL_CONTROLLER_VALUES = q.controller_values
REAL_POLICY_BINDING = q.policy_binding


def controller_fixture():
    methods = q.f.METHODS
    features = dict(c1_harm_width=.01,c2_diagnosis_stability=.6,c3_fidelity_score=.7,c3_utility_score=.1)
    snapshot = dict(runtime_checks=dict(schema_compatible=True,rollback_ready=True))
    envelope = dict(upstream=dict(c2_diagnosis=dict(diagnosis_stability=.6),
        c3_repair_package=dict(repair_package_status='APPROVED',privacy_status='PASS')))
    forecasts = {m:dict(calibrated={t+'__'+label:value for t in q.d.TARGETS for label,value in [('lower',.01),('median',.015),('upper',.02)]}) for m in methods}
    state = {m:dict(auto_deploy_supported=True) for m in methods}
    selected = dict(N2_calibration_sha256='fixture-n2',context_support=dict(unsupported_contexts=['LOW','HIGH']),
        methods={m:{family:dict(status='UNSUPPORTED_SELECTION_REVIEW_ONLY',
            selected_policy=dict(policy_id='unsupported-review-only',kind='review_only')) for family in ('global','adaptive')} for m in methods})
    return forecasts,features,snapshot,envelope,state,selected


def test_selected_review_only_cannot_become_auto_deploy_even_when_n3_safe():
    args = controller_fixture()
    result = q.controller_values(*args)
    assert result['future_targets_used'] is False and result['production_release_authorized'] is False
    assert all(m['N3']['release_signal'] == 'AUTO_DEPLOY' for m in result['methods'].values())
    assert all(f['result']['decision'] == 'HUMAN_REVIEW' for m in result['methods'].values() for f in m['policies'].values())


def test_mandatory_unsafe_escalation_survives_review_only_policy():
    forecasts,*rest = controller_fixture()
    method = q.f.METHODS[-1]
    forecasts[method]['calibrated'][q.d.TARGETS[0]+'__lower'] = -.02
    forecasts[method]['calibrated'][q.d.TARGETS[0]+'__upper'] = -.01
    result = q.controller_values(forecasts,*rest)
    assert all(f['result']['decision'] == 'ESCALATE' for f in result['methods'][method]['policies'].values())


@pytest.mark.parametrize('gate',['schema','rollback','privacy','repair','stability'])
def test_failed_gates_override_every_selected_policy(gate):
    forecasts,features,snapshot,envelope,state,selected = controller_fixture()
    if gate == 'schema': snapshot['runtime_checks']['schema_compatible'] = False
    elif gate == 'rollback': snapshot['runtime_checks']['rollback_ready'] = False
    elif gate == 'privacy': envelope['upstream']['c3_repair_package']['privacy_status'] = 'FAIL'
    elif gate == 'repair': envelope['upstream']['c3_repair_package']['repair_package_status'] = 'REJECTED'
    else: envelope['upstream']['c2_diagnosis']['diagnosis_stability'] = .49
    result = q.controller_values(forecasts,features,snapshot,envelope,state,selected)
    assert all(f['result']['decision'] == 'ESCALATE' for m in result['methods'].values() for f in m['policies'].values())


def test_frozen_unsupported_context_is_not_reestimated_on_outer_cases():
    forecasts,features,snapshot,envelope,state,selected = controller_fixture()
    features.update(c1_harm_width=0.,c2_diagnosis_stability=1.,c3_fidelity_score=1.,c3_utility_score=1.)
    for families in selected['methods'].values():
        families['global'] = dict(status='EMPIRICALLY_SELECTED',selected_policy=dict(policy_id='global-00',kind='global',buffer=0.))
        families['adaptive'] = dict(status='EMPIRICALLY_SELECTED',selected_policy=dict(policy_id='adaptive-00',kind='adaptive',buffers=dict(LOW=0.,MEDIUM=0.,HIGH=0.)))
    result = q.controller_values(forecasts,features,snapshot,envelope,state,selected)
    assert result['evidence_context']['context'] == 'LOW'
    for m in result['methods'].values():
        assert m['policies']['global']['result']['decision'] == 'AUTO_DEPLOY'
        assert m['policies']['adaptive']['result']['decision'] == 'HUMAN_REVIEW'


def test_policy_binding_requires_same_model_inventory(tmp_path,monkeypatch):
    bindings = {m:dict(fixed=True) for m in q.f.METHODS}
    path = q.f.bundle_at(tmp_path)/'policy-selection-v1/completion-v1/selection.json'
    monkeypatch.setattr(q.f,'verify_selection',lambda root:({},dict(model_bindings=bindings),{str(path):'frozen-selection'}))
    assert REAL_POLICY_BINDING(tmp_path,bindings)[1] == 'frozen-selection'
    with pytest.raises(ValueError): REAL_POLICY_BINDING(tmp_path,{})


@pytest.mark.parametrize('change',['checksum','self_rehashed_signal','policy_binding'])
def test_real_controller_checkpoint_replay_and_tamper_detection(row,tmp_path,monkeypatch,change):
    run,snapshot,_ = decision_fixture(tmp_path,row,monkeypatch)
    forecasts,features,_,envelope,_,selected = controller_fixture()
    envelope.update(scope='controlled_offline_simulation',production_package_approved=False)
    snapshot['upstream_simulation_sha256'] = save_doc(run/'pre_release_freeze-v1','upstream_simulation',envelope)
    save_doc(run/'pre_release_freeze-v1','freeze',snapshot)
    monkeypatch.setattr(q,'controller_values',REAL_CONTROLLER_VALUES)
    import _scenario_meta
    monkeypatch.setattr(_scenario_meta,'feature_row',lambda *_:features)
    class ControllerModel(FixtureModel):
        pass
    models = {m:ControllerModel() for m in q.f.METHODS}
    for model in models.values(): model.features = list(features)
    bindings = {m:dict(fixed=True) for m in models}
    monkeypatch.setattr(q,'policy_binding',lambda root,b:(selected,'fixture-policy',{}))
    record = q.freeze_forecasts(tmp_path,row,models,bindings)
    assert record == q.completed_forecast(tmp_path,row,models,bindings)
    folder = run/'forecast_freeze-v1'
    marker,_ = q.d.checked_json(folder,'freeze')
    payload = json.loads((folder/'controller_decisions.json').read_bytes())
    assert marker['selected_policy_decisions_frozen_before_future'] is True
    assert all(f['result']['decision'] == 'HUMAN_REVIEW' for m in payload['methods'].values() for f in m['policies'].values())
    if change == 'checksum': (folder/'controller_decisions.json').write_bytes(b'changed')
    elif change == 'policy_binding': monkeypatch.setattr(q,'policy_binding',lambda root,b:(selected,'changed-policy',{}))
    else:
        payload['methods'][q.PRIMARY]['policies']['global']['result']['decision'] = 'AUTO_DEPLOY'
        content = q.e.encoded(payload); (folder/'controller_decisions.json').write_bytes(content)
        marker['controller_decisions_sha256'] = q.d.sha(content)
        save_doc(folder,'freeze',marker)
    with pytest.raises(ValueError): q.completed_forecast(tmp_path,row,models,bindings)
