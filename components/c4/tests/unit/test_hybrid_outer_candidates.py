"""Outer candidate-stage technical checks; no research candidate is trained."""
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import _hybrid_outer_candidate as c
import run_hybrid_outer_candidates as r


@pytest.fixture
def tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("c4pcandidate")
    if os.name == "nt":
        raw = str(path.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\UNC\\"+raw[2:] if raw.startswith("\\\\") else "\\\\?\\"+raw
        path = Path(raw)
    yield path
    if os.name == "nt" and path.exists(): shutil.rmtree(path)


def doc(directory, stem, value):
    directory.mkdir(parents=True, exist_ok=True)
    data = c.e.encoded(value)
    (directory / (stem+".json")).write_bytes(data)
    (directory / (stem+".sha256")).write_text(c.d.sha(data)+"\n")
    return c.d.sha(data)


@pytest.fixture
def manifest():
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    return c.v.create_design(protocol)


def test_exact_validator_training_and_evaluation_reuse():
    old = (C4 / "scripts/validate_portable_training_candidate.py").read_text(encoding="utf-8-sig")
    new = (C4 / "scripts/validate_hybrid_outer_candidate.py").read_text(encoding="utf-8-sig")
    assert new == c.expected_validator_source(old)
    assert "rows[0]['split_role'] == 'outer_test'" in new
    assert "train_champion(" in new and "save_pre_release_evidence(" in new
    assert "save_candidate_model_portable(model" in new
    assert c.save_candidate_model_portable is c.original.save_candidate_model_portable
    assert c.completed_case is c.original.completed_case


def fixture_inventory(root, manifest, monkeypatch):
    doc(c.f.bundle_at(root), "scenario_manifest", manifest)
    freeze_hash = doc(root / "c4_artifacts/benchmark_protocols/adult" / c.v.VERSION, "freeze", {"fixture": True})
    gids = [g["group_id"] for g in manifest["groups"] if g["split_role"] == "outer_test"]
    admission_execution = {"allowed_group_ids": gids, "inputs": {}, "policy_selection_sha256": "a"*64}
    doc(c.f.bundle_at(root) / "outer-admission-executor-v1", "claim", admission_execution)
    records, completions = {}, []
    for group_number, gid in enumerate(gids):
        admitted_count = 6 if group_number == 0 else 5 if group_number < 11 else 0
        rows = c.f.outer_group_cases(manifest, gid)
        group_records = []
        for number, row in enumerate(rows):
            run = root / "c4_artifacts/benchmark_runs/adult" / c.v.VERSION / row["scenario_id"]
            digest = doc(run, "admission", {"scenario": row, "fixture": True})
            record = dict(scenario_id=row["scenario_id"], group_id=gid, split_role="outer_test",
                simulation_admission="READY_FOR_SCOPED_CANDIDATE_VALIDATION" if number < admitted_count else "BLOCKED_ESCALATE",
                admission_sha256=digest)
            records[row["scenario_id"]] = record
            group_records.append(record)
        path = root / "c4_artifacts/benchmark_runs/adult" / c.v.VERSION / "group_admissions" / gid / "v1"
        digest = doc(path, "group_admissions", {"records": group_records})
        completions.append(dict(group_id=gid, admitted=admitted_count, blocked=37-admitted_count, group_admissions_sha256=digest))
    summary = dict(purpose="all_fixed_fresh_outer_admissions_complete", assigned_groups=16, planned_cases=592,
        candidate_training_executed=False, future_evaluation_executed=False, N1_refitted=False, N2_refitted=False,
        policy_fitted=False, outer_admissions_executed=True, outer_future_evaluation_executed=False, production_release_authorized=False,
        records=completions, admitted_cases=56, blocked_cases=536, nonempty_groups=11)
    doc(c.f.bundle_at(root) / "outer-admission-executor-v1/completion-v1", "outer_admissions", summary)
    monkeypatch.setattr(c.a, "completed_group", lambda root, gid, manifest: next(r for r in completions if r["group_id"] == gid))
    monkeypatch.setattr(c.a, "existing_record", lambda root, row, freeze: records[row["scenario_id"]])
    return admission_execution, summary


def test_inventory_and_training_claim_bind_56_admitted_536_blocked(tmp_path, manifest, monkeypatch):
    execution, summary = fixture_inventory(tmp_path, manifest, monkeypatch)
    monkeypatch.setattr(c.f, "build_execution", lambda *a: (execution, manifest))
    monkeypatch.setattr(c, "environment", lambda: {})
    claim, rows = c.prepare(tmp_path, C4)
    assert len(rows) == claim["admitted_cases"] == 56
    assert claim["blocked_cases"] == 536
    assert claim["assigned_groups"] == 16 and claim["nonempty_admission_groups"] == 11
    assert all(row["split_role"] == "outer_test" for row in rows)
    assert claim["candidate_training_authorized"] is True
    assert claim["future_evaluation_authorized"] is False
    assert claim["N2_refitting_authorized"] is False and claim["policy_fitting_authorized"] is False
    assert claim["outer_evaluation_authorized"] is False


@pytest.mark.parametrize("change", ["missing_group", "wrong_count", "future_flag"])
def test_incomplete_or_changed_outer_admissions_rejected(tmp_path, manifest, monkeypatch, change):
    _, summary = fixture_inventory(tmp_path, manifest, monkeypatch)
    if change == "missing_group": monkeypatch.setattr(c.a, "completed_group", lambda *a: None)
    elif change == "wrong_count": summary["admitted_cases"] = 57
    else: summary["future_evaluation_executed"] = True
    doc(c.f.bundle_at(tmp_path) / "outer-admission-executor-v1/completion-v1", "outer_admissions", summary)
    with pytest.raises(ValueError): c.inventory(tmp_path)


def scope_fixture(root, manifest, row):
    claim = dict(purpose="fresh_outer_candidate_pre_release_execution_v1", policy_selection_sha256="a"*64, role="outer_test", candidate_training_authorized=True, future_evaluation_authorized=False,
        generator_refitting_authorized=False, N1_refitting_authorized=False, N2_refitting_authorized=False,
        policy_fitting_authorized=False, outer_evaluation_authorized=False, production_release_authorized=False,
        environment={}, scenario_ids=[row["scenario_id"]], inputs={})
    doc(c.f.bundle_at(root), "scenario_manifest", manifest)
    doc(c.f.bundle_at(root) / "outer-candidate-executor-v1", "claim", claim)
    return claim


@pytest.mark.parametrize("role", ["conformal_calibration", "policy_calibration"])
def test_other_roles_rejected_even_if_erroneously_allowlisted(tmp_path, manifest, monkeypatch, role):
    row = next(r for r in manifest["scenarios"] if r["split_role"] == role)
    scope_fixture(tmp_path, manifest, row)
    monkeypatch.setattr(c, "environment", lambda: {})
    monkeypatch.setattr(c.f, "verify_execution_binding", lambda *a: pytest.fail("Must reject role first"))
    with pytest.raises(ValueError, match="outer-role"): c.verify_scope(tmp_path, row["scenario_id"])


@pytest.mark.parametrize("flag", ["future_evaluation_authorized", "N2_refitting_authorized", "policy_fitting_authorized", "outer_evaluation_authorized"])
def test_unauthorized_extra_stage_flags_rejected(tmp_path, manifest, monkeypatch, flag):
    row = next(r for r in manifest["scenarios"] if r["split_role"] == "outer_test")
    claim = scope_fixture(tmp_path, manifest, row)
    claim[flag] = True
    doc(c.f.bundle_at(tmp_path) / "outer-candidate-executor-v1", "claim", claim)
    monkeypatch.setattr(c, "environment", lambda: {})
    with pytest.raises(ValueError): c.verify_scope(tmp_path, row["scenario_id"])


def test_bounded_candidate_then_resume_completed_models(tmp_path, manifest, monkeypatch):
    rows = [row for row in manifest["scenarios"] if row["split_role"] == "outer_test"][:2]
    claim = dict(inputs={}, policy_selection_sha256="a"*64, nonempty_admission_groups=11, blocked_cases=536)
    records, calls = {}, []
    monkeypatch.setattr(c, "completed_case", lambda root, row: records.get(row["scenario_id"]))
    monkeypatch.setattr(c, "verify_scope", lambda *a: None)
    def runner(command, **kwargs):
        assert command[1].endswith("validate_hybrid_outer_candidate.py")
        sid = command[command.index("--scenario-id")+1]
        calls.append(sid)
        records[sid] = {"scenario_id": sid, "fixture": True}
        return SimpleNamespace(returncode=0)
    r.run_batch(tmp_path, C4, claim, rows, 1, runner)
    assert len(calls) == 1
    r.run_batch(tmp_path, C4, claim, rows, 48, runner)
    assert len(calls) == len(set(calls)) == 2
    report, _ = c.d.checked_json(c.f.bundle_at(tmp_path) / "outer-candidate-executor-v1/completion-v1", "candidate_batch")
    assert report["candidates"] == 2 and report["assigned_groups"] == 16
    assert report["blocked_cases_retained"] == 536
    assert report["future_evaluation_executed"] is False and report["N1_forecasts_frozen"] is False
    r.run_batch(tmp_path, C4, claim, rows, 48, lambda *a, **k: pytest.fail("Must reuse checkpoints"))


def test_failed_candidate_preserves_log_and_stops(tmp_path, manifest, monkeypatch):
    rows = [row for row in manifest["scenarios"] if row["split_role"] == "outer_test"][:1]
    monkeypatch.setattr(c, "completed_case", lambda *a: None)
    monkeypatch.setattr(c, "verify_scope", lambda *a: None)
    with pytest.raises(ValueError, match="preserve"):
        r.run_batch(tmp_path, C4, {"inputs": {}}, rows, 48, lambda *a, **k: SimpleNamespace(returncode=1))
    assert len(list((c.f.bundle_at(tmp_path) / "outer-candidate-executor-v1").glob("*.log"))) == 1


def test_scope_accepts_admitted_outer_id_and_rejects_unlisted_id(tmp_path,manifest,monkeypatch):
    rows = [r for r in manifest['scenarios'] if r['split_role'] == 'outer_test']
    row = rows[0]
    scope_fixture(tmp_path,manifest,row)
    doc(c.f.bundle_at(tmp_path)/'outer-admission-executor-v1','claim',dict(policy_selection_sha256='a'*64))
    monkeypatch.setattr(c,'environment',lambda:{})
    monkeypatch.setattr(c.f,'verify_execution_binding',lambda *a:manifest)
    assert c.verify_scope(tmp_path,row['scenario_id']) == row
    with pytest.raises(ValueError,match='not admitted'): c.verify_scope(tmp_path,rows[1]['scenario_id'])


def test_changed_policy_selection_binding_rejected(tmp_path,manifest,monkeypatch):
    row = next(r for r in manifest['scenarios'] if r['split_role'] == 'outer_test')
    scope_fixture(tmp_path,manifest,row)
    doc(c.f.bundle_at(tmp_path)/'outer-admission-executor-v1','claim',dict(policy_selection_sha256='b'*64))
    monkeypatch.setattr(c,'environment',lambda:{})
    monkeypatch.setattr(c.f,'verify_execution_binding',lambda *a:manifest)
    with pytest.raises(ValueError,match='policy-selection binding'): c.verify_scope(tmp_path,row['scenario_id'])


def test_changed_input_fails_before_training_scope(tmp_path,manifest,monkeypatch):
    row = next(r for r in manifest['scenarios'] if r['split_role'] == 'outer_test')
    claim = scope_fixture(tmp_path,manifest,row)
    path = tmp_path/'bound.txt'; path.write_bytes(b'fixed')
    claim['inputs'][str(path)] = c.d.sha(path.read_bytes())
    doc(c.f.bundle_at(tmp_path)/'outer-candidate-executor-v1','claim',claim)
    path.write_bytes(b'changed')
    monkeypatch.setattr(c,'environment',lambda:{})
    with pytest.raises(ValueError): c.verify_scope(tmp_path,row['scenario_id'])


@pytest.mark.parametrize('status',['BLOCKED_ESCALATE','READY_FOR_SCOPED_CANDIDATE_VALIDATION'])
def test_group_context_rejects_blocked_or_mismatched_admission(tmp_path,manifest,monkeypatch,status):
    row = next(r for r in manifest['scenarios'] if r['split_role'] == 'outer_test')
    monkeypatch.setattr(c,'verify_scope',lambda *a:row)
    directory = tmp_path/'c4_artifacts/benchmark_runs/adult'/c.v.VERSION/'group_admissions'/row['group_id']/'v1'
    record = dict(scenario_id=row['scenario_id'],simulation_admission=status)
    doc(directory,'group_admissions',dict(records=[record]))
    doc(tmp_path/'c4_artifacts/benchmark_protocols/adult'/c.v.VERSION,'freeze',{})
    monkeypatch.setattr(c.a,'existing_record',lambda *a:dict(record,changed=True))
    with pytest.raises(ValueError): c.group_context(tmp_path,row['scenario_id'])
