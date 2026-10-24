"""Policy admission adapter tests; no research generator is fitted by fixtures."""
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
import _hybrid_policy_execution as f
import run_hybrid_policy_admissions as r
import calibrate_hybrid_fresh_n2 as n


@pytest.fixture
def tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("c4policy")
    if os.name == "nt":
        raw = str(path.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\UNC\\"+raw[2:] if raw.startswith("\\\\") else "\\\\?\\"+raw
        path = Path(raw)
    yield path
    if os.name == "nt" and path.exists(): shutil.rmtree(path)


def doc(directory, stem, value):
    directory.mkdir(parents=True, exist_ok=True)
    data = f.e.encoded(value)
    (directory / (stem+".json")).write_bytes(data)
    (directory / (stem+".sha256")).write_text(f.d.sha(data)+"\n")
    return f.d.sha(data)


@pytest.fixture
def manifest():
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    return f.v.create_design(protocol)


def test_case_adapter_reuses_entire_admission_science():
    original = (C4 / "scripts/run_hybrid_fresh_case_admission.py").read_text(encoding="utf-8-sig")
    actual = (C4 / "scripts/run_hybrid_policy_case_admission.py").read_text(encoding="utf-8-sig")
    assert actual == f.expected_case_source(original)
    assert "generate_tvae_repair(" in actual
    assert "stability >= .50" in actual
    assert "frozen_tail_screen(protocol" in actual
    assert "'future_evaluation_executed': False" in actual


def test_group_adapter_exact_checkpoint_reuse():
    original = (C4 / "scripts/run_hybrid_fresh_admissions.py").read_text(encoding="utf-8-sig")
    actual = (C4 / "scripts/run_hybrid_policy_admissions.py").read_text(encoding="utf-8-sig")
    start, end = actual.index("def run_group("), actual.index("\n\ndef completed_group(")
    assert actual[start:end] == f.expected_group_function(original)


def test_all_policy_groups_same_grid_and_pair_order_no_mutation(manifest):
    before = f.e.encoded(manifest)
    groups = [g for g in manifest["groups"] if g["split_role"] == "policy_calibration"]
    assert len(groups) == 12
    for group in groups:
        rows = f.policy_group_cases(manifest, group["group_id"])
        assert len(rows) == 37
        assert all(row["split_role"] == "policy_calibration" for row in rows)
        assert sum(row["is_no_drift_control"] for row in rows) == 1
        positions = {row["scenario_id"]: i for i, row in enumerate(rows)}
        for row in rows:
            if row["repair_condition"] == "label_corrupted_repair":
                base = next(b for b in rows if b["shared_base_package_id"] == row["shared_base_package_id"]
                    and b["repair_condition"] == "targeted_repair")
                assert positions[base["scenario_id"]] < positions[row["scenario_id"]]
    assert f.e.encoded(manifest) == before


@pytest.mark.parametrize("role", ["conformal_calibration", "outer_test", "unknown"])
def test_nonpolicy_groups_rejected(manifest, role):
    gid = "unknown" if role == "unknown" else next(g["group_id"] for g in manifest["groups"] if g["split_role"] == role)
    with pytest.raises(ValueError): f.policy_group_cases(manifest, gid)


def make_n2(root, inputs=None):
    ids = [f"s{i}" for i in range(20)]
    index = pd.DataFrame(dict(scenario_id=ids, group_id=[f"g{i}" for i in range(20)], split_role=["conformal_calibration"]*20))
    targets = pd.DataFrame({t: np.arange(20)*.001 for t in f.d.TARGETS}, index=pd.Index(ids, name="scenario_id"))
    predicted = pd.DataFrame({t+"__"+label: np.full(20, val) for t in f.d.TARGETS
        for label, val in (("lower", -.01), ("median", 0.), ("upper", .01))}, index=targets.index)
    forecasts = {method: predicted.copy() for method in ("hybrid_ridge_lightgbm_lightgbm", *f.base.b.METHODS)}
    groups = pd.DataFrame(dict(group_id=[f"g{i}" for i in range(24)], admitted=[1]*20+[0]*4, blocked=[36]*20+[37]*4))
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    claim = dict(purpose="fresh_four_method_N2_group_max_calibration_v1", environment={}, inputs=inputs or {},
        definition=protocol["N2"], model_bindings={m: {"fixture": True} for m in forecasts},
        assigned_groups=24, nonempty_groups=20, all_blocked_groups=4, cases=20, blocked_cases_retained=868,
        support_note="Numerical support only", limitation="Synthetic fixture", production_release_authorized=False)
    output = n.save_results(root, claim, index, targets, forecasts, groups, n.fit_methods(index, targets, forecasts))
    return output


def test_verified_n2_accepts_fixed_four_methods(tmp_path):
    make_n2(tmp_path)
    claim, calibration, pins = f.verify_n2(tmp_path, full_inputs=True)
    assert len(calibration["methods"]) == 4
    assert calibration["production_release_authorized"] is False
    assert len(pins) > 4 and claim["nonempty_groups"] == 20


@pytest.mark.parametrize("change", ["missing", "checksum", "source", "release_flag"])
def test_missing_or_changed_n2_rejected(tmp_path, change):
    if change == "missing":
        with pytest.raises(FileNotFoundError): f.verify_n2(tmp_path)
        return
    source = tmp_path / "frozen-source.txt"
    source.write_bytes(b"fixed")
    output = make_n2(tmp_path, {str(source): f.d.sha(source.read_bytes())})
    if change == "checksum": (output / "calibration.json").write_bytes(b"changed")
    elif change == "source": source.write_bytes(b"changed")
    else:
        metadata, _ = f.d.checked_json(output)
        metadata["production_release_authorized"] = True
        doc(output, "metadata", metadata)
    with pytest.raises(ValueError): f.verify_n2(tmp_path, full_inputs=True)


def test_build_policy_binding_and_child_scope(tmp_path, manifest, monkeypatch):
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    protocol["protocol_version"] = f.v.VERSION
    inputs = {}
    for path in (tmp_path / "uci_adult_data/reference.csv", tmp_path / "uci_adult_data/test.csv",
                 tmp_path / "c4_artifacts/champion/adult" / protocol["source_versions"]["champion"] / "metadata.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen-fixture-input")
        inputs[str(path)] = f.d.sha(path.read_bytes())
    model = tmp_path / "c4_artifacts/champion/adult" / protocol["source_versions"]["champion"] / "champion.ubj"
    model.write_bytes(b"native-fixture")
    split = tmp_path / "c4_artifacts/splits/adult" / protocol["source_versions"]["split_manifest"] / "manifest.json"
    split.parent.mkdir(parents=True)
    split.write_bytes(b"split-fixture")
    make_n2(tmp_path, inputs)
    directory = tmp_path / "c4_artifacts/benchmark_protocols/adult" / f.v.VERSION
    protocol_hash = doc(directory, "protocol", protocol)
    manifest_hash = doc(directory, "scenario_manifest", manifest)
    freeze_hash = doc(directory, "freeze", dict(protocol_sha256=protocol_hash, scenario_manifest_sha256=manifest_hash,
        champion_model_sha256=f.d.sha(model.read_bytes()), split_manifest_sha256=f.d.sha(split.read_bytes())))
    doc(f.bundle_at(tmp_path) / "admission-executor-v1", "claim", {"compatibility_hashes": {
        "protocol": protocol_hash, "scenario_manifest": manifest_hash, "freeze": freeze_hash}})
    import _hybrid_fresh_candidate
    monkeypatch.setattr(_hybrid_fresh_candidate, "environment", lambda: {})
    claim, actual = f.build_execution(tmp_path, C4)
    assert actual == manifest and len(claim["allowed_group_ids"]) == 12
    assert claim["planned_cases"] == 444 and claim["future_evaluation_authorized"] is False
    assert claim["N2_refitting_authorized"] is False and claim["policy_fitting_authorized"] is False
    f.ensure_documents(f.bundle_at(tmp_path) / "policy-admission-executor-v1", {"claim": claim})
    f.verify_execution_binding(tmp_path, claim["allowed_group_ids"][0])
    outer = next(g["group_id"] for g in manifest["groups"] if g["split_role"] == "outer_test")
    with pytest.raises(ValueError): f.verify_execution_binding(tmp_path, outer)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError): f.verify_execution_binding(tmp_path, claim["allowed_group_ids"][0])


def test_bounded_case_then_complete_resume_and_preserve_all_blocked(tmp_path, manifest, monkeypatch):
    gid = next(g["group_id"] for g in manifest["groups"] if g["split_role"] == "policy_calibration")
    freeze_hash = doc(tmp_path / "c4_artifacts/benchmark_protocols/adult" / f.v.VERSION, "freeze", {"fixture": True})
    doc(f.bundle_at(tmp_path) / "policy-admission-executor-v1", "claim", {"fixture": True})
    monkeypatch.setattr(f, "verify_execution_binding", lambda *a: manifest)
    records, calls = {}, []
    monkeypatch.setattr(r, "existing_record", lambda root, row, digest: records.get(row["scenario_id"]))
    def runner(command, **kwargs):
        assert command[1].endswith("run_hybrid_policy_case_admission.py")
        sid = command[command.index("--scenario-id")+1]
        calls.append(sid)
        records[sid] = dict(scenario_id=sid, simulation_admission="BLOCKED_ESCALATE", screen_status="FAIL")
        return SimpleNamespace(returncode=0)
    r.run_group(tmp_path, C4, gid, manifest, 1, runner)
    assert len(calls) == 1 and r.completed_group(tmp_path, gid, manifest) is None
    r.run_group(tmp_path, C4, gid, manifest, 37, runner)
    assert len(calls) == len(set(calls)) == 37
    complete = r.completed_group(tmp_path, gid, manifest)
    assert complete["admitted"] == 0 and complete["blocked"] == 37
    r.run_group(tmp_path, C4, gid, manifest, 37, lambda *a, **k: pytest.fail("Must reuse completed cases"))


def test_failed_child_stops_preserves_log(tmp_path, manifest, monkeypatch):
    gid = next(g["group_id"] for g in manifest["groups"] if g["split_role"] == "policy_calibration")
    doc(tmp_path / "c4_artifacts/benchmark_protocols/adult" / f.v.VERSION, "freeze", {})
    doc(f.bundle_at(tmp_path) / "policy-admission-executor-v1", "claim", {})
    monkeypatch.setattr(r, "existing_record", lambda *a: None)
    with pytest.raises(ValueError, match="preserve"):
        r.run_group(tmp_path, C4, gid, manifest, 37, lambda *a, **k: SimpleNamespace(returncode=1))
    logs = list((tmp_path / "c4_artifacts/benchmark_runs/adult" / f.v.VERSION / "group_admissions" / gid / "v1").glob("*.log"))
    assert len(logs) == 1


@pytest.mark.parametrize("complete", [False, True])
def test_all_twelve_group_completion_accounts_for_blocked_cases(tmp_path, manifest, monkeypatch, complete):
    gids = [g["group_id"] for g in manifest["groups"] if g["split_role"] == "policy_calibration"]
    def record(root, gid, source):
        if not complete and gid == gids[-1]: return None
        return dict(group_id=gid, group_admissions_sha256="fixture", admitted=0, blocked=37)
    monkeypatch.setattr(r, "completed_group", record)
    r.save_completion(tmp_path, {"allowed_group_ids": gids}, manifest)
    output = f.bundle_at(tmp_path) / "policy-admission-executor-v1/completion-v1"
    if not complete:
        assert not output.exists()
        return
    summary, _ = f.d.checked_json(output, "policy_admissions")
    assert summary["assigned_groups"] == 12 and summary["blocked_cases"] == 444
    assert summary["admitted_cases"] == summary["nonempty_groups"] == 0
    assert len(summary["records"]) == 12 and summary["outer_test_accessed"] is False
