"""Technical fixtures only: scope, exact science reuse and admission checkpoints.

These tests do not claim actual TVAE output quality or research performance.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import _hybrid_fresh_execution as f
import run_hybrid_fresh_admissions as r


@pytest.fixture
def manifest():
    original = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    return f.v.create_design(original)


def first_group(manifest):
    return next(g["group_id"] for g in manifest["groups"] if g["split_role"] == "conformal_calibration")


def test_all_cases_and_targeted_before_corrupted(manifest):
    rows = f.fresh_group_cases(manifest, first_group(manifest))
    assert len(rows) == 37
    for i, row in enumerate(rows):
        if row["repair_condition"] == "label_corrupted_repair":
            before = rows[:i]
            assert any(b["shared_base_package_id"] == row["shared_base_package_id"]
                       and b["repair_condition"] == "targeted_repair" for b in before)


@pytest.mark.parametrize("role", ["policy_calibration", "outer_test"])
def test_policy_and_outer_roles_blocked(manifest, role):
    group = next(g["group_id"] for g in manifest["groups"] if g["split_role"] == role)
    with pytest.raises(ValueError, match="only a frozen conformal"):
        f.fresh_group_cases(manifest, group)


def test_unknown_group_rejected(manifest):
    with pytest.raises(ValueError):
        f.fresh_group_cases(manifest, "invented-group")


@pytest.mark.parametrize("change", ["missing", "duplicate", "role", "factor"])
def test_manifest_changes_rejected(manifest, change):
    group = first_group(manifest)
    row = next(s for s in manifest["scenarios"] if s["group_id"] == group and not s["is_no_drift_control"])
    if change == "missing":
        manifest["scenarios"].remove(row)
    elif change == "duplicate":
        manifest["scenarios"].append(copy.deepcopy(row))
    elif change == "role":
        row["split_role"] = "outer_test"
    else:
        row["magnitude"] = .99
    with pytest.raises(ValueError):
        f.fresh_group_cases(manifest, group)


def test_admission_science_reused_exactly():
    original = (C4 / "scripts/run_training_group_case_admission.py").read_text(encoding="utf-8-sig")
    adapted = (C4 / "scripts/run_hybrid_fresh_case_admission.py").read_text(encoding="utf-8-sig")
    assert f.expected_admission_source(original) == adapted
    assert "adult-simulation-pp1-v2" not in adapted
    assert "verify_execution_binding(root, args.group_id)" in adapted


def test_immutable_documents_and_corruption(tmp_path):
    documents = {"claim": {"scope": "admission", "future": False}}
    f.ensure_documents(tmp_path / "bindings", documents)
    f.ensure_documents(tmp_path / "bindings", documents)
    with pytest.raises(ValueError, match="differs"):
        f.ensure_documents(tmp_path / "bindings", {"claim": {"future": True}})
    (tmp_path / "bindings/claim.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        f.ensure_documents(tmp_path / "bindings", documents)


def save_doc(path, stem, data):
    path.mkdir(parents=True, exist_ok=True)
    content = f.e.encoded(data)
    (path / (stem+".json")).write_bytes(content)
    (path / (stem+".sha256")).write_text(f.d.sha(content)+"\n")
    return f.d.sha(content)


def make_case(root, row, freeze_hash, status="BLOCKED_ESCALATE"):
    artifact = "technical-fixture-"+f.d.sha(row["scenario_id"].encode())[:12]
    package = root / "c4_artifacts/offline_repair/adult" / artifact
    package.mkdir(parents=True)
    # Byte fixtures are only hashed, never deserialized as generators.
    csv, generator = b"fixture csv", b"fixture generator"
    (package / "synthetic_unvalidated.csv").write_bytes(csv)
    (package / "generator.pkl").write_bytes(generator)
    generation_hash = save_doc(package, "metadata", {
        "synthetic_csv_sha256": f.d.sha(csv), "generator_sha256": f.d.sha(generator)})
    privacy_hash = save_doc(package, "privacy_diagnostics", {"generation_metadata_sha256": generation_hash})
    admission = dict(scenario=row, freeze_sha256=freeze_hash, candidate_training_executed=False,
        future_evaluation_executed=False, production_package_approved=False,
        simulation_admission=status, screen_status="PASS" if status.startswith("READY") else "FAIL",
        generation_metadata_sha256=generation_hash, privacy_report_sha256=privacy_hash,
        artifact_version=artifact, tail_distance_ratio=.9)
    run = root / "c4_artifacts/benchmark_runs/adult" / f.VERSION / row["scenario_id"]
    save_doc(run, "admission", admission)
    return package, run


def test_existing_case_integrity(manifest, tmp_path):
    row = f.fresh_group_cases(manifest, first_group(manifest))[0]
    assert r.existing_record(tmp_path, row, "freeze") is None
    package, _ = make_case(tmp_path, row, "freeze")
    result = r.existing_record(tmp_path, row, "freeze")
    assert result["simulation_admission"] == "BLOCKED_ESCALATE"
    (package / "generator.pkl").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        r.existing_record(tmp_path, row, "freeze")


def test_incomplete_case_not_retried(manifest, tmp_path):
    row = f.fresh_group_cases(manifest, first_group(manifest))[0]
    (tmp_path / "c4_artifacts/benchmark_runs/adult" / f.VERSION / row["scenario_id"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete case"):
        r.existing_record(tmp_path, row, "freeze")


def test_case_binding_mismatch_rejected(manifest, tmp_path):
    row = f.fresh_group_cases(manifest, first_group(manifest))[0]
    make_case(tmp_path, row, "freeze")
    with pytest.raises(ValueError, match="protocol differs"):
        r.existing_record(tmp_path, row, "different-freeze")


def test_one_case_limit_resume_and_complete_group(manifest, tmp_path, monkeypatch):
    group = first_group(manifest)
    rows = f.fresh_group_cases(manifest, group)
    frozen = tmp_path / "c4_artifacts/benchmark_protocols/adult" / f.VERSION
    freeze_hash = save_doc(frozen, "freeze", {"fixture": True})
    save_doc(f.bundle_at(tmp_path) / "admission-executor-v1", "claim", {"fixture": True})
    monkeypatch.setattr(f, "verify_execution_binding", lambda *_: manifest)
    called = []
    def runner(command, **kwargs):
        sid = command[command.index("--scenario-id")+1]
        assert "run_hybrid_fresh_case_admission.py" in command[1]
        row = next(row for row in rows if row["scenario_id"] == sid)
        make_case(tmp_path, row, freeze_hash)
        called.append(sid)
        return SimpleNamespace(returncode=0)
    r.run_group(tmp_path, C4, group, manifest, 1, runner)
    assert called == [rows[0]["scenario_id"]]
    output = tmp_path / "c4_artifacts/benchmark_runs/adult" / f.VERSION / "group_admissions" / group / "v1"
    assert not (output / "group_admissions.json").exists()
    r.run_group(tmp_path, C4, group, manifest, 1, runner)
    assert called == [rows[0]["scenario_id"], rows[1]["scenario_id"]]
    r.run_group(tmp_path, C4, group, manifest, 37, runner)
    report, _ = f.d.checked_json(output, "group_admissions")
    assert report["cases"] == report["blocked_cases"] == 37
    assert report["scoped_admissible_cases"] == 0
    assert report["future_evaluation_executed"] is False
    assert len(called) == len(set(called)) == 37
    r.run_group(tmp_path, C4, group, manifest, 37, runner)
    assert len(called) == 37


def test_subprocess_failure_stops_without_group_report(manifest, tmp_path, monkeypatch):
    group = first_group(manifest)
    save_doc(tmp_path / "c4_artifacts/benchmark_protocols/adult" / f.VERSION, "freeze", {})
    save_doc(f.bundle_at(tmp_path) / "admission-executor-v1", "claim", {})
    with pytest.raises(ValueError, match="Admission failed"):
        r.run_group(tmp_path, C4, group, manifest, 1,
                    lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert not list((tmp_path / "c4_artifacts/benchmark_runs").rglob("group_admissions.json"))


def test_preflight_materialization_and_full_binding_verification(manifest, tmp_path, monkeypatch):
    directory = f.bundle_at(tmp_path)
    original = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    original_dir = tmp_path / "c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2"
    old_protocol_hash = save_doc(original_dir, "protocol", original)
    save_doc(original_dir, "freeze", dict(protocol_sha256=old_protocol_hash,
        source_implementation_sha256={"fixture": "only-planning"},
        champion_model_sha256="champion", split_manifest_sha256="split"))
    save_doc(directory, "validation_protocol", dict(factorial_source_protocol=original,
        fixed_hybrid_metadata_sha256="hybrid"))
    save_doc(directory, "scenario_manifest", manifest)
    baseline_dir = directory / "baseline-fit-v1"
    claim_hash = save_doc(baseline_dir, "claim", dict(methods=f.b.METHODS, inputs={}))
    claim_bytes = (baseline_dir / "claim.json").read_bytes()
    save_doc(baseline_dir, "metadata", dict(new_validation_cases_accessed=False,
        production_release_authorized=False, n2_calibrated=False, policy_calibrated=False,
        claim_sha256=claim_hash, files={"claim.json": f.d.sha(claim_bytes)},
        models={"hybrid_ridge_lightgbm_lightgbm": {"metadata_sha256": "hybrid"},
                **{k: {} for k in f.b.METHODS}}))
    monkeypatch.setattr(f.b, "prepare", lambda *_: (directory, None, None, None, {}, None))
    original_version = f.importlib.metadata.version
    monkeypatch.setattr(f.importlib.metadata, "version", lambda name:
        {"ctgan": "0.12.1", "torch": "fixture", "xgboost": "fixture"}.get(name)
        if name in ("ctgan", "torch", "xgboost") else original_version(name))
    documents = f.build_execution(tmp_path, C4)
    execution, scientific, frozen, _ = documents
    assert execution["future_evaluation_authorized"] is False
    assert execution["candidate_training_authorized"] is False
    assert scientific["protocol_version"] == f.VERSION
    assert frozen["full_execution_enabled"] is False
    assert not (directory / "admission-executor-v1").exists()
    f.materialize(tmp_path, documents)
    f.materialize(tmp_path, documents)
    f.verify_execution_binding(tmp_path, first_group(manifest))
    policy_group = next(g["group_id"] for g in manifest["groups"] if g["split_role"] == "policy_calibration")
    with pytest.raises(ValueError, match="only a frozen conformal"):
        f.verify_execution_binding(tmp_path, policy_group)
    (baseline_dir / "claim.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum"):
        f.verify_execution_binding(tmp_path, first_group(manifest))
