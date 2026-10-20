"""Candidate adapter tests; fixtures are technical checks, not research results."""
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
import _hybrid_fresh_candidate as c
import run_hybrid_fresh_candidates as r


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Use extended Windows paths for fixtures that deliberately exceed MAX_PATH.

    The actual experiment already uses C4_DATA_ROOT with its extended prefix.
    The native model SAVE still receives a separate, short temporary filename.
    """
    directory = tmp_path_factory.mktemp("c4candidate")
    if os.name == "nt":
        raw = str(directory.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw
        directory = Path(raw)
    yield directory
    # Explicit prefix-aware cleanup keeps pytest's ordinary parent directory
    # free of intentionally long children on Windows.
    if os.name == "nt" and directory.exists():
        shutil.rmtree(directory)


def save_doc(directory, stem, data):
    directory.mkdir(parents=True, exist_ok=True)
    content = c.e.encoded(data)
    (directory / (stem+".json")).write_bytes(content)
    (directory / (stem+".sha256")).write_text(c.d.sha(content)+"\n")
    return c.d.sha(content)


@pytest.fixture
def manifest():
    original = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    return c.v.create_design(original)


def test_exact_original_training_and_scoring_reuse():
    original = (C4 / "scripts/validate_portable_training_candidate.py").read_text(encoding="utf-8-sig")
    adapted = (C4 / "scripts/validate_hybrid_fresh_candidate.py").read_text(encoding="utf-8-sig")
    assert c.expected_validator_source(original) == adapted
    assert "model = train_champion(training, config)" in adapted
    assert "save_candidate_model_portable(model, directory / 'candidate.ubj')" in adapted
    assert "verify_scope(root, SELECTED)" in adapted
    assert "adult-simulation-pp1-v2" not in adapted


def test_real_native_save_long_destination_and_reload(tmp_path, monkeypatch):
    from xgboost import XGBClassifier
    from _portable_model_io import load_candidate
    from c4.data import FEATURE_COLUMNS
    rng = np.random.default_rng(53)
    x = pd.DataFrame(rng.normal(size=(40, 12)), columns=FEATURE_COLUMNS)
    y = (x.iloc[:, 0] > 0).astype(int)
    model = XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, tree_method="hist").fit(x, y)
    native_save = model.save_model
    observed = []
    def checked_save(path):
        observed.append(str(path))
        assert len(str(path)) < 240
        native_save(path)
    monkeypatch.setattr(model, "save_model", checked_save)
    directory = tmp_path / ("a"*100) / ("b"*100) / ("c"*30)
    directory.mkdir(parents=True)
    destination = directory / "candidate.ubj"
    assert len(str(destination)) > 260
    c.save_candidate_model_portable(model, destination)
    assert observed and str(destination) not in observed
    save_doc(directory, "metadata", dict(role="candidate", feature_columns=list(FEATURE_COLUMNS),
        model_sha256=c.d.sha(destination.read_bytes())))
    restored, _ = load_candidate(directory)
    np.testing.assert_array_equal(model.predict(x), restored.predict(x))
    np.testing.assert_allclose(model.predict_proba(x), restored.predict_proba(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="already exists"):
        c.save_candidate_model_portable(model, destination)


def make_inventory(root, manifest):
    save_doc(c.f.bundle_at(root), "scenario_manifest", manifest)
    frozen = root / "c4_artifacts/benchmark_protocols/adult" / c.v.VERSION
    freeze_hash = save_doc(frozen, "freeze", {})
    assigned = [g for g in manifest["groups"] if g["split_role"] == "conformal_calibration"]
    report_paths = []
    for position, group in enumerate(assigned):
        rows = c.f.fresh_group_cases(manifest, group["group_id"])
        records = []
        count = 0 if position < 4 else 2
        for i, row in enumerate(rows):
            keys = ("scenario_id", "group_id", "split_role", "mechanism", "magnitude", "diagnosis_condition", "repair_condition", "is_no_drift_control")
            records.append({k: row[k] for k in keys} | dict(
                simulation_admission="READY_FOR_SCOPED_CANDIDATE_VALIDATION" if i < count else "BLOCKED_ESCALATE",
                screen_status="PASS" if i < count else "FAIL"))
        directory = root / "c4_artifacts/benchmark_runs/adult" / c.v.VERSION / "group_admissions" / group["group_id"] / "v1"
        save_doc(directory, "group_admissions", dict(group_id=group["group_id"], cases=37, records=records,
            freeze_sha256=freeze_hash, candidate_training_executed=False, future_evaluation_executed=False,
            scoped_admissible_cases=count, blocked_cases=37-count))
        report_paths.append(directory)
    return report_paths


def test_complete_admission_inventory_and_empty_groups_retained(manifest, tmp_path):
    make_inventory(tmp_path, manifest)
    _, _, _, selected, pins, groups = c.inventory(tmp_path)
    assert len(selected) == 40
    assert len(groups) == len(pins) == 24
    assert sum(g["admitted"] == 0 for g in groups) == 4
    assert sum(g["blocked"] for g in groups) == 848
    assert all(row["split_role"] == "conformal_calibration" for row in selected)


@pytest.mark.parametrize("change", ["missing", "duplicate", "role", "counts", "status"])
def test_bad_inventory_rejected(manifest, tmp_path, change):
    paths = make_inventory(tmp_path, manifest)
    path = paths[5]
    report, _ = c.d.checked_json(path, "group_admissions")
    if change == "missing":
        (path / "group_admissions.json").unlink()
        with pytest.raises(FileNotFoundError):
            c.inventory(tmp_path)
        return
    if change == "duplicate":
        report["records"][1] = dict(report["records"][0])
    elif change == "role":
        report["records"][0]["split_role"] = "outer_test"
    elif change == "counts":
        report["scoped_admissible_cases"] += 1
    else:
        report["records"][0]["simulation_admission"] = "UNKNOWN"
    save_doc(path, "group_admissions", report)
    with pytest.raises(ValueError):
        c.inventory(tmp_path)


def test_scope_binding_and_input_tampering(manifest, tmp_path, monkeypatch):
    row = next(s for s in manifest["scenarios"] if s["split_role"] == "conformal_calibration")
    source = tmp_path / "input"
    source.write_bytes(b"verified")
    claim = dict(role="conformal_calibration", candidate_training_authorized=True,
        future_evaluation_authorized=False, generator_refitting_authorized=False,
        N1_refitting_authorized=False, production_release_authorized=False,
        environment={"fixture": True}, scenario_ids=[row["scenario_id"]],
        inputs={str(source): c.d.sha(source.read_bytes())})
    save_doc(c.f.bundle_at(tmp_path) / "candidate-executor-v1", "claim", claim)
    save_doc(c.f.bundle_at(tmp_path), "scenario_manifest", manifest)
    monkeypatch.setattr(c, "environment", lambda: {"fixture": True})
    monkeypatch.setattr(c.f, "verify_execution_binding", lambda *_: manifest)
    assert c.verify_scope(tmp_path, row["scenario_id"]) == row
    with pytest.raises(ValueError, match="not admitted"):
        c.verify_scope(tmp_path, "blocked-or-invented")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        c.verify_scope(tmp_path, row["scenario_id"])


@pytest.mark.parametrize("key", ["future_evaluation_authorized", "generator_refitting_authorized",
    "N1_refitting_authorized", "production_release_authorized"])
def test_forbidden_execution_flags(tmp_path, key):
    claim = dict(role="conformal_calibration", candidate_training_authorized=True,
        future_evaluation_authorized=False, generator_refitting_authorized=False,
        N1_refitting_authorized=False, production_release_authorized=False)
    claim[key] = True
    save_doc(c.f.bundle_at(tmp_path) / "candidate-executor-v1", "claim", claim)
    with pytest.raises(ValueError, match="Invalid candidate execution"):
        c.verify_scope(tmp_path, "any")


def test_incomplete_and_orphan_candidate_stages(manifest, tmp_path):
    row = next(s for s in manifest["scenarios"] if s["split_role"] == "conformal_calibration")
    assert c.completed_case(tmp_path, row) is None
    directory = tmp_path / "c4_artifacts/benchmark_runs/adult" / c.v.VERSION / row["scenario_id"] / "scoped_candidate_validation-v1"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete candidate"):
        c.completed_case(tmp_path, row)


def test_bounded_batch_resume_and_completion(manifest, tmp_path, monkeypatch):
    rows = [s for s in manifest["scenarios"] if s["split_role"] == "conformal_calibration"][:2]
    claim = dict(inputs={}, nonempty_admission_groups=1, blocked_cases=35)
    completed = {}
    calls = []
    monkeypatch.setattr(c, "completed_case", lambda root, row: completed.get(row["scenario_id"]))
    monkeypatch.setattr(c, "verify_scope", lambda *_: None)
    def runner(command, **kwargs):
        sid = command[command.index("--scenario-id")+1]
        assert "validate_hybrid_fresh_candidate.py" in command[1]
        assert "--execute" in command
        completed[sid] = dict(scenario_id=sid, future_evaluation_executed=False)
        calls.append(sid)
        return SimpleNamespace(returncode=0)
    r.run_batch(tmp_path, C4, claim, rows, 1, runner)
    assert len(calls) == 1
    output = c.f.bundle_at(tmp_path) / "candidate-executor-v1/completion-v1"
    assert not output.exists()
    r.run_batch(tmp_path, C4, claim, rows, 2, runner)
    assert len(calls) == 2
    report, _ = c.d.checked_json(output, "candidate_batch")
    assert report["candidates"] == 2
    assert report["future_evaluation_executed"] is False
    assert report["N1_forecasts_frozen"] is False
    r.run_batch(tmp_path, C4, claim, rows, 2, runner)
    assert len(calls) == 2


def test_subprocess_failure_does_not_claim_completion(manifest, tmp_path, monkeypatch):
    rows = [s for s in manifest["scenarios"] if s["split_role"] == "conformal_calibration"][:1]
    monkeypatch.setattr(c, "completed_case", lambda *_: None)
    monkeypatch.setattr(c, "verify_scope", lambda *_: None)
    with pytest.raises(ValueError, match="Candidate stage failed"):
        r.run_batch(tmp_path, C4, dict(inputs={}), rows, 1,
                    lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert not (c.f.bundle_at(tmp_path) / "candidate-executor-v1/completion-v1").exists()
