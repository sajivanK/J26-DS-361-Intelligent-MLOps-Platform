"""Fresh conformal candidate-stage bindings and Windows-safe model saving."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import tempfile

import _hybrid_fresh_execution as f
import run_hybrid_fresh_admissions as a

d, e, v = f.d, f.e, f.v


def environment():
    return {**e.environment(), **{p: importlib.metadata.version(p) for p in ("ctgan", "torch", "xgboost")}}


def expected_validator_source(source):
    text = source.replace("in a frozen training group.", "in a frozen fresh conformal-calibration group.")
    text = text.replace("from _general_training import group_context",
        "from _hybrid_fresh_candidate import group_context, verify_scope, save_candidate_model_portable")
    text = text.replace("root = resolve_data_root(args.data_root)",
        "root = resolve_data_root(args.data_root)\n    verify_scope(root, SELECTED)")
    text = text.replace("adult-simulation-pp1-v2", v.VERSION)
    text = text.replace("rows[0]['split_role'] == 'forecast_train'", "rows[0]['split_role'] == 'conformal_calibration'")
    return text.replace("model.save_model(directory / 'candidate.ubj')",
                        "save_candidate_model_portable(model, directory / 'candidate.ubj')")


def save_candidate_model_portable(model, destination):
    """Native save uses a short path; Python copies the exact bytes to storage."""
    destination = Path(destination)
    d.require(not destination.exists(), "Candidate model already exists; never overwrite.")
    with tempfile.TemporaryDirectory(prefix="c4save-") as temporary:
        path = Path(temporary) / "candidate.ubj"
        d.require(len(str(path)) < 240, "Temporary native model path is too long.")
        model.save_model(path)
        contents = path.read_bytes()
        d.require(bool(contents), "Native model save produced empty bytes.")
        with destination.open("xb") as stream:
            stream.write(contents)
        d.checked(destination, d.sha(contents))


def inventory(root):
    root = Path(root)
    bundle = f.bundle_at(root)
    manifest, _ = d.checked_json(bundle, "scenario_manifest")
    frozen_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    frozen, freeze_hash = d.checked_json(frozen_dir, "freeze")
    selected, reports, group_rows = [], {}, []
    groups = [g for g in manifest["groups"] if g["split_role"] == "conformal_calibration"]
    d.require(len(groups) == 24, "Expected all 24 frozen calibration groups.")
    for group in groups:
        gid = group["group_id"]
        rows = f.fresh_group_cases(manifest, gid)
        directory = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / gid / "v1"
        report, digest = d.checked_json(directory, "group_admissions")
        d.require(report["group_id"] == gid and report["cases"] == 37
                  and report["freeze_sha256"] == freeze_hash
                  and report["candidate_training_executed"] is False
                  and report["future_evaluation_executed"] is False, "Admission group scope differs.")
        records = report["records"]
        by_id = {r["scenario_id"]: r for r in records}
        d.require(len(records) == len(by_id) == 37 and set(by_id) == {r["scenario_id"] for r in rows},
                  "Admission records missing/duplicated or outside manifest.")
        admitted = 0
        for row in rows:
            record = by_id[row["scenario_id"]]
            fields = ("group_id", "split_role", "mechanism", "magnitude", "diagnosis_condition", "repair_condition", "is_no_drift_control")
            d.require(all(record[k] == row[k] for k in fields), "Admission factor/role differs.")
            d.require(record["simulation_admission"] in ("READY_FOR_SCOPED_CANDIDATE_VALIDATION", "BLOCKED_ESCALATE"),
                      "Unknown admission state.")
            if record["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION":
                d.require(record["screen_status"] == "PASS", "Admitted record lacks a passing screen.")
                selected.append(row)
                admitted += 1
        d.require(report["scoped_admissible_cases"] == admitted and report["blocked_cases"] == 37-admitted,
                  "Admission totals differ.")
        reports[str(directory / "group_admissions.json")] = digest
        group_rows.append(dict(group_id=gid, admitted=admitted, blocked=37-admitted))
    return manifest, frozen, freeze_hash, selected, reports, group_rows


def prepare(root, c4):
    root, c4 = Path(root), Path(c4)
    documents = f.build_execution(root, c4)
    existing, admission_execution_hash = d.checked_json(f.bundle_at(root) / "admission-executor-v1", "claim")
    d.require(e.encoded(documents[0]) == e.encoded(existing), "Admission execution binding changed.")
    manifest, frozen, freeze_hash, selected, reports, groups = inventory(root)
    d.require(manifest == documents[3], "Fresh scenario manifest differs.")
    old_text = (c4 / "scripts/validate_portable_training_candidate.py").read_text(encoding="utf-8-sig")
    new_text = (c4 / "scripts/validate_hybrid_fresh_candidate.py").read_text(encoding="utf-8-sig")
    d.require(new_text == expected_validator_source(old_text), "Candidate science changed beyond the declared adapter.")
    pins = {**existing["inputs"], **reports,
        str(f.bundle_at(root) / "admission-executor-v1/claim.json"): admission_execution_hash,
        str(root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION / "freeze.json"): freeze_hash}
    for name in ("_hybrid_fresh_candidate.py", "validate_hybrid_fresh_candidate.py", "run_hybrid_fresh_candidates.py"):
        path = c4 / "scripts" / name
        pins[str(path)] = d.sha(path.read_bytes())
    for row in selected:
        record = a.existing_record(root, row, freeze_hash)
        d.require(record is not None and record["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION",
                  "Selected admission artifact is missing or blocked.")
        run = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]
        pins[str(run / "admission.json")] = record["admission_sha256"]
    claim = dict(purpose="fresh_conformal_candidate_pre_release_execution_v1", environment=environment(), inputs=pins,
        admission_executor_sha256=admission_execution_hash, scenario_ids=[r["scenario_id"] for r in selected],
        group_counts=groups, assigned_groups=24, nonempty_admission_groups=sum(g["admitted"] > 0 for g in groups),
        admitted_cases=len(selected), blocked_cases=888-len(selected), role="conformal_calibration",
        candidate_training_authorized=True, future_evaluation_authorized=False,
        generator_refitting_authorized=False, N1_refitting_authorized=False, production_release_authorized=False,
        training="unchanged champion configuration; reference plus admitted synthetic repair only",
        scoring="unchanged pre-release global/diagnosed-slice/complement evaluation",
        native_io="short temporary native save and byte-identical copy; existing verified portable loaders")
    return claim, selected


def verify_scope(root, scenario_id):
    root = Path(root)
    claim, _ = d.checked_json(f.bundle_at(root) / "candidate-executor-v1", "claim")
    d.require(claim["role"] == "conformal_calibration" and claim["candidate_training_authorized"] is True
              and claim["future_evaluation_authorized"] is False
              and claim["generator_refitting_authorized"] is False
              and claim["N1_refitting_authorized"] is False
              and claim["production_release_authorized"] is False, "Invalid candidate execution scope.")
    d.require(claim["environment"] == environment(), "Candidate execution environment changed.")
    d.require(scenario_id in claim["scenario_ids"], "Scenario not admitted in the frozen candidate execution claim.")
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
    manifest, _ = d.checked_json(f.bundle_at(root), "scenario_manifest")
    row = next(r for r in manifest["scenarios"] if r["scenario_id"] == scenario_id)
    f.verify_execution_binding(root, row["group_id"])
    return row


def group_context(root, scenario_id):
    root = Path(root)
    row = verify_scope(root, scenario_id)
    directory = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / row["group_id"] / "v1"
    report, digest = d.checked_json(directory, "group_admissions")
    matches = [r for r in report["records"] if r["scenario_id"] == scenario_id]
    d.require(len(matches) == 1 and matches[0]["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION",
              "Scenario is not admitted in this completed group.")
    record = matches[0]
    _, freeze_hash = d.checked_json(root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION, "freeze")
    actual = a.existing_record(root, row, freeze_hash)
    d.require(actual == record, "Admission record no longer matches verified artifacts.")
    return report, digest, record


def completed_case(root, row):
    root = Path(root)
    run = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]
    directory = run / "scoped_candidate_validation-v1"
    if not directory.exists():
        d.require(not (root / "c4_artifacts/pre_release/adult" / row["scenario_id"]).exists(),
                  "Orphan pre-release evidence exists; inspect it before execution.")
        return None
    d.require((directory / "validation.json").is_file(),
              "Incomplete candidate stage exists; preserve artifacts/logs and inspect: "+str(directory))
    report, report_hash = d.checked_json(directory, "validation")
    metadata, metadata_hash = d.checked_json(directory)
    d.require(report["scenario"] == row and report["candidate_training_executed"] is True
              and report["candidate_reload_verified"] is True and report["future_evaluation_executed"] is False
              and report["production_package_approved"] is False
              and report["candidate_metadata_sha256"] == metadata_hash, "Candidate report/state differs.")
    d.require(metadata["scenario_id"] == row["scenario_id"] and metadata["role"] == "candidate"
              and metadata["scope"] == "controlled_offline_simulation"
              and metadata["release_validation_used_for_training"] is False
              and metadata["sealed_future_used_for_training"] is False, "Candidate training scope differs.")
    d.checked(directory / "candidate.ubj", metadata["model_sha256"])
    source_claim, _ = d.checked_json(directory, "claim")
    c4 = Path(__file__).resolve().parents[1]
    for path, digest in source_claim["source_sha256"].items():
        d.checked(c4 / path, digest)
    admission, admission_hash = d.checked_json(run, "admission")
    d.require(admission["scenario"] == row and admission["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION"
              and report["admission_sha256"] == metadata["admission_sha256"] == admission_hash
              and report["freeze_sha256"] == metadata["freeze_sha256"] == admission["freeze_sha256"],
              "Candidate admission binding differs.")
    evidence_dir = Path(report["pre_release_directory"])
    d.require(evidence_dir == root / "c4_artifacts/pre_release/adult" / row["scenario_id"],
              "Unexpected pre-release artifact path.")
    evidence, evidence_hash = d.checked_json(evidence_dir, "evidence")
    d.require(evidence_hash == report["pre_release_evidence_sha256"]
              and evidence["scenario_id"] == row["scenario_id"]
              and evidence["candidate_model_sha256"] == metadata["model_sha256"]
              and evidence["champion_model_sha256"] == metadata["champion_model_sha256"]
              and evidence["evaluation_partition"] == "release_validation"
              and evidence["slice_basis"] == "c2_diagnosed_slice", "Pre-release evidence binding differs.")
    for filename, key in (("clean_release_validation.csv", "clean_release_validation_csv_sha256"),
        ("release_validation.csv", "release_validation_csv_sha256"), ("diagnosed_membership.csv", "diagnosed_membership_csv_sha256")):
        d.checked(evidence_dir / filename, evidence[key])
    return dict(scenario_id=row["scenario_id"], group_id=row["group_id"], split_role=row["split_role"],
        candidate_validation_sha256=report_hash, candidate_metadata_sha256=metadata_hash,
        candidate_model_sha256=metadata["model_sha256"], pre_release_evidence_sha256=evidence_hash,
        candidate_training_executed=True, future_evaluation_executed=False)
