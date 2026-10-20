"""Outer candidate/pre-release stage; unchanged training and scoring science."""
from __future__ import annotations
import json
from pathlib import Path

import _hybrid_outer_execution as f
import run_hybrid_outer_admissions as a
import _hybrid_fresh_candidate as original
from _hybrid_fresh_candidate import environment, completed_case, save_candidate_model_portable

d, e, v = f.d, f.e, f.v


def expected_validator_source(source):
    return original.expected_validator_source(source).replace(
        "fresh conformal-calibration", "fresh outer-test").replace(
        "from _hybrid_fresh_candidate import", "from _hybrid_outer_candidate import").replace(
        "'conformal_calibration'", "'outer_test'")


def inventory(root):
    """Verify all 592 admissions; retain blocked cases and every assigned group."""
    root = Path(root)
    manifest, _ = d.checked_json(f.bundle_at(root), "scenario_manifest")
    frozen, freeze_hash = d.checked_json(root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION, "freeze")
    execution, _ = d.checked_json(f.bundle_at(root) / "outer-admission-executor-v1", "claim")
    completion, _ = d.checked_json(f.bundle_at(root) / "outer-admission-executor-v1/completion-v1", "outer_admissions")
    d.require(completion["purpose"] == "all_fixed_fresh_outer_admissions_complete"
              and completion["assigned_groups"] == 16 and completion["planned_cases"] == 592
              and completion["candidate_training_executed"] is False
              and completion["future_evaluation_executed"] is False
              and completion["N1_refitted"] is False and completion["N2_refitted"] is False
              and completion["policy_fitted"] is False
              and completion["outer_admissions_executed"] is True
              and completion["outer_future_evaluation_executed"] is False
              and completion["production_release_authorized"] is False,
              "Complete admission-only outer batch is required.")
    groups = [g["group_id"] for g in manifest["groups"] if g["split_role"] == "outer_test"]
    d.require(len(groups) == 16 and groups == execution["allowed_group_ids"], "Outer group assignment differs.")
    selected, reports, counts, verified = [], {}, [], []
    for gid in groups:
        result = a.completed_group(root, gid, manifest)
        d.require(result is not None, "An outer admission group is incomplete.")
        verified.append(result)
        counts.append(dict(group_id=gid, admitted=result["admitted"], blocked=result["blocked"]))
        path = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / gid / "v1/group_admissions.json"
        reports[str(path)] = result["group_admissions_sha256"]
        for row in f.outer_group_cases(manifest, gid):
            record = a.existing_record(root, row, freeze_hash)
            d.require(record is not None, "Missing outer admission record.")
            if record["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION":
                selected.append(row)
    d.require(completion["records"] == verified
              and completion["admitted_cases"] == len(selected)
              and completion["blocked_cases"] == 592-len(selected)
              and completion["nonempty_groups"] == sum(g["admitted"] > 0 for g in counts),
              "Outer admission aggregate differs from verified case records.")
    return manifest, frozen, freeze_hash, selected, reports, counts


def prepare(root, c4):
    root, c4 = Path(root), Path(c4)
    execution, expected_manifest = f.build_execution(root, c4)
    existing, execution_hash = d.checked_json(f.bundle_at(root) / "outer-admission-executor-v1", "claim")
    d.require(existing == execution, "Outer admission execution binding changed.")
    manifest, frozen, freeze_hash, selected, reports, groups = inventory(root)
    d.require(manifest == expected_manifest, "Outer scenario manifest differs.")
    old_text = (c4 / "scripts/validate_portable_training_candidate.py").read_text(encoding="utf-8-sig")
    new_text = (c4 / "scripts/validate_hybrid_outer_candidate.py").read_text(encoding="utf-8-sig")
    d.require(new_text == expected_validator_source(old_text), "Outer candidate training/scoring differs beyond the role adapter.")
    completion_dir = f.bundle_at(root) / "outer-admission-executor-v1/completion-v1"
    _, completion_hash = d.checked_json(completion_dir, "outer_admissions")
    pins = {**existing["inputs"], **reports,
        str(f.bundle_at(root) / "outer-admission-executor-v1/claim.json"): execution_hash,
        str(completion_dir / "outer_admissions.json"): completion_hash,
        str(root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION / "freeze.json"): freeze_hash}
    for name in ("_hybrid_outer_candidate.py", "validate_hybrid_outer_candidate.py", "run_hybrid_outer_candidates.py"):
        path = c4 / "scripts" / name
        pins[str(path)] = d.sha(path.read_bytes())
    for row in selected:
        record = a.existing_record(root, row, freeze_hash)
        pins[str(root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"] / "admission.json")] = record["admission_sha256"]
    claim = dict(purpose="fresh_outer_candidate_pre_release_execution_v1", environment=environment(), inputs=pins,
        admission_executor_sha256=execution_hash, policy_selection_sha256=existing["policy_selection_sha256"], scenario_ids=[r["scenario_id"] for r in selected],
        group_counts=groups, assigned_groups=16, nonempty_admission_groups=sum(g["admitted"] > 0 for g in groups),
        admitted_cases=len(selected), blocked_cases=592-len(selected), role="outer_test",
        candidate_training_authorized=True, future_evaluation_authorized=False,
        generator_refitting_authorized=False, N1_refitting_authorized=False, N2_refitting_authorized=False,
        policy_fitting_authorized=False, outer_evaluation_authorized=False, production_release_authorized=False,
        training="unchanged champion configuration; reference plus admitted synthetic repair only",
        scoring="unchanged pre-release global/diagnosed-slice/complement evaluation",
        native_io="existing verified bytearray loaders and short temporary native saves")
    for path, digest in pins.items():
        d.checked(Path(path), digest)
    return claim, selected


def verify_scope(root, scenario_id):
    root = Path(root)
    claim, _ = d.checked_json(f.bundle_at(root) / "outer-candidate-executor-v1", "claim")
    d.require(claim["purpose"] == "fresh_outer_candidate_pre_release_execution_v1"
              and claim["role"] == "outer_test" and claim["candidate_training_authorized"] is True
              and claim["future_evaluation_authorized"] is False
              and claim["generator_refitting_authorized"] is False
              and claim["N1_refitting_authorized"] is False
              and claim["N2_refitting_authorized"] is False
              and claim["policy_fitting_authorized"] is False
              and claim["outer_evaluation_authorized"] is False
              and claim["production_release_authorized"] is False, "Invalid candidate execution scope.")
    d.require(claim["environment"] == environment(), "Candidate execution environment changed.")
    d.require(scenario_id in claim["scenario_ids"], "Scenario not admitted in the frozen candidate execution claim.")
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
    manifest, _ = d.checked_json(f.bundle_at(root), "scenario_manifest")
    row = next(r for r in manifest["scenarios"] if r["scenario_id"] == scenario_id)
    d.require(row["split_role"] == "outer_test", "Only admitted outer-role candidates may be trained.")
    f.verify_execution_binding(root, row["group_id"])
    admission, _ = d.checked_json(f.bundle_at(root) / "outer-admission-executor-v1", "claim")
    d.require(claim["policy_selection_sha256"] == admission["policy_selection_sha256"],
              "Candidate policy-selection binding differs from outer admission authorization.")
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

