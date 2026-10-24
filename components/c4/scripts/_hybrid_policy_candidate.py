"""Policy candidate/pre-release stage; unchanged training and scoring science."""
from __future__ import annotations
import json
from pathlib import Path

import _hybrid_policy_execution as f
import run_hybrid_policy_admissions as a
import _hybrid_fresh_candidate as original
from _hybrid_fresh_candidate import environment, completed_case, save_candidate_model_portable

d, e, v = f.d, f.e, f.v


def expected_validator_source(source):
    return original.expected_validator_source(source).replace(
        "fresh conformal-calibration", "fresh policy-calibration").replace(
        "from _hybrid_fresh_candidate import", "from _hybrid_policy_candidate import").replace(
        "'conformal_calibration'", "'policy_calibration'")


def inventory(root):
    """Verify all 444 admissions; retain blocked cases and every assigned group."""
    root = Path(root)
    manifest, _ = d.checked_json(f.bundle_at(root), "scenario_manifest")
    frozen, freeze_hash = d.checked_json(root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION, "freeze")
    execution, _ = d.checked_json(f.bundle_at(root) / "policy-admission-executor-v1", "claim")
    completion, _ = d.checked_json(f.bundle_at(root) / "policy-admission-executor-v1/completion-v1", "policy_admissions")
    d.require(completion["purpose"] == "all_fixed_fresh_policy_admissions_complete"
              and completion["assigned_groups"] == 12 and completion["planned_cases"] == 444
              and completion["candidate_training_executed"] is False
              and completion["future_evaluation_executed"] is False
              and completion["N1_refitted"] is False and completion["N2_refitted"] is False
              and completion["policy_fitted"] is False and completion["outer_test_accessed"] is False
              and completion["production_release_authorized"] is False,
              "Complete admission-only policy batch is required.")
    groups = [g["group_id"] for g in manifest["groups"] if g["split_role"] == "policy_calibration"]
    d.require(len(groups) == 12 and groups == execution["allowed_group_ids"], "Policy group assignment differs.")
    selected, reports, counts, verified = [], {}, [], []
    for gid in groups:
        result = a.completed_group(root, gid, manifest)
        d.require(result is not None, "A policy admission group is incomplete.")
        verified.append(result)
        counts.append(dict(group_id=gid, admitted=result["admitted"], blocked=result["blocked"]))
        path = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / gid / "v1/group_admissions.json"
        reports[str(path)] = result["group_admissions_sha256"]
        for row in f.policy_group_cases(manifest, gid):
            record = a.existing_record(root, row, freeze_hash)
            d.require(record is not None, "Missing policy admission record.")
            if record["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION":
                selected.append(row)
    d.require(completion["records"] == verified
              and completion["admitted_cases"] == len(selected)
              and completion["blocked_cases"] == 444-len(selected)
              and completion["nonempty_groups"] == sum(g["admitted"] > 0 for g in counts),
              "Policy admission aggregate differs from verified case records.")
    return manifest, frozen, freeze_hash, selected, reports, counts


def prepare(root, c4):
    root, c4 = Path(root), Path(c4)
    execution, expected_manifest = f.build_execution(root, c4)
    existing, execution_hash = d.checked_json(f.bundle_at(root) / "policy-admission-executor-v1", "claim")
    d.require(existing == execution, "Policy admission execution binding changed.")
    manifest, frozen, freeze_hash, selected, reports, groups = inventory(root)
    d.require(manifest == expected_manifest, "Policy scenario manifest differs.")
    old_text = (c4 / "scripts/validate_portable_training_candidate.py").read_text(encoding="utf-8-sig")
    new_text = (c4 / "scripts/validate_hybrid_policy_candidate.py").read_text(encoding="utf-8-sig")
    d.require(new_text == expected_validator_source(old_text), "Policy candidate training/scoring differs beyond the role adapter.")
    completion_dir = f.bundle_at(root) / "policy-admission-executor-v1/completion-v1"
    _, completion_hash = d.checked_json(completion_dir, "policy_admissions")
    pins = {**existing["inputs"], **reports,
        str(f.bundle_at(root) / "policy-admission-executor-v1/claim.json"): execution_hash,
        str(completion_dir / "policy_admissions.json"): completion_hash,
        str(root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION / "freeze.json"): freeze_hash}
    for name in ("_hybrid_policy_candidate.py", "validate_hybrid_policy_candidate.py", "run_hybrid_policy_candidates.py"):
        path = c4 / "scripts" / name
        pins[str(path)] = d.sha(path.read_bytes())
    for row in selected:
        record = a.existing_record(root, row, freeze_hash)
        pins[str(root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"] / "admission.json")] = record["admission_sha256"]
    claim = dict(purpose="fresh_policy_candidate_pre_release_execution_v1", environment=environment(), inputs=pins,
        admission_executor_sha256=execution_hash, scenario_ids=[r["scenario_id"] for r in selected],
        group_counts=groups, assigned_groups=12, nonempty_admission_groups=sum(g["admitted"] > 0 for g in groups),
        admitted_cases=len(selected), blocked_cases=444-len(selected), role="policy_calibration",
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
    claim, _ = d.checked_json(f.bundle_at(root) / "policy-candidate-executor-v1", "claim")
    d.require(claim["role"] == "policy_calibration" and claim["candidate_training_authorized"] is True
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
    d.require(row["split_role"] == "policy_calibration", "Only admitted policy-role candidates may be trained.")
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
