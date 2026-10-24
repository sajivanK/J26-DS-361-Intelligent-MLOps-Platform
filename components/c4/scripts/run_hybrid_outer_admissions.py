"""Bounded outer admissions after frozen N2 and policies; no future evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import _hybrid_outer_execution as f
from run_hybrid_fresh_admissions import existing_record

d, e, v = f.d, f.e, f.v


def run_group(root, c4, group_id, manifest, max_cases, runner=subprocess.run):
    root, c4 = Path(root), Path(c4)
    rows = f.outer_group_cases(manifest, group_id)
    frozen_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    _, freeze_hash = d.checked_json(frozen_dir, "freeze")
    output = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / group_id / "v1"
    _, execution_hash = d.checked_json(f.bundle_at(root) / "outer-admission-executor-v1", "claim")
    claim = dict(purpose="bounded_fresh_outer_group_admission", group_id=group_id,
        executor_claim_sha256=execution_hash, freeze_sha256=freeze_hash,
        scenario_ids=[r["scenario_id"] for r in rows], role="outer_test",
        ordering="shared_base_package_id_then_targeted_before_corrupted_then_scenario_id",
        no_outcome_based_case_selection=True, candidate_training_executed=False,
        future_evaluation_executed=False, production_release_authorized=False)
    f.ensure_documents(output, {"claim": claim})
    records = {}
    for row in rows:
        record = existing_record(root, row, freeze_hash)
        if record is not None:
            records[row["scenario_id"]] = record
    print("VERIFIED COMPLETED CASES:", len(records), "/ 37", flush=True)
    # A bounded admission call is a resource checkpoint, not selection based on
    # future performance. Every one of the 37 planned cases remains required.
    pending = [r for r in rows if r["scenario_id"] not in records]
    for row in pending[:max_cases]:
        sid = row["scenario_id"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = output / (d.sha(sid.encode())[:12]+"-"+stamp+"-admission.log")
        print("RUNNING FRESH ADMISSION:", sid, flush=True)
        print("LOG:", log_path, flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            completed = runner([sys.executable, str(c4 / "scripts/run_hybrid_outer_case_admission.py"),
                "--data-root", str(root), "--group-id", group_id, "--scenario-id", sid, "--execute-selected"],
                stdout=log, stderr=subprocess.STDOUT, text=True)
        d.require(completed.returncode == 0, "Admission failed; preserve artifacts and inspect "+str(log_path))
        record = existing_record(root, row, freeze_hash)
        d.require(record is not None, "Case did not produce a verified admission report.")
        records[sid] = record
        print("RECORDED:", record["simulation_admission"], "SCREEN:", record["screen_status"], flush=True)
    f.verify_execution_binding(root, group_id)
    if len(records) < len(rows):
        print("BOUNDED ADMISSION COMPLETE:", len(records), "/ 37 cases verified.")
        print("Remaining cases retained. Repeat --execute to resume; do not remove checkpoints.")
        print("CANDIDATE TRAINING: 0; FUTURE EVALUATION: 0; N2/POLICY FITS: 0")
        return
    admitted = sum(r["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION" for r in records.values())
    report = dict(purpose="fresh_outer_group_admission_results", group_id=group_id,
        freeze_sha256=freeze_hash, executor_claim_sha256=execution_hash,
        cases=37, records=[records[r["scenario_id"]] for r in rows],
        scoped_admissible_cases=admitted, blocked_cases=37-admitted,
        candidate_training_executed=False, future_evaluation_executed=False,
        outcome_based_selection=False, production_release_authorized=False)
    if (output / "group_admissions.json").exists():
        previous, _ = d.checked_json(output, "group_admissions")
        d.require(previous == report, "Completed group report differs.")
    else:
        content = e.encoded(report)
        (output / "group_admissions.json").write_bytes(content)
        (output / "group_admissions.sha256").write_text(d.sha(content)+"\n", encoding="utf-8")
    print("GROUP ADMISSION COMPLETE: 37 cases; admitted:", admitted, "blocked:", 37-admitted)
    print("REPORT:", output / "group_admissions.json")
    print("No candidate or future evaluation has been performed.")


def completed_group(root, group_id, manifest):
    rows = f.outer_group_cases(manifest, group_id)
    directory = Path(root) / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / group_id / "v1"
    if not (directory / "group_admissions.json").exists():
        return None
    report, digest = d.checked_json(directory, "group_admissions")
    _, freeze_hash = d.checked_json(Path(root) / "c4_artifacts/benchmark_protocols/adult" / v.VERSION, "freeze")
    _, executor_hash = d.checked_json(f.bundle_at(root) / "outer-admission-executor-v1", "claim")
    d.require(report["purpose"] == "fresh_outer_group_admission_results"
              and report["group_id"] == group_id and report["cases"] == 37
              and report["freeze_sha256"] == freeze_hash
              and report["executor_claim_sha256"] == executor_hash
              and report["candidate_training_executed"] is False
              and report["future_evaluation_executed"] is False
              and report["outcome_based_selection"] is False
              and report["production_release_authorized"] is False, "Policy group report scope differs.")
    records = [existing_record(root, row, freeze_hash) for row in rows]
    d.require(all(record is not None for record in records) and report["records"] == records,
              "Policy group completion differs from verified case artifacts.")
    admitted = sum(r["simulation_admission"] == "READY_FOR_SCOPED_CANDIDATE_VALIDATION" for r in records)
    d.require(report["scoped_admissible_cases"] == admitted and report["blocked_cases"] == 37-admitted,
              "Policy group admission totals differ.")
    return dict(group_id=group_id, group_admissions_sha256=digest, admitted=admitted, blocked=37-admitted)


def save_completion(root, claim, manifest):
    records = [completed_group(root, gid, manifest) for gid in claim["allowed_group_ids"]]
    if any(record is None for record in records):
        print("OUTER ADMISSION GROUPS COMPLETE:", sum(r is not None for r in records), "/ 16")
        return
    report = dict(purpose="all_fixed_fresh_outer_admissions_complete", assigned_groups=16, planned_cases=592,
        records=records, admitted_cases=sum(r["admitted"] for r in records),
        blocked_cases=sum(r["blocked"] for r in records),
        nonempty_groups=sum(r["admitted"] > 0 for r in records),
        candidate_training_executed=False, future_evaluation_executed=False,
        N1_refitted=False, N2_refitted=False, policy_fitted=False, outer_admissions_executed=True, outer_future_evaluation_executed=False,
        production_release_authorized=False)
    directory = f.bundle_at(root) / "outer-admission-executor-v1/completion-v1"
    f.ensure_documents(directory, {"outer_admissions": report})
    print("ALL 16 OUTER ADMISSION GROUPS COMPLETE; CASES: 592")
    print("ADMITTED:", report["admitted_cases"], "BLOCKED:", report["blocked_cases"],
          "NONEMPTY GROUPS:", report["nonempty_groups"])
    print("CANDIDATE / FUTURE / POLICY FITS: 0; OUTER FUTURE EVALUATIONS: 0")
    print("REPORT:", directory / "outer_admissions.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--group-id")
    selection.add_argument("--all-groups", action="store_true")
    p.add_argument("--max-cases", type=int, default=37)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    d.require(1 <= args.max_cases <= 37, "max-cases must be 1 through 37.")
    root, c4 = Path(raw), Path(__file__).resolve().parents[1]
    claim, manifest = f.build_execution(root, c4)
    selected = claim["allowed_group_ids"] if args.all_groups else [args.group_id or claim["allowed_group_ids"][0]]
    for gid in selected:
        f.outer_group_cases(manifest, gid)
    print("FROZEN FOUR-METHOD N2 AND RELEASE POLICIES VERIFIED BEFORE OUTER ADMISSIONS", flush=True)
    print("SELECTED OUTER GROUPS:", len(selected), "; 37 PLANNED CASES PER GROUP", flush=True)
    print("CANDIDATE TRAINING / FUTURE EVALUATION / POLICY FITTING: DISABLED", flush=True)
    if not args.execute:
        print("PLAN ONLY. No files written or generators fitted.")
        return
    f.ensure_documents(f.bundle_at(root) / "outer-admission-executor-v1", {"claim": claim})
    for number, gid in enumerate(selected, 1):
        f.verify_execution_binding(root, gid)
        if completed_group(root, gid, manifest) is not None:
            print("REUSED VERIFIED COMPLETE OUTER GROUP:", gid, flush=True)
            continue
        print("OUTER GROUP:", number, "/", len(selected), gid, flush=True)
        run_group(root, c4, gid, manifest, args.max_cases)
    save_completion(root, claim, manifest)


if __name__ == "__main__":
    main()
