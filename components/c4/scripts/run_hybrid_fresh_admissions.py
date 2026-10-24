"""Plan/run a bounded fresh conformal admission batch; resume completed cases.

No candidate fitting, new future outcomes, N2 calibration or policy evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import _hybrid_fresh_execution as f

d, e, v = f.d, f.e, f.v


def existing_record(root, row, freeze_hash):
    run = Path(root) / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]
    if not run.exists():
        return None
    d.require((run / "admission.json").is_file(),
              "Incomplete case exists; preserve its logs/artifacts and inspect before retry: "+str(run))
    admission, digest = d.checked_json(run, "admission")
    d.require(admission["scenario"] == row and admission["freeze_sha256"] == freeze_hash,
              "Existing case/protocol differs.")
    d.require(admission["candidate_training_executed"] is False
              and admission["future_evaluation_executed"] is False
              and admission["production_package_approved"] is False, "Existing case scope differs.")
    status = admission["simulation_admission"]
    d.require(status in ("READY_FOR_SCOPED_CANDIDATE_VALIDATION", "BLOCKED_ESCALATE"),
              "Invalid admission status.")
    package = Path(root) / "c4_artifacts/offline_repair/adult" / admission["artifact_version"]
    generation, generation_hash = d.checked_json(package)
    privacy, privacy_hash = d.checked_json(package, "privacy_diagnostics")
    d.require(generation_hash == admission["generation_metadata_sha256"]
              and privacy_hash == admission["privacy_report_sha256"]
              and privacy["generation_metadata_sha256"] == generation_hash, "Repair provenance differs.")
    for filename, checksum in (("synthetic_unvalidated.csv", generation["synthetic_csv_sha256"]),
                               ("generator.pkl", generation["generator_sha256"])):
        d.checked(package / filename, checksum)
    if generation.get("corruption_variant"):
        d.checked(package / "corruption_mask.csv", generation["corruption_variant"]["mask_sha256"])
    d.require(status != "READY_FOR_SCOPED_CANDIDATE_VALIDATION" or admission["screen_status"] == "PASS",
              "Admitted case lacks a passing screen.")
    return {key: row[key] for key in ("scenario_id", "group_id", "split_role", "mechanism", "magnitude",
        "diagnosis_condition", "repair_condition", "is_no_drift_control")} | {
        "simulation_admission": status, "screen_status": admission["screen_status"],
        "tail_distance_ratio": admission["tail_distance_ratio"], "artifact_version": admission["artifact_version"],
        "admission_sha256": digest}


def run_group(root, c4, group_id, manifest, max_cases, runner=subprocess.run):
    root, c4 = Path(root), Path(c4)
    rows = f.fresh_group_cases(manifest, group_id)
    frozen_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    _, freeze_hash = d.checked_json(frozen_dir, "freeze")
    output = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / "group_admissions" / group_id / "v1"
    _, execution_hash = d.checked_json(f.bundle_at(root) / "admission-executor-v1", "claim")
    claim = dict(purpose="bounded_fresh_conformal_group_admission", group_id=group_id,
        executor_claim_sha256=execution_hash, freeze_sha256=freeze_hash,
        scenario_ids=[r["scenario_id"] for r in rows], role="conformal_calibration",
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
            completed = runner([sys.executable, str(c4 / "scripts/run_hybrid_fresh_case_admission.py"),
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
    report = dict(purpose="fresh_conformal_group_admission_results", group_id=group_id,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--group-id")
    p.add_argument("--max-cases", type=int, default=37, help="Maximum new admissions this invocation; completed ones reused.")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    d.require(1 <= args.max_cases <= 37, "max-cases must be 1 through 37.")
    root, c4 = Path(raw), Path(__file__).resolve().parents[1]
    documents = f.build_execution(root, c4)
    manifest = documents[3]
    group = args.group_id or next(g["group_id"] for g in manifest["groups"]
        if g["split_role"] == "conformal_calibration")
    rows = f.fresh_group_cases(manifest, group)
    print("SELECTED GROUP:", group, "ROLE: conformal_calibration; 37 planned cases")
    print("FIRST CASE:", rows[0]["scenario_id"])
    print("MAXIMUM NEW ADMISSIONS THIS CALL:", args.max_cases)
    print("CANDIDATE TRAINING AND FUTURE EVALUATION: DISABLED")
    if not args.execute:
        print("PLAN ONLY. No files written or generation executed.")
        return
    f.materialize(root, documents)
    f.verify_execution_binding(root, group)
    run_group(root, c4, group, manifest, args.max_cases)


if __name__ == "__main__":
    main()
