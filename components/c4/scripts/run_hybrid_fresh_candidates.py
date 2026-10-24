"""Train admitted fresh conformal candidates and measure pre-release performance.

Completed repairs are reused. No TVAE fitting, future evaluation or N1/N2 fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import _hybrid_fresh_candidate as c

d, e, f = c.d, c.e, c.f


def run_batch(root, c4, claim, rows, max_cases, runner=subprocess.run):
    root, c4 = Path(root), Path(c4)
    directory = f.bundle_at(root) / "candidate-executor-v1"
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
    f.ensure_documents(directory, {"claim": claim})
    records = {}
    for row in rows:
        result = c.completed_case(root, row)
        if result is not None:
            records[row["scenario_id"]] = result
    print("VERIFIED COMPLETED CANDIDATES:", len(records), "/", len(rows), flush=True)
    pending = [r for r in rows if r["scenario_id"] not in records]
    for row in pending[:max_cases]:
        sid = row["scenario_id"]
        c.verify_scope(root, sid)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = directory / (d.sha(sid.encode())[:12]+"-"+stamp+"-candidate.log")
        print("TRAINING ADMITTED CANDIDATE:", sid, flush=True)
        print("LOG:", log_path, flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            result = runner([sys.executable, str(c4 / "scripts/validate_hybrid_fresh_candidate.py"),
                "--data-root", str(root), "--scenario-id", sid, "--execute"],
                stdout=log, stderr=subprocess.STDOUT, text=True)
        d.require(result.returncode == 0, "Candidate stage failed; preserve artifacts and inspect "+str(log_path))
        record = c.completed_case(root, row)
        d.require(record is not None, "Candidate stage did not produce verified pre-release evidence.")
        records[sid] = record
        print("CANDIDATE / RELOAD / PRE-RELEASE EVIDENCE PASS:", sid, flush=True)
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
    if len(records) < len(rows):
        print("BOUNDED CANDIDATE STAGE COMPLETE:", len(records), "/", len(rows))
        print("Repeat --execute with a larger --max-cases to resume. Completed candidates are reused.")
        print("GENERATOR REFITS: 0; FUTURE EVALUATIONS: 0; N1/N2/POLICY FITS: 0")
        return
    report = dict(purpose="completed_fresh_conformal_candidate_pre_release_batch",
        candidates=len(records), records=[records[r["scenario_id"]] for r in rows],
        assigned_groups=24, nonempty_admission_groups=claim["nonempty_admission_groups"],
        blocked_cases_retained=claim["blocked_cases"], future_evaluation_executed=False,
        complete_decision_evidence_freeze=False, N1_forecasts_frozen=False,
        production_release_authorized=False)
    f.ensure_documents(directory / "completion-v1", {"candidate_batch": report})
    print("ALL ADMITTED CANDIDATES / PRE-RELEASE EVIDENCE COMPLETE:", len(records))
    print("FUTURE EVALUATION / N2 CALIBRATION: NOT PERFORMED")
    print("REPORT:", directory / "completion-v1/candidate_batch.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--max-cases", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    d.require(args.max_cases > 0, "max-cases must be positive.")
    root, c4 = Path(raw), Path(__file__).resolve().parents[1]
    claim, rows = c.prepare(root, c4)
    print("VERIFIED ALL 24 CALIBRATION ADMISSION GROUPS")
    print("ADMITTED CASES:", claim["admitted_cases"], "BLOCKED CASES:", claim["blocked_cases"])
    print("NONEMPTY ADMISSION GROUPS:", claim["nonempty_admission_groups"], "/ 24")
    print("MAXIMUM NEW CANDIDATES THIS CALL:", min(args.max_cases, len(rows)))
    print("REPAIR GENERATION / FUTURE EVALUATION / N1 REFITS: DISABLED")
    if not args.execute:
        print("PLAN ONLY. No candidate models or output claims written.")
        return
    run_batch(root, c4, claim, rows, args.max_cases)


if __name__ == "__main__":
    main()
