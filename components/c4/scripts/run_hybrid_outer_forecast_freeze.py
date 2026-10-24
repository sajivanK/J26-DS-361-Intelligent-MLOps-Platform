"""Freeze all admitted outer evidence and four-method N1 forecasts.

This stage performs inference only. No future data/outcomes or calibration fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import _hybrid_outer_forecasts as q

d, e, f, v = q.d, q.e, q.f, q.v


def run_batch(root, c4, claim, rows, models, max_cases, runner=subprocess.run):
    root, c4 = Path(root), Path(c4)
    directory = f.bundle_at(root) / "outer-forecast-freezer-v1"
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
    f.ensure_documents(directory, {"claim": claim})
    records = {}
    for row in rows:
        record = q.completed_forecast(root, row, models, claim["model_bindings"])
        if record is not None:
            records[row["scenario_id"]] = record
    print("VERIFIED FROZEN FORECAST CASES:", len(records), "/", len(rows), flush=True)
    pending = [r for r in rows if r["scenario_id"] not in records]
    for row in pending[:max_cases]:
        sid = row["scenario_id"]
        q.c.verify_scope(root, sid)
        run = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / sid
        decision_dir = run / "pre_release_freeze-v1"
        if decision_dir.exists():
            q.verify_decision(root, row)
            print("REUSED VERIFIED DECISION FREEZE:", sid, flush=True)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            log_path = directory / (d.sha(sid.encode())[:12]+"-"+stamp+"-decision-freeze.log")
            print("FREEZING DECISION EVIDENCE:", sid, flush=True)
            print("LOG:", log_path, flush=True)
            with log_path.open("x", encoding="utf-8") as log:
                result = runner([sys.executable, str(c4 / "scripts/freeze_hybrid_outer_case.py"),
                    "--data-root", str(root), "--scenario-id", sid, "--freeze"],
                    stdout=log, stderr=subprocess.STDOUT, text=True)
            d.require(result.returncode == 0, "Decision freeze failed; preserve artifacts and inspect "+str(log_path))
        record = q.freeze_forecasts(root, row, models, claim["model_bindings"])
        d.require(record is not None, "Forecast checkpoint did not complete.")
        records[sid] = record
        print("DECISION / 75 FEATURES / FOUR-METHOD FORECASTS / SELECTED POLICIES FROZEN:", sid, flush=True)
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
    if len(records) < len(rows):
        print("BOUNDED FORECAST FREEZE COMPLETE:", len(records), "/", len(rows))
        print("Future evaluation remains disabled. Repeat --freeze with a larger --max-cases to resume.")
        return
    report = dict(purpose="all_fresh_outer_forecasts_and_policies_frozen_before_future", cases=len(rows),
        records=[records[row["scenario_id"]] for row in rows], model_bindings=claim["model_bindings"],
        assigned_groups=16, nonempty_admission_groups=claim["nonempty_admission_groups"],
        blocked_cases_retained=claim["blocked_cases_retained"], forecasts_frozen_before_future=True,
        N2_calibration_sha256=claim["N2_calibration_sha256"],
        policy_selection_sha256=claim['policy_selection_sha256'],
        selected_policy_decisions_frozen_before_future=True,
        future_evaluation_executed=False, N2_calibrated=True, production_release_authorized=False)
    f.ensure_documents(directory / "completion-v1", {"forecast_batch": report})
    print("ALL OUTER EVIDENCE / FOUR-METHOD FORECASTS / SELECTED POLICIES FROZEN:", len(rows))
    print("OUTER FUTURE TARGETS READ: 0; MODEL FITS: 0; N2/POLICY FITS: 0")
    print("REPORT:", directory / "completion-v1/forecast_batch.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--max-cases", type=int, default=1)
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    d.require(args.max_cases > 0, "max-cases must be positive.")
    root, c4 = Path(raw), Path(__file__).resolve().parents[1]
    claim, rows, models = q.prepare(root, c4)
    print("VERIFIED COMPLETED CANDIDATES:", len(rows))
    print("FORECAST METHODS:", ", ".join(models))
    print("FEATURES: 75 decision-time features; future outcomes disabled")
    print("NO GENERATOR, CANDIDATE OR N1 REFITTING")
    if not args.freeze:
        print("PLAN ONLY. No evidence or forecast artifacts written.")
        return
    run_batch(root, c4, claim, rows, models, args.max_cases)


if __name__ == "__main__":
    main()
