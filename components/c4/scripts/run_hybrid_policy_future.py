"""Evaluate the frozen fresh-policy forecasts; inference only, resumable."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import _hybrid_policy_future as u

d, e, f = u.d, u.e, u.f


def run_batch(root, claim, rows, shared, feature_types, max_cases):
    d.require(type(max_cases) is int and max_cases > 0, "max-cases must be positive.")
    directory = f.bundle_at(root) / "policy-future-evaluator-v1"
    u.verify_inputs(claim["inputs"])
    # Persist authorization and all source hashes before the first outcome parse.
    f.ensure_documents(directory, {"claim": claim})
    _, claim_hash = d.checked_json(directory, "claim")
    clean = u.load_cohort(claim)
    records = {}
    for row in rows:
        record = u.completed_case(root, row, claim, claim_hash, clean)
        if record is not None:
            records[row["scenario_id"]] = record
    print("VERIFIED COMPLETED FUTURE CASES:", len(records), "/", len(rows), flush=True)
    pending = [r for r in rows if r["scenario_id"] not in records]
    champion = None
    if pending:
        champion, meta = u.load_champion(Path(claim["champion_directory"]))
        d.require(meta["model_sha256"] == claim["champion_model_sha256"], "Loaded champion differs.")
    for row in pending[:max_cases]:
        print("EVALUATING FROZEN CANDIDATE:", row["scenario_id"], flush=True)
        record = u.evaluate_case(root, row, claim, claim_hash, clean, champion, shared, feature_types)
        records[row["scenario_id"]] = record
        print("FUTURE PAIRED TARGETS VERIFIED:", len(records), "/", len(rows), record["primary_targets"], flush=True)
    u.verify_inputs(claim["inputs"])
    if len(records) < len(rows):
        print("BOUNDED FUTURE EVALUATION COMPLETE:", len(records), "/", len(rows))
        print("Repeat --execute with a larger --max-cases. Completed outcomes are verified and reused.")
        return
    ordered = [records[r["scenario_id"]] for r in rows]
    index, targets, metrics = u.metric_tables(root, rows, ordered, claim)
    intervals, coverage = u.interval_diagnostics(root, rows, targets, index, claim)
    report = dict(purpose="fresh_policy_future_evaluation_complete_not_final_outer_results",
        evaluator_claim_sha256=claim_hash, cases=len(rows), records=ordered,
        assigned_groups=claim["assigned_groups"], nonempty_groups=int(index.group_id.nunique()),
        blocked_cases_retained=claim["blocked_cases_retained"],
        cohort_csv_sha256=claim["cohort_csv_sha256"], forecasts_frozen_before_future=True,
        N2_calibration_sha256=claim["N2_calibration_sha256"],
        training_executed=False, N2_calibrated=True, policy_calibrated=False,
        production_release_authorized=False, limitation=claim["limitation"])
    files = {"case_index.csv": index.to_csv(index=False, lineterminator="\n").encode(),
        "targets.csv": targets.to_csv(index_label="scenario_id", lineterminator="\n").encode(),
        "forecast_metrics.csv": metrics.to_csv(index=False, lineterminator="\n").encode()}
    files["N2_policy_diagnostics.json"] = e.encoded(dict(methods=coverage,
        N2_calibration_sha256=claim["N2_calibration_sha256"], N2_refitted=False,
        evaluation="policy_groups_not_final_outer_test", independent_coverage_validated=False))
    for method, frame in intervals.items():
        files[method+"_fixed_N2_intervals.csv"] = frame.to_csv(index_label="scenario_id", lineterminator="\n").encode()
    report["files"] = {name: d.sha(data) for name, data in files.items()}
    output = directory / "completion-v1"
    if output.exists():
        saved, _ = d.checked_json(output, "future_batch")
        d.require(saved == report, "Existing completion differs; preserve it.")
        for name, checksum in saved["files"].items():
            d.checked(output / name, checksum)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            (output / name).write_bytes(data)
        data = e.encoded(report)
        (output / "future_batch.json").write_bytes(data)
        (output / "future_batch.sha256").write_text(d.sha(data)+"\n", encoding="utf-8")
    print(metrics[["method", "target", "MAE_pp", "RMSE_pp", "r2", "Pinball_pp"]].to_string(index=False))
    for method, result in coverage.items():
        print("FIXED N2 POLICY COVERAGE:", method, "SCENARIO JOINT:", result["scenario_joint_coverage"],
              "WHOLE-GROUP:", result["complete_group_joint_coverage"])
    print("ALL FRESH POLICY FUTURE CASES COMPLETE:", len(rows))
    print("MODEL FITS: 0; N2/POLICY FITS: 0; OUTER CASES ACCESSED: 0")
    print("Policy-set metrics; keep N1/N2 fixed. Policy fitting and final outer evaluation remain pending.")
    print("RESULTS:", output / "forecast_metrics.csv")
    print("REPORT:", output / "future_batch.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--max-cases", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    d.require(args.max_cases > 0, "max-cases must be positive.")
    claim, rows, shared, types = u.prepare(Path(raw))
    print("VERIFIED ENTIRE PRE-FUTURE FORECAST BATCH:", len(rows), flush=True)
    print("EXTERNAL ROWS:", claim["expected_cohort_rows"], "; FOUR FIXED FORECASTERS; NO MODEL FITTING", flush=True)
    if not args.execute:
        print("PLAN ONLY. External observations not parsed; no outcomes evaluated or files written.")
        return
    run_batch(Path(raw), claim, rows, shared, types, args.max_cases)


if __name__ == "__main__":
    main()
