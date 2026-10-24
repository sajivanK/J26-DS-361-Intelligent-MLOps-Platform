"""Freeze four-method N2 calibration from completed fresh calibration outcomes.

No forecast/model tuning, candidate prediction, repair generation or outer reads.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import _hybrid_fresh_future as u
from c4.conformal.benchmark_n2 import (JointCalibration, fit_joint_calibration,
    apply_joint_calibration, calibration_diagnostics)

d, e, f, v = u.d, u.e, u.f, u.v
C4 = Path(__file__).resolve().parents[1]
EXPECTED_RULES = dict(alpha=.05, raw_quantiles=[.05, .5, .95],
    joint_score="max_over_three_targets_of_max_0_lower_minus_y_y_minus_upper",
    calibration_unit="replication_group",
    group_score="max_joint_score_across_admitted_variants_within_group",
    quantile_rank="ceil((nonempty_calibration_groups_plus_1)*0.95)",
    empty_groups="reported_excluded_without_adaptive_replacement",
    too_few_nonempty_groups="uninformative_margin_bounds_and_disable_AUTO_DEPLOY_no_informative_95_percent_claim")


def checked_table(directory, filename, report):
    content = d.checked(Path(directory) / filename, report["files"][filename])
    return pd.read_csv(io.BytesIO(content), float_precision="round_trip")


def prepare(root, c4=C4):
    root, c4 = Path(root), Path(c4)
    evaluator_claim, rows, _, _ = u.prepare(root, c4)
    evaluator_dir = f.bundle_at(root) / "future-evaluator-v1"
    saved, evaluator_hash = d.checked_json(evaluator_dir, "claim")
    d.require(saved == evaluator_claim, "Future evaluator code/input binding differs.")
    output = evaluator_dir / "completion-v1"
    report, report_hash = d.checked_json(output, "future_batch")
    d.require(report["purpose"] == "fresh_conformal_future_evaluation_complete_not_final_outer_results"
              and report["evaluator_claim_sha256"] == evaluator_hash
              and report["cases"] == len(rows) and report["forecasts_frozen_before_future"] is True
              and report["training_executed"] is False and report["N2_calibrated"] is False
              and report["policy_calibrated"] is False and report["production_release_authorized"] is False
              and report["cohort_csv_sha256"] == evaluator_claim["cohort_csv_sha256"],
              "A complete, correctly bound calibration future batch is required.")
    pins = dict(evaluator_claim["inputs"])
    pins[str(evaluator_dir / "claim.json")] = evaluator_hash
    pins[str(output / "future_batch.json")] = report_hash
    for name, checksum in report["files"].items():
        d.checked(output / name, checksum)
        pins[str(output / name)] = checksum
    clean = u.load_cohort(evaluator_claim)
    records = [u.completed_case(root, row, evaluator_claim, evaluator_hash, clean) for row in rows]
    d.require(all(record is not None for record in records) and records == report["records"],
              "Completed calibration targets differ from verified paired outcomes.")
    index = checked_table(output, "case_index.csv", report)
    targets = checked_table(output, "targets.csv", report).set_index("scenario_id")
    expected_index = pd.DataFrame([{k: row[k] for k in ("scenario_id", "group_id", "split_role")} for row in rows])
    expected_targets = pd.DataFrame([dict(scenario_id=r["scenario_id"], **r["primary_targets"]) for r in records]).set_index("scenario_id")
    pd.testing.assert_frame_equal(index, expected_index, check_exact=True, check_dtype=False)
    pd.testing.assert_frame_equal(targets, expected_targets, check_exact=True, check_dtype=False)
    group_claim, _ = d.checked_json(f.bundle_at(root) / "candidate-executor-v1", "claim")
    group_summary = pd.DataFrame(group_claim["group_counts"])
    verify_inventory(index, targets, group_summary)
    d.require(report["assigned_groups"] == len(group_summary)
              and report["nonempty_groups"] == index.group_id.nunique()
              and report["blocked_cases_retained"] == int(group_summary.blocked.sum()),
              "Calibration completion admission/group totals differ.")
    forecasts = {name: [] for name in evaluator_claim["model_bindings"]}
    for row, record in zip(rows, records):
        run = u.run_at(root, row)
        outcome, digest = d.checked_json(run / "external_future-v1", "outcomes")
        pins[str(run / "external_future-v1/outcomes.json")] = digest
        for name, checksum in outcome["files"].items():
            pins[str(run / "external_future-v1" / name)] = checksum
        source = u.source_record(evaluator_claim, row)
        predictions = json.loads(d.checked(run / "forecast_freeze-v1/forecasts.json", source["forecasts_sha256"]))
        d.require(set(predictions) == set(forecasts), "Four-method forecast inventory differs.")
        for method in forecasts:
            forecasts[method].append(predictions[method]["processed"])
    columns = [target+"__"+label for target in d.TARGETS for _, label in d.QUANTILES]
    forecasts = {method: pd.DataFrame(values, index=targets.index).loc[:, columns] for method, values in forecasts.items()}
    protocol_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    frozen, _ = d.checked_json(protocol_dir, "freeze")
    protocol = json.loads(d.checked(protocol_dir / "protocol.json", frozen["protocol_sha256"]))
    definition = protocol["N2"]
    d.require(all(definition.get(key) == value for key, value in EXPECTED_RULES.items()),
              "N2 rules differ from the frozen alpha/rank/scoring/fallback definition.")
    for relative in ("scripts/calibrate_hybrid_fresh_n2.py", "src/c4/conformal/benchmark_n2.py"):
        path = c4 / relative
        checksum = d.sha(path.read_bytes())
        if str(path) in pins:
            d.require(pins[str(path)] == checksum, "Previously frozen N2 implementation changed.")
        pins[str(path)] = checksum
    claim = dict(purpose="fresh_four_method_N2_group_max_calibration_v1", inputs=pins,
        environment=u.q.c.environment(), future_batch_sha256=report_hash, evaluator_claim_sha256=evaluator_hash,
        model_bindings=evaluator_claim["model_bindings"], definition=definition,
        scenario_ids=index.scenario_id.tolist(), assigned_groups=len(group_summary),
        nonempty_groups=int(index.group_id.nunique()), all_blocked_groups=int((group_summary.admitted == 0).sum()),
        cases=len(index), blocked_cases_retained=int(group_summary.blocked.sum()),
        N1_refitting_authorized=False, candidate_training_authorized=False,
        future_reevaluation_authorized=False, policy_fitting_authorized=False,
        outer_evaluation_authorized=False, production_release_authorized=False,
        limitation=evaluator_claim["limitation"],
        support_note="auto_deploy_supported in the core calibration state means numerical order-statistic support only. It is neither a policy decision nor release authorization.")
    u.verify_inputs(pins)
    return claim, index, targets, forecasts, group_summary


def verify_inventory(index, targets, groups):
    d.require(list(index) == ["scenario_id", "group_id", "split_role"]
              and index.scenario_id.is_unique and index.split_role.eq("conformal_calibration").all()
              and index.scenario_id.tolist() == targets.index.tolist()
              and list(targets) == list(d.TARGETS), "N2 input roles/identities/schema differ.")
    d.require(list(groups) == ["group_id", "admitted", "blocked"]
              or set(groups) == {"group_id", "admitted", "blocked"}, "Invalid assigned group summary.")
    d.require(len(groups) == 24 and groups.group_id.is_unique
              and all(type(value) in (int, np.int64, np.int32) and value >= 0
                      for value in [*groups.admitted, *groups.blocked])
              and (groups.admitted+groups.blocked).eq(37).all(), "Assigned admission inventory differs.")
    counts = index.group_id.value_counts()
    expected = dict(zip(groups.group_id, groups.admitted))
    d.require(set(counts.index).issubset(expected)
              and all(int(counts.get(gid, 0)) == int(count) for gid, count in expected.items())
              and int(groups.admitted.sum()) == len(index), "Admitted group counts differ from calibration rows.")


def fit_methods(index, targets, forecasts):
    d.require(set(forecasts) == {u.q.PRIMARY, *u.q.b.METHODS}, "Exactly four frozen methods are required.")
    result = {}
    for method, predicted in forecasts.items():
        d.require(list(predicted) == [t+"__"+label for t in d.TARGETS for _, label in d.QUANTILES],
                  "Unexpected forecast column schema.")
        calibration, scores, grouped = fit_joint_calibration(targets, predicted, index)
        intervals = apply_joint_calibration(calibration, predicted)
        reloaded = JointCalibration.from_dict(json.loads(e.encoded(calibration.to_dict())))
        pd.testing.assert_frame_equal(intervals, apply_joint_calibration(reloaded, predicted), check_exact=True)
        result[method] = dict(calibration=calibration.to_dict(), scores=scores, group_scores=grouped,
            intervals=intervals, diagnostics=calibration_diagnostics(targets, intervals, index))
    return result


def save_results(root, claim, index, targets, forecasts, groups, results):
    directory = f.bundle_at(root) / "n2-calibration-v1"
    u.verify_inputs(claim["inputs"])
    # Claim is persisted before N2 output files; incomplete outputs are preserved.
    f.ensure_documents(directory, {"claim": claim})
    _, claim_hash = d.checked_json(directory, "claim")
    files = {"case_index.csv": index.to_csv(index=False, lineterminator="\n").encode(),
        "targets.csv": targets.to_csv(index_label="scenario_id", lineterminator="\n").encode(),
        "group_summary.csv": groups.to_csv(index=False, lineterminator="\n").encode()}
    methods, diagnostics = {}, {}
    for method, result in results.items():
        methods[method], diagnostics[method] = result["calibration"], result["diagnostics"]
        for name, frame in (("forecasts", forecasts[method]), ("scores", result["scores"]),
                            ("group_scores", result["group_scores"]), ("intervals", result["intervals"])):
            files[method+"_"+name+".csv"] = frame.to_csv(index=True, lineterminator="\n").encode()
    calibration = dict(purpose="fresh_fixed_four_method_N2_calibration", methods=methods,
        definition=claim["definition"], model_bindings=claim["model_bindings"], claim_sha256=claim_hash,
        assigned_groups=claim["assigned_groups"], nonempty_groups=claim["nonempty_groups"],
        all_blocked_groups=claim["all_blocked_groups"], independent_coverage_validated=False,
        policy_calibrated=False, production_release_authorized=False, support_note=claim["support_note"],
        limitation=claim["limitation"])
    files["calibration.json"] = e.encoded(calibration)
    files["calibration.sha256"] = (d.sha(files["calibration.json"])+"\n").encode()
    files["calibration_diagnostics.json"] = e.encoded(dict(methods=diagnostics,
        evaluation="in_sample_calibration_only_not_final_test", independent_coverage_validated=False))
    summary = dict(purpose="verified_fresh_N2_calibration", claim_sha256=claim_hash,
        files={name: d.sha(data) for name, data in files.items()}, cases=len(index),
        assigned_groups=claim["assigned_groups"], nonempty_groups=claim["nonempty_groups"],
        all_blocked_groups=claim["all_blocked_groups"], blocked_cases_retained=claim["blocked_cases_retained"],
        N1_refitted=False, candidates_retrained=False, future_reevaluated=False,
        outer_test_evaluated=False, independent_coverage_validated=False, production_release_authorized=False)
    completion = directory / "completion-v1"
    if completion.exists():
        saved, _ = d.checked_json(completion)
        d.require(saved == summary, "Existing N2 completion differs; do not overwrite.")
        for name, checksum in saved["files"].items():
            d.checked(completion / name, checksum)
    else:
        completion.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            (completion / name).write_bytes(data)
        data = e.encoded(summary)
        (completion / "metadata.json").write_bytes(data)
        (completion / "metadata.sha256").write_text(d.sha(data)+"\n", encoding="utf-8")
    u.verify_inputs(claim["inputs"])
    return completion


def display(results):
    for method, result in results.items():
        state = result["calibration"]
        adjustment = None if state["adjustment"] is None else state["adjustment"]*100
        print("METHOD:", method)
        print("NONEMPTY GROUPS:", state["nonempty_groups"], "RANK:", state["rank"],
              "FALLBACK:", state["fallback"], "ADJUSTMENT (pp):", adjustment)
        print("IN-SAMPLE JOINT COVERAGE:", result["diagnostics"]["scenario_joint_coverage"],
              "WHOLE-GROUP:", result["diagnostics"]["complete_group_joint_coverage"])
    print("In-sample calibration coverage is not independent validation or a release guarantee.")
    print("N1/CANDIDATE FITS: 0; FUTURE REEVALUATIONS: 0; POLICY/OUTER ACCESSED: 0")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    root = Path(raw)
    claim, index, targets, forecasts, groups = prepare(root)
    print("VERIFIED CALIBRATION CASES:", len(index), "ASSIGNED GROUPS:", len(groups),
          "NONEMPTY:", index.group_id.nunique(), flush=True)
    results = fit_methods(index, targets, forecasts)
    display(results)
    if not args.save:
        print("PLAN ONLY. No files written. Use --save to freeze all four calibrations.")
        return
    output = save_results(root, claim, index, targets, forecasts, groups, results)
    print("FOUR N2 CALIBRATIONS SAVED / JSON RELOAD VERIFIED")
    print("CALIBRATION:", output / "calibration.json")
    print("METADATA:", output / "metadata.json")
    print("PRODUCTION RELEASE AUTHORIZED: False")


if __name__ == "__main__":
    main()
