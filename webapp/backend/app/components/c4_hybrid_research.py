"""Read the fixed hybrid research summary without loading or fitting models."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path

VERSION = "adult-hybrid-fresh-validation-v1"
EXPECTED_PROTOCOL_SHA256 = "c6241cfe55680a767ecfd381294be0852bf4fced6b8c6cf2da4f4fa880dcf837"
METHODS = {
    "hybrid_ridge_lightgbm_lightgbm": "Hybrid · Ridge / LightGBM / LightGBM",
    "same_external_training_lightgbm": "LightGBM",
    "same_external_training_linear": "Linear quantile",
    "same_external_training_constant_quantiles": "Constant quantiles",
}
TARGETS = {
    "future_global_margin": "Global performance",
    "future_slice_margin": "True injected slice",
    "future_nonaffected_margin": "True slice complement",
}
PRIMARY = next(iter(METHODS))
DECISIONS = ("AUTO_DEPLOY", "HUMAN_REVIEW", "ESCALATE")


class ResearchUnavailable(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, digest):
    require(isinstance(digest, str) and len(digest) == 64, "Invalid checksum.")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "Research artifact checksum differs.")
    return data


def document(directory, stem):
    expected = (directory / (stem + ".sha256")).read_text(encoding="utf-8").strip()
    payload = json.loads(checked(directory / (stem + ".json"), expected))
    require(isinstance(payload, dict), "Research document must be an object.")
    return payload, expected


def number(value, low=None, high=None):
    require(not isinstance(value, bool), "Boolean is not a measurement.")
    value = float(value)
    require(math.isfinite(value), "Nonfinite research measurement.")
    require(low is None or value >= low, "Research measurement below valid range.")
    require(high is None or value <= high, "Research measurement above valid range.")
    return value


def read_results(data_root=None):
    """Return a checksum-verified saved comparison, separate from case inference."""
    raw = data_root if data_root is not None else os.environ.get("C4_DATA_ROOT")
    if not raw:
        raise ResearchUnavailable("Revised N1 research artifacts are unavailable. Configure C4_DATA_ROOT in the C4 worker.")
    try:
        return _read(Path(raw))
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        raise ResearchUnavailable("Revised N1 research artifacts are missing or could not be verified. Original recorded-case evaluation is separate.") from exc


def _read(root):
    bundle = root / "c4_artifacts/forecasting_experiments/adult" / VERSION
    protocol, protocol_hash = document(bundle, "validation_protocol")
    require(protocol_hash == EXPECTED_PROTOCOL_SHA256 and protocol["primary_method"] == PRIMARY,
            "Unexpected fixed validation protocol.")
    output = bundle / "outer-future-evaluator-v1/completion-v1"
    report, report_hash = document(output, "future_batch")
    claim, claim_hash = document(output.parent, "claim")
    require(report["purpose"] == "fresh_outer_future_evaluation_complete"
            and claim["purpose"] == "fresh_outer_external_future_evaluation_v1"
            and report["evaluator_claim_sha256"] == claim_hash
            and report["cases"] == 56 and report["assigned_groups"] == 16
            and report["nonempty_groups"] == 11 and report["blocked_cases_retained"] == 536
            and report["production_release_authorized"] is False
            and claim["production_release_authorized"] is False
            and report["training_executed"] is False and report["policy_refitted"] is False
            and report["N2_calibrated"] is True and report["policy_selection_frozen"] is True
            and claim["future_evaluation_authorized"] is True
            and report["independent_dataset_validation"] is False
            and report["held_out_scenario_group_evaluation"] is True
            and report["selected_policy_decisions_frozen_before_future"] is True
            and claim["selected_policy_decisions_frozen_before_future"] is True
            and report["forecasts_frozen_before_future"] is True
            and claim["expected_cohort_rows"] == 12899
            and set(claim["model_bindings"]) == set(METHODS), "Unexpected final research scope.")
    for flag in ("candidate_training_authorized", "generator_refitting_authorized", "N1_refitting_authorized", "N2_fitting_authorized", "policy_fitting_authorized"):
        require(claim[flag] is False, "Final evaluator authorized a fitting stage.")
    contents = {}
    for name, digest in report["files"].items():
        require(Path(name).name == name and name not in (".", ".."), "Invalid research filename.")
        contents[name] = checked(output / name, digest)
    required = {"forecast_metrics.csv", "N2_outer_diagnostics.json", "policy_diagnostics.json", "case_index.csv", "selected_policy_decisions.csv"}
    require(required <= set(contents), "Incomplete final comparison outputs.")
    rows = list(csv.DictReader(io.StringIO(contents["case_index.csv"].decode("utf-8"))))
    ids = [row["scenario_id"] for row in rows]
    require(len(ids) == len(set(ids)) == 56 and ids == claim["scenario_ids"]
            and ids == [row["scenario_id"] for row in report["records"]]
            and all(row["split_role"] == "outer_test" for row in rows)
            and len({row["group_id"] for row in rows}) == 11, "Final case inventory differs.")
    for record, row in zip(report["records"], rows):
        require(record["group_id"] == row["group_id"] and record["split_role"] == "outer_test", "Case role/group differs.")
    n2, n2_hash = document(bundle / "n2-calibration-v1/completion-v1", "calibration")
    selected, selected_hash = document(bundle / "policy-selection-v1/completion-v1", "selection")
    require(n2_hash == report["N2_calibration_sha256"] == claim["N2_calibration_sha256"] == selected["N2_calibration_sha256"]
            and selected_hash == report["policy_selection_sha256"] == claim["policy_selection_sha256"]
            and n2["model_bindings"] == selected["model_bindings"] == claim["model_bindings"]
            and selected["production_release_authorized"] is False, "N2/selected-policy lineage differs.")
    coverage = json.loads(contents["N2_outer_diagnostics.json"])
    governance = json.loads(contents["policy_diagnostics.json"])
    require(coverage["N2_calibration_sha256"] == n2_hash and coverage["N2_refitted"] is False
            and governance["policy_selection_sha256"] == selected_hash and governance["policy_refitted"] is False
            and governance["formal_conditional_risk_guarantee_established"] is False
            and governance["production_release_authorized"] is False
            and governance["admitted_cases"] == 56 and governance["blocked_cases_retained"] == 536
            and governance["assigned_groups"] == 16 and governance["nonempty_groups"] == 11
            and governance["all_blocked_groups"] == 5
            and set(coverage["methods"]) == set(governance["methods"]) == set(METHODS), "Research statistics scope differs.")
    measured = list(csv.DictReader(io.StringIO(contents["forecast_metrics.csv"].decode("utf-8"))))
    by_key = {(row["method"], row["target"]): row for row in measured}
    require(len(measured) == len(by_key) == 12
            and set(by_key) == {(m,t) for m in METHODS for t in TARGETS}, "Forecast metric inventory differs.")
    signals = list(csv.DictReader(io.StringIO(contents["selected_policy_decisions.csv"].decode("utf-8"))))
    require(len(signals) == 56 * 4 * 2, "Incomplete saved policy signals.")
    methods = []
    membership = {row["scenario_id"]: row["group_id"] for row in rows}
    for method, label in METHODS.items():
        cov = coverage["methods"][method]
        require(set(cov["targets"]) == set(TARGETS), "Coverage population inventory differs.")
        metrics = []
        for target, target_label in TARGETS.items():
            row, band = by_key[(method,target)], cov["targets"][target]
            mae, rmse, pinball = (number(row[key], 0) for key in ("MAE_pp", "RMSE_pp", "Pinball_pp"))
            for displayed, raw_name in ((mae,"mae"),(rmse,"rmse"),(pinball,"mean_pinball")):
                require(math.isclose(displayed, 100 * number(row[raw_name], 0), rel_tol=1e-10, abs_tol=1e-10), "Metric units differ.")
            r2 = None if row["r2"] == "" else number(row["r2"], high=1.0000000001)
            metrics.append(dict(target=target,label=target_label,mae_pp=mae,rmse_pp=rmse,r2=r2,
                mean_pinball_pp=pinball,width_pp=100*number(band["equal_group_weighted_mean_width_fraction"],0,2),
                interval_score_pp=100*number(band["equal_group_weighted_interval_score_alpha_0_05"],0)))
        policies = {}
        for family in ("global", "adaptive"):
            entry = governance["methods"][method][family]
            fixed = selected["methods"][method][family]
            stats = entry["statistics"]
            require(entry["selected_policy"] == fixed["selected_policy"]
                    and fixed["status"] == "UNSUPPORTED_SELECTION_REVIEW_ONLY"
                    and fixed["selected_policy"]["kind"] == "review_only", "Unexpected selected-policy state.")
            subset = [r for r in signals if r["method"] == method and r["family"] == family]
            require(len(subset) == 56 and {r["scenario_id"] for r in subset} == set(ids)
                    and all(r["split_role"] == "outer_test" and membership[r["scenario_id"]] == r["group_id"]
                            and r["decision"] in DECISIONS for r in subset), "Policy signal inventory differs.")
            counts = Counter(r["decision"] for r in subset)
            decision_counts = {key:counts[key] for key in DECISIONS}
            require(stats["decision_counts"] == decision_counts and decision_counts["AUTO_DEPLOY"] == 0
                    and stats["equal_group_conditional_unsafe_rate"] is None
                    and number(stats["equal_group_automation"],0,1) == 0,
                    "Review-only decision/risk statistics differ.")
            policies[family] = dict(kind="review_only",decision_counts=decision_counts,automation=0.0,conditional_unsafe_rate=None)
        methods.append(dict(id=method,label=label,is_primary=method==PRIMARY,metrics=metrics,policies=policies,
            coverage=dict(scenario_joint=number(cov["scenario_joint_coverage"],0,1),
                          whole_group_joint=number(cov["complete_group_joint_coverage"],0,1))))
    return dict(version=VERSION,scope="held_out_scenario_groups_shared_external_development_cohort",
        evidence_verified=True,inference_executed=False,production_release_authorized=False,
        independent_dataset_validation=False,nominal_coverage=0.95,primary_method=PRIMARY,
        counts=dict(assigned_groups=16,nonempty_groups=11,all_blocked_groups=5,planned_cases=592,
                    admitted_cases=56,blocked_cases=536,cohort_rows=12899),methods=methods,
        unsupported_policy_contexts=selected["context_support"]["unsupported_contexts"],
        provenance=dict(final_report_sha256=report_hash,evaluator_claim_sha256=claim_hash,
            validation_protocol_sha256=protocol_hash,N2_calibration_sha256=n2_hash,policy_selection_sha256=selected_hash))
