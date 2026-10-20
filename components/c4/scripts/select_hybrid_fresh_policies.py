"""Select/freeze four-method release policies under the predeclared rules.

Signals are replayed after outcomes from hash-bound pre-future decision evidence
using the unchanged predeclared functions. Only policy rows select policies.
No models, calibrations, future classifiers or final-test cases are executed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import pandas as pd

import _hybrid_policy_future as u
from c4.risk.benchmark_n3 import BOUNDARIES, assess_safety
from c4.risk.benchmark_policy import (BUFFER_GRID, CONTEXTS, GATES, candidate_policies,
    evidence_context, policy_decision)
from c4.risk.benchmark_policy_selection import select_policy, policy_statistics

d, e, f = u.d, u.e, u.f
C4 = Path(__file__).resolve().parents[1]


def verify_definition(definition):
    d.require(definition["safety_boundaries"] == BOUNDARIES
              and definition["additional_buffer_grid_fraction"] == list(BUFFER_GRID)
              and definition["assigned_groups"] == 12 and definition["planned_cases"] == 444
              and definition["calibration_fold"] == "policy_calibration"
              and definition["change_rules_after_policy_outcomes"] is False
              and definition["production_release_authorized"] is False,
              "Frozen policy boundaries/grid/scope differ.")
    d.require(definition["selection"]["role"] == "policy_calibration_only"
              and definition["selection"]["minimum_auto_deploy_groups"] == 3
              and definition["selection"]["empirical_conditional_unsafe_ceiling"] == .05,
              "Frozen policy support/risk rule differs.")
    policies = candidate_policies()
    d.require(len(policies["global"]) == 4 and len(policies["adaptive"]) == 20,
              "Expected four global and twenty monotone adaptive policies.")
    return policies


def context_support(contexts):
    d.require(len({r["scenario_id"] for r in contexts}) == len(contexts) > 0
              and all(r["split_role"] == "policy_calibration" and r["context"] in CONTEXTS for r in contexts),
              "Context support requires unique admitted policy rows only.")
    counts = {name: len({r["group_id"] for r in contexts if r["context"] == name}) for name in CONTEXTS}
    return dict(distinct_admitted_groups_by_context=counts, unsupported_contexts=[name for name in CONTEXTS if counts[name] < 3],
        minimum_distinct_groups=3, context_features_frozen_before_future=True,
        registry_saved_before_future=False,
        derivation="Computed now solely from checksum-bound pre-future features and the predeclared context formula; no future targets used.")


def explicit_gates(snapshot, upstream):
    c2, c3 = upstream["c2_diagnosis"], upstream["c3_repair_package"]
    return dict(repair_approved=c3["repair_package_status"] == "APPROVED",
        privacy_check=c3["privacy_status"] == "PASS", diagnosis_stability=c2["diagnosis_stability"] >= .5,
        schema_compatible=snapshot["runtime_checks"].get("schema_compatible") is True,
        rollback_ready=snapshot["runtime_checks"].get("rollback_ready") is True)


def frozen_policy_rules(root, c4, pins, model_bindings):
    methods = [u.q.PRIMARY, *u.q.b.METHODS]
    original_claim_path = f.bundle_at(root) / "admission-executor-v1/claim.json"
    d.require(str(original_claim_path) in pins, "Original validation protocol provenance is missing.")
    original_execution = json.loads(d.checked(original_claim_path, pins[str(original_claim_path)]))
    validation, validation_hash = d.checked_json(f.bundle_at(root), "validation_protocol")
    d.require(validation_hash == original_execution["validation_protocol_sha256"], "Pre-outcome validation protocol differs.")
    pins[str(f.bundle_at(root) / "validation_protocol.json")] = validation_hash
    definition = validation["policy_rules"]
    config_path = c4 / "configs/policy_benchmark_v1.json"
    expected_config_hash = validation["inputs"].get(str(config_path))
    d.require(bool(expected_config_hash), "Policy configuration was not bound before outcomes.")
    d.require(json.loads(d.checked(config_path, expected_config_hash)) == definition, "Frozen policy specification changed.")
    pins[str(config_path)] = expected_config_hash
    policies = verify_definition(definition)
    d.require(validation["model_methods_override_policy_source"] == methods
              and set(model_bindings) == set(methods), "Four-method protocol override differs.")
    for relative in ("src/c4/risk/benchmark_policy.py", "src/c4/risk/benchmark_n3.py", "src/c4/risk/benchmark_policy_selection.py"):
        path = c4 / relative
        digest = validation["inputs"].get(str(path))
        d.require(bool(digest), "Operational policy implementation was not frozen before outcomes.")
        d.checked(path, digest)
        pins[str(path)] = digest
    return definition, policies


def prepare(root, c4=C4):
    root, c4 = Path(root), Path(c4)
    evaluator, rows, _, _ = u.prepare(root, c4)
    directory = f.bundle_at(root) / "policy-future-evaluator-v1"
    saved, evaluator_hash = d.checked_json(directory, "claim")
    d.require(saved == evaluator, "Policy future evaluator inputs/implementation changed.")
    output = directory / "completion-v1"
    report, report_hash = d.checked_json(output, "future_batch")
    d.require(report["purpose"] == "fresh_policy_future_evaluation_complete_not_final_outer_results"
              and report["evaluator_claim_sha256"] == evaluator_hash
              and report["cases"] == len(rows) and report["assigned_groups"] == 12
              and report["N2_calibrated"] is True and report["policy_calibrated"] is False
              and report["training_executed"] is False and report["production_release_authorized"] is False
              and report["N2_calibration_sha256"] == evaluator["N2_calibration_sha256"],
              "A complete, bound policy future batch is required.")
    pins = dict(evaluator["inputs"])
    pins[str(directory / "claim.json")] = evaluator_hash
    pins[str(output / "future_batch.json")] = report_hash
    for name, digest in report["files"].items():
        d.checked(output / name, digest)
        pins[str(output / name)] = digest
    clean = u.load_cohort(evaluator)
    records = [u.completed_case(root, row, evaluator, evaluator_hash, clean) for row in rows]
    d.require(all(r is not None for r in records) and records == report["records"], "Policy outcomes do not reproduce from paired prediction artifacts.")
    index = pd.read_csv(io.BytesIO(d.checked(output / "case_index.csv", report["files"]["case_index.csv"])), float_precision="round_trip")
    targets = pd.read_csv(io.BytesIO(d.checked(output / "targets.csv", report["files"]["targets.csv"])), float_precision="round_trip").set_index("scenario_id")
    expected_index = pd.DataFrame([{key: row[key] for key in ("scenario_id", "group_id", "split_role")} for row in rows])
    expected_targets = pd.DataFrame([dict(scenario_id=r["scenario_id"], **r["primary_targets"]) for r in records]).set_index("scenario_id")
    pd.testing.assert_frame_equal(index, expected_index, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(targets, expected_targets, check_dtype=False, check_exact=True)
    d.require(index.split_role.eq("policy_calibration").all()
              and report["nonempty_groups"] == int(index.group_id.nunique()), "Policy role/group inventory differs.")
    methods = [u.q.PRIMARY, *u.q.b.METHODS]
    definition, policies = frozen_policy_rules(root, c4, pins, evaluator["model_bindings"])
    n2_claim, calibration, n2_pins = f.verify_n2(root)
    d.require(calibration["model_bindings"] == evaluator["model_bindings"], "N2/model bindings differ.")
    pins.update(n2_pins)
    contexts, decision_inputs = [], []
    for row, record in zip(rows, records):
        sid = row["scenario_id"]
        source = u.source_record(evaluator, row)
        run = u.run_at(root, row)
        feature_payload = json.loads(d.checked(run / "forecast_freeze-v1/features.json", source["feature_sha256"]))
        x = u.q.make_feature_frame(dict(zip(feature_payload["columns"], feature_payload["values"])), feature_payload["columns"], sid)
        values = x.iloc[0].to_dict()
        snapshot, _ = d.checked_json(run / "pre_release_freeze-v1", "freeze")
        envelope, envelope_hash = d.checked_json(run / "pre_release_freeze-v1", "upstream_simulation")
        d.require(envelope_hash == snapshot["upstream_simulation_sha256"], "Policy upstream envelope link differs.")
        forecasts = json.loads(d.checked(run / "forecast_freeze-v1/forecasts.json", source["forecasts_sha256"]))
        context = evidence_context(values)
        contexts.append(dict(scenario_id=sid, group_id=row["group_id"], split_role=row["split_role"], **context))
        decision_inputs.append(dict(scenario_id=sid, group_id=row["group_id"], split_role=row["split_role"],
            features=values, gates=explicit_gates(snapshot, envelope["upstream"]),
            intervals={method: {t: [forecasts[method]["calibrated"][t+"__lower"], forecasts[method]["calibrated"][t+"__upper"]]
                                  for t in d.TARGETS} for method in methods}))
        outcome, digest = d.checked_json(run / "external_future-v1", "outcomes")
        pins[str(run / "external_future-v1/outcomes.json")] = digest
        for name, checksum in outcome["files"].items(): pins[str(run / "external_future-v1" / name)] = checksum
    support = context_support(contexts)
    source_path = c4 / "scripts/select_hybrid_fresh_policies.py"
    pins[str(source_path)] = d.sha(source_path.read_bytes())
    claim = dict(purpose="fresh_four_method_policy_selection_v1", inputs=pins, environment=u.q.c.environment(),
        policy_future_batch_sha256=report_hash, N2_calibration_sha256=evaluator["N2_calibration_sha256"],
        model_bindings=evaluator["model_bindings"], primary_method=u.q.PRIMARY, methods=methods,
        definition=definition, candidates=policies, cases=len(rows), assigned_groups=12,
        nonempty_groups=int(index.group_id.nunique()), blocked_cases_retained=report["blocked_cases_retained"],
        scenario_ids=index.scenario_id.tolist(), selection_role="policy_calibration",
        signal_derivation="Post-outcome replay of pre-future evidence and predeclared deterministic rules; no rule/parameter changes.",
        context_support=support, N1_refitting_authorized=False, N2_refitting_authorized=False,
        future_reevaluation_authorized=False, outer_evaluation_authorized=False, production_release_authorized=False,
        formal_conditional_risk_guarantee_established=False)
    u.verify_inputs(pins)
    return claim, index, targets, decision_inputs, calibration["methods"], pd.DataFrame(contexts)


def replay_signals(decision_inputs, n2_states, policies, unsupported_contexts):
    """No future targets are accepted by this deterministic decision replay."""
    d.require(decision_inputs and len({r["scenario_id"] for r in decision_inputs}) == len(decision_inputs)
              and all(r["split_role"] == "policy_calibration" for r in decision_inputs), "Only unique policy decision inputs allowed.")
    signals, safety = [], {}
    for item in decision_inputs:
        sid = item["scenario_id"]
        d.require(set(item["gates"]) == set(GATES) and set(item["intervals"]) == set(n2_states), "Decision-time gate/method inventory differs.")
        safety[sid] = {}
        for method, state in n2_states.items():
            supported = state["auto_deploy_supported"]
            safety[sid][method] = assess_safety(item["intervals"][method], auto_deploy_supported=supported)
            reference = None
            for family, candidates in policies.items():
                for policy in candidates:
                    result = policy_decision(item["intervals"][method], item["features"], item["gates"],
                        n2_supported=supported, policy=policy, unsupported_contexts=unsupported_contexts)
                    if reference is None: reference = result["decision"] == "ESCALATE"
                    d.require((result["decision"] == "ESCALATE") == reference, "Candidate buffers changed mandatory escalation.")
                    signals.append(dict(scenario_id=sid, group_id=item["group_id"], split_role="policy_calibration",
                        method=method, family=family, policy_id=policy["policy_id"], decision=result["decision"], reason=result["reason"]))
    return pd.DataFrame(signals), safety


def select_methods(signals, targets, index, definition, policies):
    d.require(index.split_role.eq("policy_calibration").all() and index.scenario_id.tolist() == targets.index.tolist()
              and list(targets) == list(d.TARGETS), "Policy selection requires aligned policy targets only.")
    methods = set(signals.method)
    d.require(methods == {u.q.PRIMARY, *u.q.b.METHODS}, "Exactly four fixed methods required.")
    expected_ids = {p["policy_id"] for candidates in policies.values() for p in candidates}
    d.require(not signals.duplicated(["method", "policy_id", "scenario_id"]).any()
              and len(signals) == len(index)*len(methods)*len(expected_ids), "Missing/duplicate policy signals.")
    membership = index.set_index("scenario_id").group_id.to_dict()
    d.require(signals.split_role.eq("policy_calibration").all()
              and all(membership.get(r.scenario_id) == r.group_id for r in signals.itertuples()), "Signal role/group binding differs.")
    target_rows, selections, metrics_rows, group_rows, final_rows = targets.to_dict("index"), {}, [], [], []
    for method in sorted(methods):
        subset = signals.loc[signals.method.eq(method)]
        d.require(set(subset.policy_id) == expected_ids, "Method policy inventory differs.")
        reference = subset.loc[subset.policy_id.eq("global-00")].set_index("scenario_id").decision.to_dict()
        selections[method] = {}
        for family, candidates in policies.items():
            decisions = {}
            for policy in candidates:
                selected = subset.loc[subset.policy_id.eq(policy["policy_id"])]
                d.require(selected.family.eq(family).all(), "Candidate family differs.")
                decisions[policy["policy_id"]] = selected.set_index("scenario_id").decision.to_dict()
            result = select_policy(candidates, decisions, target_rows, index.to_dict("records"), definition["selection"])
            final = ({sid: "ESCALATE" if value == "ESCALATE" else "HUMAN_REVIEW" for sid, value in reference.items()}
                     if result["selected_policy"]["kind"] == "review_only" else decisions[result["selected_policy"]["policy_id"]])
            result["selected_or_fallback_metrics"] = policy_statistics(final, target_rows, index.to_dict("records"))
            selections[method][family] = result
            final_rows.extend(dict(method=method, family=family, scenario_id=sid, group_id=membership[sid],
                split_role="policy_calibration", selected_policy_id=result["selected_policy"]["policy_id"], decision=value) for sid, value in final.items())
            for entry in result["candidates"]:
                metrics = entry["metrics"]
                metrics_rows.append(dict(method=method, family=family, policy_id=entry["policy"]["policy_id"], eligible=entry["eligible"],
                    equal_group_automation=metrics["equal_group_automation"], equal_group_conditional_unsafe_rate=metrics["equal_group_conditional_unsafe_rate"],
                    auto_deploy_groups=metrics["auto_deploy_groups"], unsafe_auto_deploy_count=metrics["unsafe_auto_deploy_count"], **metrics["decision_counts"]))
                group_rows.extend(dict(method=method, family=family, policy_id=entry["policy"]["policy_id"], **row) for row in metrics["per_group"])
    return selections, pd.DataFrame(metrics_rows), pd.DataFrame(group_rows), pd.DataFrame(final_rows)


def save_results(root, claim, index, targets, contexts, signals, safety, results):
    selections, candidate_metrics, group_metrics, final_decisions = results
    directory = f.bundle_at(root) / "policy-selection-v1"
    u.verify_inputs(claim["inputs"])
    f.ensure_documents(directory, {"claim": claim})
    _, claim_hash = d.checked_json(directory, "claim")
    selection = dict(purpose="fresh_frozen_empirical_global_and_adaptive_policy_selection", claim_sha256=claim_hash,
        primary_method=claim["primary_method"], methods=selections, model_bindings=claim["model_bindings"],
        N2_calibration_sha256=claim["N2_calibration_sha256"], definition=claim["definition"], context_support=claim["context_support"],
        evaluation="policy_calibration_selection_not_final_test", formal_conditional_risk_guarantee_established=False,
        production_release_authorized=False)
    files = {"selection.json": e.encoded(selection), "N3_assessments.json": e.encoded(safety),
        "context_support.json": e.encoded(claim["context_support"])}
    files["selection.sha256"] = (d.sha(files["selection.json"])+"\n").encode()
    for name, table, indexed in (("case_index.csv", index, False), ("targets.csv", targets, True),
        ("contexts.csv", contexts, False), ("candidate_signals.csv", signals, False),
        ("candidate_metrics.csv", candidate_metrics, False), ("candidate_group_metrics.csv", group_metrics, False),
        ("selected_decisions.csv", final_decisions, False)):
        files[name] = table.to_csv(index=indexed, lineterminator="\n").encode()
    summary = dict(purpose="verified_fresh_four_method_policy_selection", claim_sha256=claim_hash,
        files={name: d.sha(data) for name, data in files.items()}, cases=claim["cases"], assigned_groups=12,
        nonempty_groups=claim["nonempty_groups"], blocked_cases_retained=claim["blocked_cases_retained"],
        N1_refitted=False, N2_refitted=False, future_reevaluated=False, outer_test_evaluated=False,
        production_release_authorized=False, selection_frozen_for_final_test=True)
    output = directory / "completion-v1"
    if output.exists():
        previous, _ = d.checked_json(output)
        d.require(previous == summary, "Existing selection differs; preserve it.")
        for name, digest in previous["files"].items(): d.checked(output / name, digest)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items(): (output / name).write_bytes(data)
        data = e.encoded(summary)
        (output / "metadata.json").write_bytes(data)
        (output / "metadata.sha256").write_text(d.sha(data)+"\n", encoding="utf-8")
    u.verify_inputs(claim["inputs"])
    loaded, _ = d.checked_json(output, "selection")
    d.require(loaded["methods"] == selections, "Selected policy JSON reload differs.")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    raw = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(raw), "Set C4_DATA_ROOT.")
    root = Path(raw)
    claim, index, targets, decision_inputs, n2_states, contexts = prepare(root)
    print("VERIFIED POLICY CASES:", len(index), "NONEMPTY GROUPS:", index.group_id.nunique(), flush=True)
    print("CONTEXT SUPPORT:", claim["context_support"]["distinct_admitted_groups_by_context"], flush=True)
    policies = claim["candidates"]
    signals, safety = replay_signals(decision_inputs, n2_states, policies, claim["context_support"]["unsupported_contexts"])
    results = select_methods(signals, targets, index, claim["definition"], policies)
    for method, families in results[0].items():
        for family, result in families.items():
            metrics = result["selected_or_fallback_metrics"]
            print("METHOD:", method, "FAMILY:", family, "STATUS:", result["status"])
            print("SELECTED:", result["selected_policy"], "DECISIONS:", metrics["decision_counts"])
            print("GROUP AUTOMATION:", metrics["equal_group_automation"], "CONDITIONAL UNSAFE RATE:", metrics["equal_group_conditional_unsafe_rate"])
    print("EMPIRICAL POLICY SELECTION; NO FORMAL CONDITIONAL-RISK GUARANTEE")
    if not args.save:
        print("PLAN ONLY. No files written. Use --save to freeze the selection.")
        return
    output = save_results(root, claim, index, targets, contexts, signals, safety, results)
    print("FOUR-METHOD GLOBAL / ADAPTIVE POLICIES FROZEN / JSON RELOAD VERIFIED")
    print("N1/N2 REFITS: 0; FUTURE REEVALUATIONS: 0; OUTER CASES ACCESSED: 0")
    print("SELECTION:", output / "selection.json")
    print("METADATA:", output / "metadata.json")


if __name__ == "__main__":
    main()
