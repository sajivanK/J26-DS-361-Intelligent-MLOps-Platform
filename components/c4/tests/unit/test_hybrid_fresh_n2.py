"""N2 technical checks; synthetic coverage is not empirical research evidence."""
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import calibrate_hybrid_fresh_n2 as n


def doc(directory, stem, value):
    directory.mkdir(parents=True, exist_ok=True)
    data = n.e.encoded(value)
    (directory / (stem+".json")).write_bytes(data)
    (directory / (stem+".sha256")).write_text(n.d.sha(data)+"\n")
    return n.d.sha(data)


@pytest.fixture
def tmp_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("c4n2")
    if os.name == "nt":
        raw = str(directory.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\UNC\\"+raw[2:] if raw.startswith("\\\\") else "\\\\?\\"+raw
        directory = Path(raw)
    yield directory
    if os.name == "nt" and directory.exists():
        shutil.rmtree(directory)


def fixture_tables(nonempty=20):
    ids = [f"s{i}" for i in range(nonempty)]
    index = pd.DataFrame(dict(scenario_id=ids, group_id=[f"g{i:02}" for i in range(nonempty)],
        split_role=["conformal_calibration"]*nonempty))
    target_index = pd.Index(ids, name="scenario_id")
    targets = pd.DataFrame({t: np.arange(nonempty)*.001 for t in n.d.TARGETS}, index=target_index)
    predictions = pd.DataFrame({t+"__"+label: np.full(nonempty, value) for t in n.d.TARGETS
        for label, value in (("lower", -.01), ("median", 0.), ("upper", .01))}, index=target_index)
    forecasts = {name: predictions.copy() for name in (n.u.q.PRIMARY, *n.u.q.b.METHODS)}
    groups = pd.DataFrame(dict(group_id=[f"g{i:02}" for i in range(24)],
        admitted=[1]*nonempty+[0]*(24-nonempty), blocked=[36]*nonempty+[37]*(24-nonempty)))
    return index, targets, forecasts, groups


def test_exact_twenty_group_rank_and_adjustment():
    index, targets, forecasts, groups = fixture_tables()
    n.verify_inventory(index, targets, groups)
    results = n.fit_methods(index, targets, forecasts)
    assert len(results) == 4
    for result in results.values():
        state = result["calibration"]
        assert state["nonempty_groups"] == state["rank"] == 20
        assert state["fallback"] is False
        assert state["adjustment"] == pytest.approx(.009)
        assert result["diagnostics"]["independent_coverage_validation"] is False


@pytest.mark.parametrize("count,rank,fallback", [(18, 19, True), (19, 19, False), (20, 20, False)])
def test_original_support_boundary(count, rank, fallback):
    index, targets, forecasts, _ = fixture_tables(count)
    results = n.fit_methods(index, targets, forecasts)
    for result in results.values():
        state = result["calibration"]
        assert state["rank"] == rank and state["fallback"] is fallback
        if fallback:
            assert state["adjustment"] is None
            for t in n.d.TARGETS:
                assert result["intervals"][t+"__lower"].eq(-1).all()
                assert result["intervals"][t+"__upper"].eq(1).all()


def test_group_max_across_variants_and_all_three_targets():
    index, targets, forecasts, _ = fixture_tables()
    # A second admitted variant shares g00. Only one target has a large error.
    second = pd.DataFrame(dict(scenario_id=["extra"], group_id=["g00"], split_role=["conformal_calibration"]))
    index = pd.concat([index, second], ignore_index=True)
    targets.loc["extra"] = [0., -.7, 0.]
    for predicted in forecasts.values(): predicted.loc["extra"] = predicted.iloc[0]
    results = n.fit_methods(index, targets, forecasts)
    for result in results.values():
        assert len(result["group_scores"]) == 20
        assert result["group_scores"].loc["g00", "scenario_count"] == 2
        assert result["group_scores"].loc["g00", "group_score"] == pytest.approx(.69)
        assert result["calibration"]["adjustment"] == pytest.approx(.69)


@pytest.mark.parametrize("change", ["policy", "outer", "duplicate", "target_order", "unknown_group", "drop_group", "counts"])
def test_invalid_inventory_rejected(change):
    index, targets, _, groups = fixture_tables()
    if change in ("policy", "outer"):
        index.loc[0, "split_role"] = "policy_calibration" if change == "policy" else "outer_test"
    elif change == "duplicate": index.loc[0, "scenario_id"] = index.loc[1, "scenario_id"]
    elif change == "target_order": targets = targets.iloc[::-1]
    elif change == "unknown_group": index.loc[0, "group_id"] = "unknown"
    elif change == "drop_group": groups = groups.iloc[:-1]
    else: groups.loc[0, "admitted"] = 2; groups.loc[0, "blocked"] = 35
    with pytest.raises(ValueError): n.verify_inventory(index, targets, groups)


def test_missing_method_rejected():
    index, targets, forecasts, _ = fixture_tables()
    forecasts.pop(n.u.q.PRIMARY)
    with pytest.raises(ValueError): n.fit_methods(index, targets, forecasts)


def test_misordered_forecast_rejected():
    index, targets, forecasts, _ = fixture_tables()
    forecasts[n.u.q.PRIMARY] = forecasts[n.u.q.PRIMARY].iloc[::-1]
    with pytest.raises(ValueError): n.fit_methods(index, targets, forecasts)


def test_four_methods_are_calibrated_separately():
    index, targets, forecasts, _ = fixture_tables()
    method = n.u.q.PRIMARY
    for target in n.d.TARGETS:
        forecasts[method][target+"__upper"] = .5
    result = n.fit_methods(index, targets, forecasts)
    assert result[method]["calibration"]["adjustment"] == 0
    for key in set(result)-{method}:
        assert result[key]["calibration"]["adjustment"] == pytest.approx(.009)


def artifact_fixture(tmp_path):
    index, targets, forecasts, groups = fixture_tables()
    source = tmp_path / "source.txt"
    source.write_bytes(b"fixed source")
    claim = dict(inputs={str(source): n.d.sha(source.read_bytes())}, definition=n.EXPECTED_RULES,
        model_bindings={method: {"fixture": True} for method in forecasts},
        assigned_groups=24, nonempty_groups=20, all_blocked_groups=4,
        blocked_cases_retained=int(groups.blocked.sum()), support_note="Numerical support only",
        limitation="Synthetic technical fixture")
    return claim, index, targets, forecasts, groups, n.fit_methods(index, targets, forecasts)


def test_save_reload_and_idempotent_reuse(tmp_path):
    args = artifact_fixture(tmp_path)
    output = n.save_results(tmp_path, *args)
    before = (output / "metadata.json").read_bytes()
    state, _ = n.d.checked_json(output, "calibration")
    assert len(state["methods"]) == 4
    assert state["independent_coverage_validated"] is False
    assert state["production_release_authorized"] is False
    assert state["all_blocked_groups"] == 4
    assert n.save_results(tmp_path, *args) == output
    assert (output / "metadata.json").read_bytes() == before
    for method, payload in state["methods"].items():
        predicted = args[3][method]
        restored = n.apply_joint_calibration(n.JointCalibration.from_dict(payload), predicted)
        pd.testing.assert_frame_equal(restored, args[-1][method]["intervals"], check_exact=True)


@pytest.mark.parametrize("change", ["input", "output", "claim"])
def test_changed_artifacts_never_overwritten(tmp_path, change):
    args = artifact_fixture(tmp_path)
    output = n.save_results(tmp_path, *args)
    if change == "input": Path(next(iter(args[0]["inputs"]))).write_bytes(b"changed")
    elif change == "output": (output / "group_summary.csv").write_bytes(b"changed")
    else: args[0]["limitation"] = "changed"
    with pytest.raises(ValueError): n.save_results(tmp_path, *args)


def test_boundaries_and_frozen_rules_unchanged():
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    assert all(protocol["N2"][key] == value for key, value in n.EXPECTED_RULES.items())
    policy = json.loads((C4 / "configs/policy_benchmark_v1.json").read_text())
    assert policy["safety_boundaries"] == dict(zip(n.d.TARGETS, (0., 0., -.02)))


def test_incomplete_future_completion_rejected_before_calibration(tmp_path, monkeypatch):
    claim = {"fixture": "claim"}
    monkeypatch.setattr(n.u, "prepare", lambda *a: (claim, [], None, None))
    directory = n.f.bundle_at(tmp_path) / "future-evaluator-v1"
    n.f.ensure_documents(directory, {"claim": claim})
    with pytest.raises(FileNotFoundError): n.prepare(tmp_path)


def test_completed_snapshot_preflight_and_forecast_assembly(tmp_path, monkeypatch):
    index, targets, forecasts, groups = fixture_tables()
    rows = index.to_dict("records")
    records = [dict(scenario_id=sid, group_id=index.iloc[i].group_id,
        split_role="conformal_calibration", primary_targets=targets.loc[sid].to_dict())
        for i, sid in enumerate(targets.index)]
    evaluator = dict(inputs={}, cohort_csv_sha256="cohort-fixture",
        model_bindings={method: {} for method in forecasts}, limitation="fixture")
    directory = n.f.bundle_at(tmp_path) / "future-evaluator-v1"
    n.f.ensure_documents(directory, {"claim": evaluator})
    _, evaluator_hash = n.d.checked_json(directory, "claim")
    output = directory / "completion-v1"
    output.mkdir()
    tables = {"case_index.csv": index.to_csv(index=False).encode(),
        "targets.csv": targets.to_csv(index_label="scenario_id").encode()}
    for name, data in tables.items(): (output / name).write_bytes(data)
    report = dict(purpose="fresh_conformal_future_evaluation_complete_not_final_outer_results",
        evaluator_claim_sha256=evaluator_hash, cases=20, forecasts_frozen_before_future=True,
        training_executed=False, N2_calibrated=False, policy_calibrated=False, production_release_authorized=False,
        cohort_csv_sha256="cohort-fixture", files={name: n.d.sha(data) for name, data in tables.items()},
        records=records, assigned_groups=24, nonempty_groups=20, blocked_cases_retained=int(groups.blocked.sum()))
    # Write report beside existing CSVs using the existing immutable document helper.
    doc(output, "future_batch", report)
    doc(n.f.bundle_at(tmp_path) / "candidate-executor-v1", "claim", {"group_counts": groups.to_dict("records")})
    for row, record in zip(rows, records):
        run = n.u.run_at(tmp_path, row)
        payload = {method: {"processed": frame.loc[row["scenario_id"]].to_dict()} for method, frame in forecasts.items()}
        data = n.e.encoded(payload)
        path = run / "forecast_freeze-v1/forecasts.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        paired = run / "external_future-v1/paired_predictions.csv"
        paired.parent.mkdir()
        paired.write_bytes(b"paired-fixture")
        doc(paired.parent, "outcomes", {"files": {"paired_predictions.csv": n.d.sha(paired.read_bytes())}})
        record["forecasts_sha256"] = n.d.sha(data)
    # Report record list stays exactly matched to the verified records.
    doc(output, "future_batch", report)
    protocol_dir = tmp_path / "c4_artifacts/benchmark_protocols/adult" / n.v.VERSION
    protocol_hash = doc(protocol_dir, "protocol", {"N2": n.EXPECTED_RULES})
    doc(protocol_dir, "freeze", {"protocol_sha256": protocol_hash})
    monkeypatch.setattr(n.u, "prepare", lambda *a: (evaluator, rows, None, None))
    monkeypatch.setattr(n.u, "load_cohort", lambda _: "verified-cohort-fixture")
    monkeypatch.setattr(n.u, "completed_case", lambda root, row, *a: next(r for r in records if r["scenario_id"] == row["scenario_id"]))
    monkeypatch.setattr(n.u, "source_record", lambda claim, row: next(r for r in records if r["scenario_id"] == row["scenario_id"]))
    monkeypatch.setattr(n.u.q.c, "environment", lambda: {})
    claim, actual_index, actual_targets, actual_forecasts, actual_groups = n.prepare(tmp_path)
    pd.testing.assert_frame_equal(actual_index, index)
    pd.testing.assert_frame_equal(actual_targets, targets)
    assert claim["nonempty_groups"] == 20 and claim["all_blocked_groups"] == 4
    assert len(actual_groups) == 24
    for method in forecasts:
        pd.testing.assert_frame_equal(actual_forecasts[method], forecasts[method])
