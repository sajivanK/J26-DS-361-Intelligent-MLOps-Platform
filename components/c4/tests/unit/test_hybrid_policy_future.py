"""Technical future-evaluator checks; fixture metrics are not research results."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import _hybrid_policy_future as u
import run_hybrid_policy_future as r


@pytest.fixture
def tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("c4future")
    if os.name == "nt":
        raw = str(path.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\UNC\\"+raw[2:] if raw.startswith("\\\\") else "\\\\?\\"+raw
        path = Path(raw)
    yield path
    if os.name == "nt" and path.exists():
        shutil.rmtree(path)


def doc(directory, stem, value):
    directory.mkdir(parents=True, exist_ok=True)
    data = u.e.encoded(value)
    (directory / (stem+".json")).write_bytes(data)
    (directory / (stem+".sha256")).write_text(u.d.sha(data)+"\n")
    return u.d.sha(data)


@pytest.fixture
def case(tmp_path):
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    manifest = u.v.create_design(protocol)
    row = next(s for s in manifest["scenarios"] if s["split_role"] == "policy_calibration" and s["is_no_drift_control"])
    clean = pd.DataFrame({col: np.zeros(8) for col in u.FEATURE_COLUMNS},
        index=pd.Index(np.arange(1_000_000, 1_000_008), name="source_row_id"))
    clean["age"] = [-1, -1, -1, -1, 1, 1, 1, 1]
    clean["target"] = [0, 1, 0, 1, 0, 1, 0, 1]
    cohort_path = tmp_path / "external.csv"
    data = clean.to_csv(index_label="source_row_id").encode()
    cohort_path.write_bytes(data)
    run = u.run_at(tmp_path, row)
    forecast = run / "forecast_freeze-v1"
    forecast.mkdir(parents=True)
    values = {name: {kind: {t+"__"+label: val for t in u.d.TARGETS
                for label, val in (("lower", -.1), ("median", 0.), ("upper", .1))}
                for kind in ("raw", "processed", "calibrated")} for name in (u.q.PRIMARY, *u.q.b.METHODS)}
    forecasts = u.e.encoded(values)
    (forecast / "forecasts.json").write_bytes(forecasts)
    freeze_hash = doc(forecast, "freeze", {"fixture": True})
    source = dict(N2_calibration_sha256="fixture-n2", scenario_id=row["scenario_id"], group_id=row["group_id"], split_role=row["split_role"],
        forecast_freeze_sha256=freeze_hash, forecasts_sha256=u.d.sha(forecasts))
    claim = dict(N2_calibration_sha256="fixture-n2", scenario_ids=[row["scenario_id"]], forecast_records=[source],
        cohort_path=str(cohort_path), cohort_csv_sha256=u.d.sha(data), expected_cohort_rows=8,
        champion_model_sha256="champion-fixture", champion_directory=str(tmp_path / "champion"),
        true_slice=dict(name="age", feature="age", operator="le", resolved_threshold=0.),
        shared_drift_types_sha256="shared-fixture", feature_metadata_sha256="types-fixture",
        assigned_groups=12, nonempty_admission_groups=1, blocked_cases_retained=443,
        model_bindings={name: {} for name in values}, inputs={}, limitation="Synthetic technical fixture")
    claim["candidate_models"] = {row["scenario_id"]: "candidate-fixture"}
    candidate_dir = run / "scoped_candidate_validation-v1"
    doc(candidate_dir, "metadata", dict(model_sha256="candidate-fixture", champion_model_sha256="champion-fixture"))
    from c4.scenarios.config import CONTINUOUS_FEATURES
    types = {col: "continuous" if col in CONTINUOUS_FEATURES else "categorical" for col in u.FEATURE_COLUMNS}
    return SimpleNamespace(root=tmp_path, row=row, clean=clean, claim=claim, types=types, candidate_dir=candidate_dir)


class PatternModel:
    def __init__(self, values):
        self.values = values

    def predict(self, x):
        return np.asarray(self.values[:len(x)])


def use_fixture_models(monkeypatch):
    champion = PatternModel([0, 0, 0, 1, 0, 1, 0, 0])
    candidate = PatternModel([0, 1, 0, 1, 0, 1, 1, 1])
    monkeypatch.setattr(u, "load_candidate", lambda _: (candidate, {"model_sha256": "candidate-fixture", "champion_model_sha256": "champion-fixture"}))
    return champion


def test_paired_accuracy_sign_and_population_aggregation():
    predictions = pd.DataFrame(dict(source_row_id=[1, 2, 3, 4], target=[0, 1, 0, 1],
        true_slice_member=[1, 1, 0, 0], champion_prediction=[0, 0, 0, 1], candidate_prediction=[0, 1, 1, 0]))
    targets, comparisons = u.paired_results(predictions)
    assert list(targets.values()) == [-.25, .5, -1.]
    assert comparisons["true_injected_slice"]["candidate_only_correct"] == 1
    assert targets[u.d.TARGETS[0]] == .5*(targets[u.d.TARGETS[1]]+targets[u.d.TARGETS[2]])


@pytest.mark.parametrize("change", ["label", "prediction", "empty_slice", "duplicate_id", "schema"])
def test_invalid_paired_records_rejected(change):
    frame = pd.DataFrame(dict(source_row_id=[1, 2], target=[0, 1], true_slice_member=[1, 0],
        champion_prediction=[0, 1], candidate_prediction=[1, 1]))
    if change == "label": frame.loc[0, "target"] = 2
    elif change == "prediction": frame.loc[0, "candidate_prediction"] = np.nan
    elif change == "empty_slice": frame["true_slice_member"] = 0
    elif change == "duplicate_id": frame["source_row_id"] = 1
    else: frame["extra"] = 0
    with pytest.raises(ValueError): u.paired_results(frame)


def test_external_roundtrip_and_identity(case):
    actual = u.load_cohort(case.claim)
    pd.testing.assert_frame_equal(actual, case.clean)


@pytest.mark.parametrize("change", ["sha", "count", "id", "nonfinite"])
def test_invalid_cohort_rejected(case, change):
    claim = dict(case.claim)
    if change == "sha": claim["cohort_csv_sha256"] = "0"*64
    elif change == "count": claim["expected_cohort_rows"] += 1
    else:
        clean = case.clean.copy()
        if change == "id": clean.index = [1]*len(clean)
        else:
            clean = clean.astype(float)
            clean.iloc[0, 0] = np.inf
        data = clean.to_csv(index_label="source_row_id").encode()
        Path(claim["cohort_path"]).write_bytes(data)
        claim["cohort_csv_sha256"] = u.d.sha(data)
    with pytest.raises(ValueError): u.load_cohort(claim)


def test_case_save_reload_and_no_overwrite(case, monkeypatch):
    champion = use_fixture_models(monkeypatch)
    record = u.evaluate_case(case.root, case.row, case.claim, "claim", case.clean, champion, None, case.types)
    assert u.completed_case(case.root, case.row, case.claim, "claim", case.clean) == record
    assert record["primary_targets"][u.d.TARGETS[1]] == .25
    assert record["primary_targets"][u.d.TARGETS[2]] == 0
    with pytest.raises(FileExistsError):
        u.evaluate_case(case.root, case.row, case.claim, "claim", case.clean, champion, None, case.types)


@pytest.mark.parametrize("change", ["claim", "forecast", "cohort", "candidate", "target", "membership", "file_bytes"])
def test_tampered_checkpoint_rejected(case, monkeypatch, change):
    champion = use_fixture_models(monkeypatch)
    u.evaluate_case(case.root, case.row, case.claim, "claim", case.clean, champion, None, case.types)
    directory = u.run_at(case.root, case.row) / "external_future-v1"
    outcome, _ = u.d.checked_json(directory, "outcomes")
    if change == "claim": outcome["evaluator_claim_sha256"] = "changed"
    elif change == "forecast": outcome["forecast_record"] = {}
    elif change == "cohort": outcome["cohort_csv_sha256"] = "changed"
    elif change == "candidate": outcome["candidate_model_sha256"] = "changed"
    elif change == "target": outcome["primary_targets"][u.d.TARGETS[0]] = .9
    elif change == "membership":
        path = directory / "paired_predictions.csv"
        predictions = pd.read_csv(path)
        predictions["true_slice_member"] = 1-predictions.true_slice_member
        data = predictions.to_csv(index=False).encode()
        path.write_bytes(data)
        outcome["files"]["paired_predictions.csv"] = u.d.sha(data)
    else:
        with (directory / "paired_predictions.csv").open("ab") as stream: stream.write(b"changed")
    doc(directory, "outcomes", outcome)
    with pytest.raises(ValueError): u.completed_case(case.root, case.row, case.claim, "claim", case.clean)


def test_incomplete_checkpoint_stops(case):
    (u.run_at(case.root, case.row) / "external_future-v1").mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        u.completed_case(case.root, case.row, case.claim, "claim", case.clean)


def test_policy_case_rejected_before_loading(case, monkeypatch):
    monkeypatch.setattr(u, "load_candidate", lambda _: pytest.fail("Must not load candidate"))
    with pytest.raises(ValueError, match="allowlist"):
        u.evaluate_case(case.root, {**case.row, "split_role": "outer_test"}, case.claim,
            "claim", case.clean, None, None, case.types)


def test_missing_full_forecast_batch_rejected_before_outcome_parse(case, monkeypatch):
    claim = dict(N2_calibration_sha256="fixture-n2", N2_calibrated=True, environment={}, scenario_ids=[case.row["scenario_id"]], future_evaluation_authorized=False,
        model_refitting_authorized=False, production_release_authorized=False, model_bindings={}, inputs={},
        assigned_groups=12, nonempty_admission_groups=1, blocked_cases_retained=443)
    monkeypatch.setattr(u.q.c, "environment", lambda: {})
    batch = dict(N2_calibration_sha256="fixture-n2", cases=1, model_bindings={}, forecasts_frozen_before_future=True, future_evaluation_executed=False,
        N2_calibrated=True, production_release_authorized=False, assigned_groups=12, nonempty_admission_groups=1,
        blocked_cases_retained=443, records=[])
    monkeypatch.setattr(u.q, "completed_forecast", lambda *a: None)
    with pytest.raises(ValueError, match="completion"):
        u.verify_forecast_batch(case.root, [case.row], claim, batch, {})


def test_real_native_candidate_long_path_evaluation(case):
    from xgboost import XGBClassifier
    x, y = case.clean[list(u.FEATURE_COLUMNS)], case.clean.target
    model = XGBClassifier(n_estimators=3, max_depth=1, n_jobs=1).fit(x, y)
    # Native writer receives only a short temp path; evaluator reads bytearray.
    u.q.c.save_candidate_model_portable(model, case.candidate_dir / "candidate.ubj")
    sha = u.d.sha((case.candidate_dir / "candidate.ubj").read_bytes())
    case.claim["candidate_models"][case.row["scenario_id"]] = sha
    doc(case.candidate_dir, "metadata", dict(role="candidate", feature_columns=list(u.FEATURE_COLUMNS),
        model_sha256=sha, champion_model_sha256=case.claim["champion_model_sha256"]))
    record = u.evaluate_case(case.root, case.row, case.claim, "claim", case.clean, model, None, case.types)
    assert all(value == 0 for value in record["primary_targets"].values())


def test_batch_resume_reuses_outcomes_and_exact_metrics(case, monkeypatch):
    champion = use_fixture_models(monkeypatch)
    monkeypatch.setattr(u, "load_champion", lambda _: (champion, {"model_sha256": "champion-fixture"}))
    r.run_batch(case.root, case.claim, [case.row], None, case.types, 1)
    output = u.f.bundle_at(case.root) / "policy-future-evaluator-v1/completion-v1"
    before = (output / "future_batch.json").read_bytes()
    monkeypatch.setattr(u, "evaluate_case", lambda *a: pytest.fail("Completed case must be reused"))
    r.run_batch(case.root, case.claim, [case.row], None, case.types, 76)
    assert (output / "future_batch.json").read_bytes() == before
    metrics = pd.read_csv(output / "forecast_metrics.csv")
    assert len(metrics) == 12
    assert set(metrics.method) == {u.q.PRIMARY, *u.q.b.METHODS}


def test_future_seed_uses_frozen_sealed_future_role(case):
    assert u.configuration(case.row).seed == case.row["drift_seeds"]["sealed_future"]


def test_bounded_batch_one_then_resume_remaining(case, monkeypatch):
    champion = use_fixture_models(monkeypatch)
    monkeypatch.setattr(u, "load_champion", lambda _: (champion, {"model_sha256": "champion-fixture"}))
    second = {**case.row, "scenario_id": case.row["scenario_id"]+"-fixture2"}
    second_run = u.run_at(case.root, second)
    shutil.copytree(u.run_at(case.root, case.row) / "forecast_freeze-v1", second_run / "forecast_freeze-v1")
    doc(second_run / "scoped_candidate_validation-v1", "metadata", dict(model_sha256="candidate-fixture"))
    case.claim["scenario_ids"].append(second["scenario_id"])
    case.claim["candidate_models"][second["scenario_id"]] = "candidate-fixture"
    case.claim["forecast_records"].append({**case.claim["forecast_records"][0], "scenario_id": second["scenario_id"]})
    rows = [case.row, second]
    r.run_batch(case.root, case.claim, rows, None, case.types, 1)
    assert not (second_run / "external_future-v1").exists()
    output = u.f.bundle_at(case.root) / "policy-future-evaluator-v1/completion-v1"
    assert not output.exists()
    first = (u.run_at(case.root, case.row) / "external_future-v1/outcomes.json").read_bytes()
    r.run_batch(case.root, case.claim, rows, None, case.types, 76)
    report, _ = u.d.checked_json(output, "future_batch")
    assert report["cases"] == 2
    assert (u.run_at(case.root, case.row) / "external_future-v1/outcomes.json").read_bytes() == first


def test_metric_known_values_equal_group_weighting(case):
    records = [dict(scenario_id=case.row["scenario_id"], primary_targets={t: .02 for t in u.d.TARGETS})]
    _, _, metrics = u.metric_tables(case.root, [case.row], records, case.claim)
    np.testing.assert_allclose(metrics.MAE_pp, 2.)
    np.testing.assert_allclose(metrics.RMSE_pp, 2.)
    assert metrics.r2.isna().all()  # One constant target: R2 is undefined.
    # q05 loss=.006, q50=.01, q95=.004 => mean .0066667 fraction.
    np.testing.assert_allclose(metrics.Pinball_pp, 2./3.)


@pytest.mark.parametrize("binding_mode", ["direct", "via_n2"])
def test_prepare_binds_original_cohort_and_sidecarless_champion(case, monkeypatch, binding_mode):
    """Exercise actual preflight document/link logic with a synthetic artifact tree."""
    bundle = u.f.bundle_at(case.root)
    c4 = case.root / "repo/components/c4"
    relatives = ("scripts/_hybrid_policy_future.py", "scripts/run_hybrid_policy_future.py",
        "scripts/_all_mechanism_transforms.py", "scripts/_portable_model_io.py",
        "src/c4/scenarios/config.py", "src/c4/scenarios/slices.py",
        "src/c4/scenarios/adapter.py", "src/c4/scenarios/feature_metadata.py")
    for relative in relatives:
        path = c4 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-source")
    shared_path = c4.parent.parent / "shared/injection/drift_types.py"
    shared_path.parent.mkdir(parents=True)
    shared_path.write_bytes(b"shared-fixture")
    shared_hash = u.d.sha(shared_path.read_bytes())
    types_path = shared_path.parent / "feature_types.json"
    types_path.write_bytes(b"types-fixture")
    types_hash = u.d.sha(types_path.read_bytes())
    champion_dir = case.root / "c4_artifacts/champion/adult/v1"
    champion_dir.mkdir(parents=True)
    champion_data = b"native-model-fixture-not-loaded-in-preflight"
    (champion_dir / "champion.ubj").write_bytes(champion_data)
    champion_hash = u.d.sha(champion_data)
    meta_bytes = u.e.encoded({"model_sha256": champion_hash})
    (champion_dir / "metadata.json").write_bytes(meta_bytes)
    assert not (champion_dir / "metadata.sha256").exists()
    protocol_dir = case.root / "c4_artifacts/benchmark_protocols/adult" / u.v.VERSION
    protocol = dict(source_versions={"champion": "v1"}, true_slice=case.claim["true_slice"])
    protocol_hash = doc(protocol_dir, "protocol", protocol)
    freeze_hash = doc(protocol_dir, "freeze", dict(protocol_sha256=protocol_hash, champion_model_sha256=champion_hash))
    row = case.row
    run = u.run_at(case.root, row)
    doc(run / "pre_release_freeze-v1", "freeze", dict(protocol_freeze_sha256=freeze_hash,
        dependencies={str(champion_dir / "metadata.json"): u.d.sha(meta_bytes)}))
    doc(run, "admission", {"preview_id": "preview"})
    preview = case.root / "c4_artifacts/scenario_previews/adult/preview/release_validation"
    doc(preview, "metadata", {"transformation": {"shared_drift_types_sha256": None,
        "feature_metadata_sha256": types_hash}})
    development = case.root / "c4_artifacts/forecasting_experiments/adult" / u.COHORT_RUN
    recovery = development / "portable-resume-v1"
    cohort = development / "results/external_clean.csv"
    cohort.parent.mkdir(parents=True)
    cohort.write_bytes(Path(case.claim["cohort_path"]).read_bytes())
    original_hash = doc(development, "specification", {"version": "adult-external-development-v3", "inputs": {}})
    from run_n1_fast_improvement import EXTERNAL_SHA
    official = development / "results/adult.test"
    official.write_bytes(b"official-fixture")
    # Patch the constant only for this synthetic provenance tree.
    import run_n1_fast_improvement
    monkeypatch.setattr(run_n1_fast_improvement, "EXTERNAL_SHA", u.d.sha(official.read_bytes()))
    recovery_hash = doc(recovery, "recovery_claim", {"cohort_csv_sha256": u.d.sha(cohort.read_bytes()),
        "original_specification_sha256": original_hash})
    meta_hash = doc(recovery, "metadata", dict(recovery_claim_sha256=recovery_hash,
        files={"recovery_claim.json": recovery_hash}, calibration_policy_outer_cases_accessed=False,
        original_specification_sha256=original_hash,
        cohort={"eligible_rows": 12899, "feature_only_old_overlap_exclusion": True}))
    doc(bundle / "policy-candidate-executor-v1", "claim", {})
    forecast_claim = dict(N2_calibration_sha256="fixture-n2", N2_calibrated=True, environment={}, scenario_ids=[row["scenario_id"]],
        future_evaluation_authorized=False, model_refitting_authorized=False, production_release_authorized=False,
        model_bindings={}, inputs={str(recovery / "metadata.json"): meta_hash},
        assigned_groups=12, nonempty_admission_groups=1, blocked_cases_retained=443)
    if binding_mode == "via_n2":
        parent_dir = bundle / "n2-calibration-v1"
        parent_hash = doc(parent_dir, "claim", dict(purpose="fresh_four_method_N2_group_max_calibration_v1",
            production_release_authorized=False, inputs=forecast_claim["inputs"]))
        forecast_claim["inputs"] = {str(parent_dir / "claim.json"): parent_hash}
    doc(bundle / "policy-forecast-freezer-v1", "claim", forecast_claim)
    source = case.claim["forecast_records"][0]
    source.update(decision_freeze_sha256="fixture", feature_sha256="fixture")
    # Preflight pins exact feature bytes even though this fixture does not reproduce them.
    (run / "forecast_freeze-v1/features.json").write_bytes(b"fixture-features")
    doc(bundle / "policy-forecast-freezer-v1/completion-v1", "forecast_batch", dict(N2_calibration_sha256="fixture-n2", cases=1,
        model_bindings={}, forecasts_frozen_before_future=True, future_evaluation_executed=False,
        N2_calibrated=True, production_release_authorized=False, assigned_groups=12,
        nonempty_admission_groups=1, blocked_cases_retained=443, records=[source]))
    # Preserve verification helper behaviour, mocking only costly upstream/model checks.
    monkeypatch.setattr(u.q.c, "prepare", lambda *a: ({}, [row]))
    monkeypatch.setattr(u.q.c, "environment", lambda: {})
    monkeypatch.setattr(u, "load_models", lambda _: {})
    monkeypatch.setattr(u.q, "completed_forecast", lambda *a: source)
    monkeypatch.setattr(u, "_load_shared_functions", lambda: (None, shared_hash))
    monkeypatch.setattr(u, "load_adult_feature_metadata", lambda: SimpleNamespace(
        feature_types=case.types, source_path=types_path, source_sha256=types_hash))
    # Replace fixture record hashes with their actual documents before preflight.
    source["decision_freeze_sha256"] = u.d.sha((run / "pre_release_freeze-v1/freeze.json").read_bytes())
    source["feature_sha256"] = u.d.sha((run / "forecast_freeze-v1/features.json").read_bytes())
    doc(bundle / "policy-forecast-freezer-v1/completion-v1", "forecast_batch", dict(N2_calibration_sha256="fixture-n2", cases=1,
        model_bindings={}, forecasts_frozen_before_future=True, future_evaluation_executed=False,
        N2_calibrated=True, production_release_authorized=False, assigned_groups=12,
        nonempty_admission_groups=1, blocked_cases_retained=443, records=[source]))
    # Original candidate model bytes are pinned without native loading in preflight.
    candidate_bytes = b"candidate-model-fixture"
    (case.candidate_dir / "candidate.ubj").write_bytes(candidate_bytes)
    doc(case.candidate_dir, "metadata", {"model_sha256": u.d.sha(candidate_bytes)})
    claim, rows, _, types = u.prepare(case.root, c4)
    assert rows == [row] and types == case.types
    assert claim["cohort_csv_sha256"] == u.d.sha(cohort.read_bytes())
    assert claim["champion_model_sha256"] == champion_hash
    assert claim["future_evaluation_authorized"] is True and claim["N2_fitting_authorized"] is False
    assert claim["candidate_models"][row["scenario_id"]] == u.d.sha(candidate_bytes)


@pytest.mark.parametrize("change", ["missing_parent", "changed_parent", "different_source", "wrong_scope", "conflicting_direct_pin"])
def test_unverified_n2_provenance_chain_rejected(tmp_path, change):
    source = tmp_path / "c4_artifacts/forecasting_experiments/adult" / u.COHORT_RUN / "portable-resume-v1/metadata.json"
    directory = u.f.bundle_at(tmp_path) / "n2-calibration-v1"
    parent = dict(purpose="fresh_four_method_N2_group_max_calibration_v1", production_release_authorized=False,
        inputs={str(source): "expected-source"})
    if change == "different_source": parent["inputs"][str(source)] = "different-source"
    if change == "wrong_scope": parent["purpose"] = "unrelated"
    parent_hash = doc(directory, "claim", parent)
    pins = {str(directory / "claim.json"): parent_hash}
    if change == "missing_parent": pins.clear()
    elif change == "changed_parent": (directory / "claim.json").write_bytes(b"changed")
    elif change == "conflicting_direct_pin": pins[str(source)] = "different-source"
    with pytest.raises(ValueError): u.bind_development_metadata(tmp_path, pins, "expected-source")


@pytest.mark.parametrize("actual,expected_coverage", [(.02, 1.), (.2, 0.)])
def test_coverage_uses_fixed_n2_bands_without_refitting(case, actual, expected_coverage):
    sid = case.row["scenario_id"]
    targets = pd.DataFrame({t: [actual] for t in u.d.TARGETS}, index=pd.Index([sid], name="scenario_id"))
    index = pd.DataFrame([dict(scenario_id=sid, group_id=case.row["group_id"], split_role="policy_calibration")])
    intervals, diagnostics = u.interval_diagnostics(case.root, [case.row], targets, index, case.claim)
    for method, result in diagnostics.items():
        assert result["scenario_joint_coverage"] == expected_coverage
        assert result["complete_group_joint_coverage"] == expected_coverage
        assert result["N2_refitted"] is False and result["independent_coverage_validation"] is False
        assert result["evaluation"] == "policy_groups_with_previously_fixed_N2_not_final_outer_test"
        for target in u.d.TARGETS:
            assert intervals[method][target+"__median"].iloc[0] == 0.
            assert result["targets"][target]["mean_width_fraction"] == pytest.approx(.2)


def test_invalid_fixed_n2_band_rejected(case):
    path = u.run_at(case.root, case.row) / "forecast_freeze-v1/forecasts.json"
    payload = json.loads(path.read_bytes())
    payload[u.q.PRIMARY]["calibrated"][u.d.TARGETS[0]+"__lower"] = .5
    data = u.e.encoded(payload)
    path.write_bytes(data)
    case.claim["forecast_records"][0]["forecasts_sha256"] = u.d.sha(data)
    sid = case.row["scenario_id"]
    targets = pd.DataFrame({t: [0.] for t in u.d.TARGETS}, index=pd.Index([sid], name="scenario_id"))
    index = pd.DataFrame([dict(scenario_id=sid, group_id=case.row["group_id"], split_role="policy_calibration")])
    with pytest.raises(ValueError): u.interval_diagnostics(case.root, [case.row], targets, index, case.claim)


def test_different_case_n2_binding_rejected_before_outcomes(case, monkeypatch):
    claim = dict(environment={}, scenario_ids=[case.row["scenario_id"]], future_evaluation_authorized=False,
        model_refitting_authorized=False, production_release_authorized=False, model_bindings={}, inputs={},
        N2_calibrated=True, N2_calibration_sha256="expected", assigned_groups=12,
        nonempty_admission_groups=1, blocked_cases_retained=443)
    record = {"N2_calibration_sha256": "different"}
    batch = dict(cases=1, model_bindings={}, forecasts_frozen_before_future=True, future_evaluation_executed=False,
        N2_calibrated=True, N2_calibration_sha256="expected", production_release_authorized=False,
        assigned_groups=12, nonempty_admission_groups=1, blocked_cases_retained=443, records=[record])
    monkeypatch.setattr(u.q.c, "environment", lambda: {})
    monkeypatch.setattr(u.q, "completed_forecast", lambda *a: record)
    with pytest.raises(ValueError, match="different N2"):
        u.verify_forecast_batch(case.root, [case.row], claim, batch, {})
