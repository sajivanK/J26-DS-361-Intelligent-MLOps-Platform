"""Decision evidence and four-method forecast freezing; no future targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _hybrid_fresh_candidate as c
import fit_n1_validation_baselines as b

d, e, f, v = c.d, c.e, c.f, c.v
PRIMARY = "hybrid_ridge_lightgbm_lightgbm"
SCOPE = "one_admitted_fresh_conformal_scenario_under_simulated_upstream_evidence"


def expected_freezer_source(source):
    source = source.replace("for the admitted training case.", "for an admitted fresh conformal case.")
    source = source.replace("from _general_training import group_context", "from _hybrid_fresh_candidate import group_context")
    source = source.replace("adult-simulation-pp1-v2", v.VERSION)
    source = source.replace("rows[0]['split_role'] == 'forecast_train'", "rows[0]['split_role'] == 'conformal_calibration'")
    return source.replace("one_admitted_forecast_training_scenario_under_simulated_upstream_evidence", SCOPE)


def prepare(root, c4):
    root, c4 = Path(root), Path(c4)
    candidate_claim, rows = c.prepare(root, c4)
    candidate_dir = f.bundle_at(root) / "candidate-executor-v1"
    saved_claim, candidate_claim_hash = d.checked_json(candidate_dir, "claim")
    d.require(e.encoded(saved_claim) == e.encoded(candidate_claim), "Candidate executor binding differs.")
    completion, completion_hash = d.checked_json(candidate_dir / "completion-v1", "candidate_batch")
    d.require(completion["candidates"] == len(rows) and completion["future_evaluation_executed"] is False
              and completion["complete_decision_evidence_freeze"] is False
              and completion["N1_forecasts_frozen"] is False, "Incomplete or invalid candidate batch.")
    records = [c.completed_case(root, row) for row in rows]
    d.require(all(r is not None for r in records) and records == completion["records"],
              "Candidate completion records differ from verified artifacts.")
    original = (c4 / "scripts/freeze_portable_training_case.py").read_text(encoding="utf-8-sig")
    adapted = (c4 / "scripts/freeze_hybrid_fresh_case.py").read_text(encoding="utf-8-sig")
    d.require(adapted == expected_freezer_source(original), "Decision-freeze science changed beyond declared scope adapter.")
    baseline_dir = f.bundle_at(root) / "baseline-fit-v1"
    metadata, baseline_hash = d.checked_json(baseline_dir)
    d.require(set(metadata["models"]) == {PRIMARY, *b.METHODS}, "Incomplete forecaster inventory.")
    models = {}
    for name, binding in metadata["models"].items():
        directory = Path(binding["directory"])
        if name == PRIMARY:
            expected_dir = root / "c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1/fit-v1"
            d.require(directory == expected_dir, "Unexpected hybrid directory.")
            _, digest = d.checked_json(directory)
            d.require(digest == binding["metadata_sha256"], "Hybrid model binding differs.")
            models[name] = b.h.HybridForecaster.load(directory)
        else:
            d.require(directory == baseline_dir / name, "Unexpected baseline directory.")
            _, digest = d.checked_json(directory, "model")
            d.require(digest == binding["model_sha256"], "Baseline model binding differs.")
            models[name] = b.FixedBaseline.load(directory)
            d.require(models[name].kind == b.METHODS[name], "Baseline family differs.")
    columns = models[PRIMARY].features
    d.require(len(columns) == 75 and all(m.features == columns for m in models.values()),
              "Four models must share the exact original 75 feature names/order.")
    pins = dict(saved_claim["inputs"])
    pins[str(candidate_dir / "claim.json")] = candidate_claim_hash
    pins[str(candidate_dir / "completion-v1/candidate_batch.json")] = completion_hash
    pins[str(baseline_dir / "metadata.json")] = baseline_hash
    for row in rows:
        run = root / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]
        for folder in ("external_future-v1", "future_research-v1"):
            d.require(not (run / folder).exists(), "Future-stage artifacts already exist before forecast freezing.")
        for stem in ("validation", "metadata"):
            _, digest = d.checked_json(run / "scoped_candidate_validation-v1", stem)
            pins[str(run / "scoped_candidate_validation-v1" / (stem+".json"))] = digest
    for name in ("_hybrid_fresh_forecasts.py", "freeze_hybrid_fresh_case.py", "run_hybrid_fresh_forecast_freeze.py", "_scenario_meta.py"):
        path = c4 / "scripts" / name
        pins[str(path)] = d.sha(path.read_bytes())
    claim = dict(purpose="fresh_conformal_decision_and_four_method_forecast_freeze_v1",
        candidate_completion_sha256=completion_hash, baseline_metadata_sha256=baseline_hash,
        candidate_executor_sha256=candidate_claim_hash, inputs=pins, environment=c.environment(),
        scenario_ids=[row["scenario_id"] for row in rows], model_bindings=metadata["models"],
        feature_columns=columns, target_columns=list(d.TARGETS), quantiles=[q for q, _ in d.QUANTILES],
        admitted_cases=len(rows), blocked_cases_retained=candidate_claim["blocked_cases"],
        assigned_groups=24, nonempty_admission_groups=candidate_claim["nonempty_admission_groups"],
        feature_populations="global, C2 diagnosed slice, diagnosed-slice complement",
        forecast_target_populations="global, true injected slice, true-slice complement",
        future_evaluation_authorized=False, model_refitting_authorized=False,
        N2_calibrated=False, production_release_authorized=False)
    return claim, rows, models


def verify_decision(root, row):
    run = Path(root) / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]
    directory = run / "pre_release_freeze-v1"
    d.require((directory / "freeze.json").is_file(), "Incomplete/missing decision freeze; preserve artifacts.")
    freeze, digest = d.checked_json(directory, "freeze")
    d.require(freeze["scenario"] == row and row["split_role"] == "conformal_calibration"
              and freeze["scope"] == SCOPE and freeze["complete_decision_evidence_frozen"] is True
              and freeze["future_evaluation_executed"] is False
              and freeze["production_package_approved"] is False
              and freeze["release_authorized"] is False, "Invalid fresh decision-freeze scope.")
    d.require(freeze["runtime_checks"]["schema_compatible"] is True
              and freeze["runtime_checks"]["rollback_ready"] is True, "Decision runtime checks failed.")
    for path, checksum in freeze["dependencies"].items():
        d.checked(Path(path), checksum)
    envelope, envelope_hash = d.checked_json(directory, "upstream_simulation")
    d.require(envelope_hash == freeze["upstream_simulation_sha256"]
              and envelope["scope"] == "controlled_offline_simulation"
              and envelope["production_package_approved"] is False, "Upstream envelope link/scope differs.")
    evidence_dir = Path(root) / "c4_artifacts/pre_release/adult" / row["scenario_id"]
    evidence, evidence_hash = d.checked_json(evidence_dir, "evidence")
    d.require(evidence_hash == freeze["pre_release_evidence_sha256"]
              and evidence["scenario_id"] == row["scenario_id"]
              and evidence["slice_basis"] == "c2_diagnosed_slice", "Pre-release feature evidence link differs.")
    candidate = c.completed_case(root, row)
    d.require(candidate is not None and candidate["pre_release_evidence_sha256"] == evidence_hash,
              "Frozen evidence does not match the completed candidate.")
    return freeze, digest, envelope, evidence


def make_feature_frame(values, columns, scenario_id):
    d.require(set(values) == set(columns) and len(values) == len(columns)
              and len(set(columns)) == len(columns), "Decision feature names differ from trained models.")
    d.require(all(isinstance(value, (float, int, np.number)) and np.isfinite(float(value))
                  for value in values.values()), "Nonfinite/non-numeric decision feature.")
    return pd.DataFrame([[float(values[c]) for c in columns]], columns=columns,
                        index=pd.Index([scenario_id], name="scenario_id"))


def forecast_values(models, x):
    names = [target+"__"+label for target in d.TARGETS for _, label in d.QUANTILES]
    result = {}
    for name, model in models.items():
        d.require(model.features == list(x), "Loaded model feature order differs.")
        raw, processed = model.predict(x, raw=True), model.predict(x)
        d.require(raw.index.equals(x.index) and processed.index.equals(x.index)
                  and list(raw) == list(processed) == names and len(x) == 1,
                  "Invalid forecast output identity/schema.")
        d.require(np.isfinite(raw.to_numpy()).all() and np.isfinite(processed.to_numpy()).all(), "Nonfinite forecasts.")
        for j in range(3):
            expected = e.ordered(raw.iloc[:, j*3:(j+1)*3].to_numpy())
            d.require(np.array_equal(expected, processed.iloc[:, j*3:(j+1)*3].to_numpy()),
                      "Forecast postprocessing differs from the fixed recipe.")
        result[name] = dict(raw={col: float(raw.iloc[0][col]) for col in names},
                           processed={col: float(processed.iloc[0][col]) for col in names})
    return result


def completed_forecast(root, row, models, bindings):
    run = Path(root) / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]
    directory = run / "forecast_freeze-v1"
    if not directory.exists():
        return None
    d.require((directory / "freeze.json").is_file(), "Incomplete forecast checkpoint; preserve artifacts and inspect.")
    state, state_hash = d.checked_json(directory, "freeze")
    _, decision_hash, envelope, evidence = verify_decision(root, row)
    d.require(state["scenario"] == row and state["decision_freeze_sha256"] == decision_hash
              and state["model_bindings"] == bindings
              and state["forecasts_frozen_before_future"] is True
              and state["future_evaluation_executed"] is False
              and state["N2_calibrated"] is False and state["production_release_authorized"] is False,
              "Forecast checkpoint provenance/state differs.")
    payload = json.loads(d.checked(directory / "features.json", state["features_sha256"]))
    x = make_feature_frame(dict(zip(payload["columns"], payload["values"])), payload["columns"], row["scenario_id"])
    d.require(payload["scenario_id"] == row["scenario_id"] and payload["columns"] == models[PRIMARY].features,
              "Feature checkpoint identity/order differs.")
    from _scenario_meta import feature_row
    expected = make_feature_frame(feature_row(envelope["upstream"], evidence), models[PRIMARY].features, row["scenario_id"])
    d.require(np.array_equal(x.to_numpy(), expected.to_numpy()), "Features do not match bound decision-time evidence.")
    stored = json.loads(d.checked(directory / "forecasts.json", state["forecasts_sha256"]))
    d.require(stored == forecast_values(models, x), "Saved forecasts do not reproduce from bound models/features.")
    return dict(scenario_id=row["scenario_id"], group_id=row["group_id"], split_role=row["split_role"],
        decision_freeze_sha256=decision_hash, forecast_freeze_sha256=state_hash,
        feature_sha256=state["features_sha256"], forecasts_sha256=state["forecasts_sha256"])


def freeze_forecasts(root, row, models, bindings):
    _, decision_hash, envelope, evidence = verify_decision(root, row)
    from _scenario_meta import feature_row
    values = feature_row(envelope["upstream"], evidence)
    x = make_feature_frame(values, models[PRIMARY].features, row["scenario_id"])
    features = e.encoded(dict(scenario_id=row["scenario_id"], columns=list(x), values=x.iloc[0].tolist(),
        basis="pre_release_diagnosed_populations_only_no_future_targets"))
    forecasts = e.encoded(forecast_values(models, x))
    # Verify source links again immediately before creating the checkpoint.
    verify_decision(root, row)
    directory = Path(root) / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"] / "forecast_freeze-v1"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "features.json").write_bytes(features)
    (directory / "forecasts.json").write_bytes(forecasts)
    marker = dict(purpose="decision_time_features_and_four_method_raw_forecasts", scenario=row,
        decision_freeze_sha256=decision_hash, model_bindings=bindings,
        features_sha256=d.sha(features), forecasts_sha256=d.sha(forecasts),
        forecasts_frozen_before_future=True, future_evaluation_executed=False,
        N2_calibrated=False, interval_note="Raw N1 bands; no calibrated coverage or release authorization.",
        production_release_authorized=False)
    content = e.encoded(marker)
    (directory / "freeze.json").write_bytes(content)
    (directory / "freeze.sha256").write_text(d.sha(content)+"\n", encoding="utf-8")
    return completed_forecast(root, row, models, bindings)
