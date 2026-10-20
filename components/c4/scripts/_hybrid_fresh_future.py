"""Fresh calibration outcomes, strictly after the complete forecast batch freeze.

The external row pool is shared with development. These are new scenario groups,
not a new independent source dataset. No fitting or release is performed here.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import _hybrid_fresh_forecasts as q

d, e, f, v = q.d, q.e, q.f, q.v
C4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C4 / "src"))
from c4.data import FEATURE_COLUMNS
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule
from c4.scenarios.adapter import _load_shared_functions
from c4.scenarios.feature_metadata import load_adult_feature_metadata
from _all_mechanism_transforms import transform_future_role
from _portable_model_io import load_champion, load_candidate

COHORT_RUN = "adult-external-development-v3-01"
SCOPE = "fresh_conformal_scenarios_on_shared_external_development_cohort"


def run_at(root, row):
    return Path(root) / "c4_artifacts/benchmark_runs/adult" / v.VERSION / row["scenario_id"]


def verify_inputs(pins):
    for path, checksum in pins.items():
        d.checked(Path(path), checksum)


def load_models(bindings):
    d.require(set(bindings) == {q.PRIMARY, *q.b.METHODS}, "Incomplete four-method inventory.")
    models = {}
    for name, binding in bindings.items():
        directory = Path(binding["directory"])
        if name == q.PRIMARY:
            meta, digest = d.checked_json(directory)
            d.require(digest == binding["metadata_sha256"], "Hybrid binding differs.")
            for relative, checksum in meta["files"].items():
                d.checked(directory / relative, checksum)
            models[name] = q.b.h.HybridForecaster.load(directory)
        else:
            _, digest = d.checked_json(directory, "model")
            d.require(digest == binding["model_sha256"], "Baseline binding differs.")
            models[name] = q.b.FixedBaseline.load(directory)
            d.require(models[name].kind == q.b.METHODS[name], "Baseline family differs.")
    return models


def verify_forecast_batch(root, rows, claim, batch, models):
    d.require(claim["environment"] == q.c.environment(), "Forecast environment changed.")
    d.require(claim["scenario_ids"] == [r["scenario_id"] for r in rows]
              and claim["future_evaluation_authorized"] is False
              and claim["model_refitting_authorized"] is False
              and claim["production_release_authorized"] is False,
              "Forecast claim scope/inventory differs.")
    d.require(len(rows) > 0 and len({r["scenario_id"] for r in rows}) == len(rows)
              and all(r["split_role"] == "conformal_calibration" for r in rows),
              "Only unique admitted calibration scenarios are allowed.")
    d.require(batch["cases"] == len(rows) and batch["model_bindings"] == claim["model_bindings"]
              and batch["forecasts_frozen_before_future"] is True
              and batch["future_evaluation_executed"] is False
              and batch["N2_calibrated"] is False
              and batch["production_release_authorized"] is False,
              "Complete pre-future forecast batch is required.")
    for key in ("assigned_groups", "nonempty_admission_groups", "blocked_cases_retained"):
        d.require(batch[key] == claim[key], "Forecast batch group/admission totals differ.")
    verify_inputs(claim["inputs"])
    records = [q.completed_forecast(root, row, models, claim["model_bindings"]) for row in rows]
    d.require(all(record is not None for record in records) and records == batch["records"],
              "Full forecast completion does not match verified artifacts.")
    return records


def prepare(root, c4=C4):
    """Validate all frozen evidence before parsing external observations."""
    root, c4 = Path(root), Path(c4)
    # Do not call q.prepare: it intentionally refuses any existing future output.
    # The original immutable pre-future batch is checked directly on every resume.
    candidate_claim, rows = q.c.prepare(root, c4)
    saved, _ = d.checked_json(f.bundle_at(root) / "candidate-executor-v1", "claim")
    d.require(e.encoded(saved) == e.encoded(candidate_claim), "Candidate execution binding changed.")
    forecast_dir = f.bundle_at(root) / "forecast-freezer-v1"
    forecast_claim, claim_hash = d.checked_json(forecast_dir, "claim")
    batch, batch_hash = d.checked_json(forecast_dir / "completion-v1", "forecast_batch")
    models = load_models(forecast_claim["model_bindings"])
    records = verify_forecast_batch(root, rows, forecast_claim, batch, models)
    pins = dict(forecast_claim["inputs"])
    pins[str(forecast_dir / "claim.json")] = claim_hash
    pins[str(forecast_dir / "completion-v1/forecast_batch.json")] = batch_hash
    candidate_models = {}
    for row, record in zip(rows, records):
        run = run_at(root, row)
        pins[str(run / "pre_release_freeze-v1/freeze.json")] = record["decision_freeze_sha256"]
        pins[str(run / "forecast_freeze-v1/freeze.json")] = record["forecast_freeze_sha256"]
        pins[str(run / "forecast_freeze-v1/features.json")] = record["feature_sha256"]
        pins[str(run / "forecast_freeze-v1/forecasts.json")] = record["forecasts_sha256"]
        candidate_meta, candidate_hash = d.checked_json(run / "scoped_candidate_validation-v1")
        candidate_models[row["scenario_id"]] = candidate_meta["model_sha256"]
        pins[str(run / "scoped_candidate_validation-v1/metadata.json")] = candidate_hash
        pins[str(run / "scoped_candidate_validation-v1/candidate.ubj")] = candidate_meta["model_sha256"]
    protocol_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    frozen, freeze_hash = d.checked_json(protocol_dir, "freeze")
    protocol = json.loads(d.checked(protocol_dir / "protocol.json", frozen["protocol_sha256"]))
    pins[str(protocol_dir / "freeze.json")] = freeze_hash
    pins[str(protocol_dir / "protocol.json")] = frozen["protocol_sha256"]
    for record in records:
        snapshot, _ = d.checked_json(run_at(root, {"scenario_id": record["scenario_id"]}) / "pre_release_freeze-v1", "freeze")
        d.require(snapshot["protocol_freeze_sha256"] == freeze_hash, "Decision/protocol link differs.")
    # The external cohort was fixed during development, before fresh admissions.
    development = root / "c4_artifacts/forecasting_experiments/adult" / COHORT_RUN
    recovery = development / "portable-resume-v1"
    meta, meta_hash = d.checked_json(recovery)
    d.require(pins.get(str(recovery / "metadata.json")) == meta_hash,
              "External-development metadata is not bound by the pre-future forecast claim.")
    recovery_claim, recovery_hash = d.checked_json(recovery, "recovery_claim")
    d.require(meta["recovery_claim_sha256"] == recovery_hash
              and meta["files"]["recovery_claim.json"] == recovery_hash
              and meta["calibration_policy_outer_cases_accessed"] is False
              and meta["cohort"]["eligible_rows"] == 12899
              and meta["cohort"]["feature_only_old_overlap_exclusion"] is True,
              "External-cohort provenance differs.")
    original_spec, original_spec_hash = d.checked_json(development, "specification")
    d.require(meta["original_specification_sha256"] == recovery_claim["original_specification_sha256"] == original_spec_hash
              and original_spec["version"] == "adult-external-development-v3",
              "Original preprocessing/cohort experiment binding differs.")
    verify_inputs(original_spec["inputs"])
    pins.update(original_spec["inputs"])
    pins[str(development / "specification.json")] = original_spec_hash
    from run_n1_fast_improvement import EXTERNAL_SHA
    pins[str(development / "results/adult.test")] = EXTERNAL_SHA
    cohort_path = development / "results/external_clean.csv"
    pins[str(recovery / "recovery_claim.json")] = recovery_hash
    pins[str(cohort_path)] = recovery_claim["cohort_csv_sha256"]
    # Hashing pinned bytes is not target evaluation; parsing occurs only in execute.
    d.checked(cohort_path, pins[str(cohort_path)])
    shared, drift_hash = _load_shared_functions()
    feature_metadata = load_adult_feature_metadata()
    repository = c4.parent.parent
    drift_path = repository / "shared/injection/drift_types.py"
    d.checked(drift_path, drift_hash)
    pins[str(drift_path)] = drift_hash
    pins[str(feature_metadata.source_path)] = feature_metadata.source_sha256
    # Match the code/types actually used for each admitted release preview.
    for row in rows:
        admission, _ = d.checked_json(run_at(root, row), "admission")
        preview_dir = root / "c4_artifacts/scenario_previews/adult" / admission["preview_id"] / "release_validation"
        preview, preview_hash = d.checked_json(preview_dir)
        transform = preview["transformation"]
        d.require(transform["shared_drift_types_sha256"] == (None if row["mechanism"] == "none" else drift_hash)
                  and transform["feature_metadata_sha256"] == feature_metadata.source_sha256,
                  "Future drift implementation/types differ from frozen admission previews.")
        pins[str(preview_dir / "metadata.json")] = preview_hash
    champion_dir = root / "c4_artifacts/champion/adult" / protocol["source_versions"]["champion"]
    # Champion v1 predates JSON sidecars. Its exact metadata bytes are already
    # bound in every decision freeze; do not require a nonexistent new sidecar.
    champion_meta_path = champion_dir / "metadata.json"
    first_snapshot, _ = d.checked_json(run_at(root, rows[0]) / "pre_release_freeze-v1", "freeze")
    champion_meta_hash = first_snapshot["dependencies"].get(str(champion_meta_path))
    d.require(bool(champion_meta_hash), "Champion metadata missing from the decision dependency manifest.")
    champion_meta = json.loads(d.checked(champion_meta_path, champion_meta_hash))
    d.require(champion_meta["model_sha256"] == frozen["champion_model_sha256"], "Champion binding differs.")
    pins[str(champion_dir / "metadata.json")] = champion_meta_hash
    pins[str(champion_dir / "champion.ubj")] = frozen["champion_model_sha256"]
    for relative in ("scripts/_hybrid_fresh_future.py", "scripts/run_hybrid_fresh_future.py",
                     "scripts/_all_mechanism_transforms.py", "scripts/_portable_model_io.py",
                     "src/c4/scenarios/config.py", "src/c4/scenarios/slices.py",
                     "src/c4/scenarios/adapter.py", "src/c4/scenarios/feature_metadata.py"):
        path = c4 / relative
        pins[str(path)] = d.sha(path.read_bytes())
    claim = dict(purpose="fresh_conformal_external_future_evaluation_v1", scope=SCOPE,
        environment=q.c.environment(), inputs=pins, scenario_ids=[r["scenario_id"] for r in rows],
        forecast_batch_sha256=batch_hash, forecast_records=records, model_bindings=forecast_claim["model_bindings"],
        candidate_models=candidate_models,
        cohort_path=str(cohort_path), cohort_csv_sha256=pins[str(cohort_path)], expected_cohort_rows=12899,
        champion_directory=str(champion_dir), champion_model_sha256=frozen["champion_model_sha256"],
        protocol_freeze_sha256=freeze_hash, true_slice=protocol["true_slice"],
        shared_drift_types_sha256=drift_hash, feature_metadata_sha256=feature_metadata.source_sha256,
        assigned_groups=batch["assigned_groups"], nonempty_admission_groups=batch["nonempty_admission_groups"],
        blocked_cases_retained=batch["blocked_cases_retained"],
        target_columns=list(d.TARGETS), target_units="accuracy_margin_fraction",
        primary_slice_basis="true_injected_slice", target_sign="candidate_minus_champion",
        weighting="equal total weight per nonempty scenario group",
        future_evaluation_authorized=True, candidate_training_authorized=False,
        generator_refitting_authorized=False, N1_refitting_authorized=False,
        N2_fitting_authorized=False, policy_fitting_authorized=False, production_release_authorized=False,
        limitation="New scenario groups on the same external row pool used for development; not independent dataset validation. Calibration metrics are not final held-out results.")
    verify_inputs(pins)
    return claim, rows, shared, feature_metadata.feature_types


def load_cohort(claim):
    data = d.checked(Path(claim["cohort_path"]), claim["cohort_csv_sha256"])
    clean = pd.read_csv(io.BytesIO(data), index_col="source_row_id", float_precision="round_trip")
    d.require(list(clean) == [*FEATURE_COLUMNS, "target"] and len(clean) == claim["expected_cohort_rows"],
              "External cohort schema/count differs.")
    d.require(clean.index.is_unique and pd.api.types.is_integer_dtype(clean.index.dtype)
              and (clean.index >= 1_000_000).all() and clean.index.is_monotonic_increasing,
              "External cohort source identities differ.")
    d.require(all(pd.api.types.is_numeric_dtype(clean[col]) for col in clean)
              and np.isfinite(clean.to_numpy(float)).all() and set(clean.target.unique()) == {0, 1},
              "Invalid external processed observations/labels.")
    return clean


def configuration(row):
    return ScenarioConfig(mechanism=row["mechanism"], magnitude=row["magnitude"],
        features=tuple(row["features"]), interaction_pairs=tuple(tuple(pair) for pair in row["interaction_pairs"]),
        seed=row["drift_seeds"]["sealed_future"])


def true_rule(claim):
    rule = claim["true_slice"]
    return SliceRule(name=rule["name"], feature=rule["feature"], operator=rule["operator"], value=rule["resolved_threshold"])


def paired_results(predictions):
    expected = ["source_row_id", "target", "true_slice_member", "champion_prediction", "candidate_prediction"]
    d.require(list(predictions) == expected and len(predictions) > 0, "Invalid paired prediction schema.")
    d.require(predictions.source_row_id.is_unique and pd.api.types.is_integer_dtype(predictions.source_row_id.dtype),
              "Invalid paired source IDs.")
    for col in expected[1:]:
        d.require(predictions[col].isin([0, 1]).all(), "Predictions, labels and memberships must be binary.")
    mask = predictions.true_slice_member.to_numpy(bool)
    d.require(mask.any() and (~mask).any(), "Both target subpopulations require rows.")
    y = predictions.target.to_numpy()
    before = predictions.champion_prediction.to_numpy() == y
    after = predictions.candidate_prediction.to_numpy() == y
    diff = after.astype(int)-before.astype(int)
    comparisons, targets = {}, {}
    for target, label, selected in zip(d.TARGETS, ("global", "true_injected_slice", "true_slice_complement"),
                                     (np.ones(len(y), bool), mask, ~mask)):
        targets[target] = float(diff[selected].mean())
        comparisons[label] = dict(rows=int(selected.sum()), champion_accuracy=float(before[selected].mean()),
            candidate_accuracy=float(after[selected].mean()), candidate_minus_champion=targets[target],
            candidate_only_correct=int((diff[selected] == 1).sum()),
            champion_only_correct=int((diff[selected] == -1).sum()))
    return targets, comparisons


def source_record(claim, row):
    matches = [r for r in claim["forecast_records"] if r["scenario_id"] == row["scenario_id"]]
    d.require(len(matches) == 1 and matches[0]["group_id"] == row["group_id"], "Missing forecast identity.")
    return matches[0]


def completed_case(root, row, claim, claim_hash, clean):
    directory = run_at(root, row) / "external_future-v1"
    if not directory.exists():
        return None
    d.require((directory / "outcomes.json").is_file(), "Incomplete future checkpoint; preserve it and inspect before resuming.")
    outcome, digest = d.checked_json(directory, "outcomes")
    source = source_record(claim, row)
    d.require(outcome["scenario"] == row and outcome["scope"] == SCOPE
              and outcome["evaluator_claim_sha256"] == claim_hash
              and outcome["forecast_record"] == source
              and outcome["cohort_csv_sha256"] == claim["cohort_csv_sha256"]
              and outcome["champion_model_sha256"] == claim["champion_model_sha256"]
              and outcome["shared_drift_types_sha256"] == claim["shared_drift_types_sha256"]
              and outcome["feature_metadata_sha256"] == claim["feature_metadata_sha256"]
              and outcome["configuration"] == configuration(row).model_dump(mode="json")
              and outcome["primary_slice_basis"] == "true_injected_slice"
              and outcome["target_units"] == "accuracy_margin_fraction"
              and outcome["target_sign"] == "candidate_minus_champion"
              and outcome["future_evaluation_executed"] is True
              and outcome["training_executed"] is False
              and outcome["controller_received_future_targets"] is False
              and outcome["N2_calibrated"] is False
              and outcome["production_release_authorized"] is False,
              "Future checkpoint provenance/state differs.")
    meta, _ = d.checked_json(run_at(root, row) / "scoped_candidate_validation-v1")
    d.require(outcome["candidate_model_sha256"] == meta["model_sha256"] == claim["candidate_models"][row["scenario_id"]],
              "Candidate model link differs.")
    d.require(set(outcome["files"]) == {"paired_predictions.csv"}, "Unexpected/missing future output files.")
    content = d.checked(directory / "paired_predictions.csv", outcome["files"]["paired_predictions.csv"])
    predictions = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
    d.require(predictions.source_row_id.tolist() == clean.index.tolist()
              and np.array_equal(predictions.target.to_numpy(), clean.target.to_numpy())
              and np.array_equal(predictions.true_slice_member.to_numpy(bool), true_rule(claim).mask(clean).to_numpy()),
              "Paired rows/labels/population membership differ from the bound cohort.")
    targets, comparisons = paired_results(predictions)
    d.require(targets == outcome["primary_targets"] and comparisons == outcome["comparisons"],
              "Saved targets do not reproduce from saved paired predictions.")
    return dict(scenario_id=row["scenario_id"], group_id=row["group_id"], split_role=row["split_role"],
        outcomes_sha256=digest, forecast_freeze_sha256=source["forecast_freeze_sha256"], primary_targets=targets)


def evaluate_case(root, row, claim, claim_hash, clean, champion, shared, feature_types):
    d.require(row["split_role"] == "conformal_calibration" and row["scenario_id"] in claim["scenario_ids"],
              "Future evaluation outside the calibration allowlist.")
    source = source_record(claim, row)
    run = run_at(root, row)
    d.checked(run / "forecast_freeze-v1/freeze.json", source["forecast_freeze_sha256"])
    d.checked(run / "forecast_freeze-v1/forecasts.json", source["forecasts_sha256"])
    candidate, metadata = load_candidate(run / "scoped_candidate_validation-v1")
    d.require(metadata["champion_model_sha256"] == claim["champion_model_sha256"]
              and metadata["model_sha256"] == claim["candidate_models"][row["scenario_id"]], "Candidate/champion differ.")
    config = configuration(row)
    transformed, mask = transform_future_role(clean, config=config, true_rule=true_rule(claim),
        shared=shared, feature_types=feature_types)
    d.require(np.array_equal(mask.to_numpy(), true_rule(claim).mask(clean).to_numpy()), "Future membership changed.")
    x = transformed.loc[:, list(FEATURE_COLUMNS)]
    before, after = np.asarray(champion.predict(x)), np.asarray(candidate.predict(x))
    d.require(before.shape == after.shape == (len(clean),), "Native prediction row count differs.")
    predictions = pd.DataFrame(dict(source_row_id=clean.index.to_numpy(), target=transformed.target.to_numpy(),
        true_slice_member=mask.to_numpy(dtype=int), champion_prediction=before, candidate_prediction=after))
    targets, comparisons = paired_results(predictions)
    content = predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    outcome = dict(purpose="fresh_calibration_external_future_targets", scope=SCOPE, scenario=row,
        evaluator_claim_sha256=claim_hash, forecast_record=source, cohort_csv_sha256=claim["cohort_csv_sha256"],
        configuration=config.model_dump(mode="json"), shared_drift_types_sha256=claim["shared_drift_types_sha256"],
        feature_metadata_sha256=claim["feature_metadata_sha256"], champion_model_sha256=claim["champion_model_sha256"],
        candidate_model_sha256=metadata["model_sha256"], primary_slice_basis="true_injected_slice",
        membership_basis="clean_before_future_injection", target_units="accuracy_margin_fraction",
        target_sign="candidate_minus_champion", primary_targets=targets, comparisons=comparisons,
        files={"paired_predictions.csv": d.sha(content)}, future_evaluation_executed=True,
        training_executed=False, controller_received_future_targets=False, N2_calibrated=False,
        production_release_authorized=False, limitation=claim["limitation"])
    directory = run / "external_future-v1"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "paired_predictions.csv").write_bytes(content)
    document = e.encoded(outcome)
    (directory / "outcomes.json").write_bytes(document)
    (directory / "outcomes.sha256").write_text(d.sha(document)+"\n", encoding="utf-8")
    return completed_case(root, row, claim, claim_hash, clean)


def metric_tables(root, rows, records, claim):
    d.require([r["scenario_id"] for r in records] == [r["scenario_id"] for r in rows], "Metric scenario alignment differs.")
    index = pd.DataFrame([{k: row[k] for k in ("scenario_id", "group_id", "split_role")} for row in rows])
    targets = pd.DataFrame([dict(scenario_id=r["scenario_id"], **r["primary_targets"]) for r in records]).set_index("scenario_id")
    forecasts = {}
    for row in rows:
        source = source_record(claim, row)
        forecasts[row["scenario_id"]] = json.loads(d.checked(run_at(root, row) / "forecast_freeze-v1/forecasts.json", source["forecasts_sha256"]))
    summaries = []
    for method in sorted(claim["model_bindings"]):
        for target in d.TARGETS:
            pred = np.asarray([[forecasts[row["scenario_id"]][method]["processed"][target+"__"+label]
                                for _, label in d.QUANTILES] for row in rows], float)
            d.require(np.isfinite(pred).all() and (np.diff(pred, axis=1) >= 0).all() and (np.abs(pred) <= 1).all(),
                      "Invalid frozen forecast bands in metric aggregation.")
            metric = e.losses(targets[target], pred, index.group_id)
            summaries.append(dict(method=method, target=target, **metric,
                MAE_pp=metric["mae"]*100, RMSE_pp=metric["rmse"]*100, Pinball_pp=metric["mean_pinball"]*100))
    return index, targets, pd.DataFrame(summaries)
