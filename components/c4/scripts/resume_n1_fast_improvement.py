"""Resume the frozen larger-cohort experiment with verified native byte loading.

Original runner, scientific specification, source models and completed legacy
checkpoints remain unchanged. An engineering-recovery claim pins the old inputs,
legacy checkpoints and this adapter before evaluation continues. No Git commands.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import run_n1_fast_improvement as h

d, e = h.d, h.e


class VerifiedBytes:
    """Reuse exactly hash-verified bytes; recheck all unique source files at end."""
    def __init__(self):
        self.contents = {}

    def __call__(self, path, expected):
        key = (str(Path(path)), expected)
        if key not in self.contents:
            try:
                self.contents[key] = d.checked(Path(path), expected)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Python cannot read this source file. Check Google Drive availability: {path}") from exc
        return self.contents[key]

    def recheck(self):
        for (path, digest) in self.contents:
            d.checked(Path(path), digest)


def model_bytes(directory, role, expected_model_hash, reader):
    directory = Path(directory)
    metadata_bytes = (directory / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    d.require(metadata.get("role") == role and metadata.get("feature_columns") == h.FEATURES,
              "Model role/feature metadata differs.")
    d.require(metadata.get("model_sha256") == expected_model_hash, "Model metadata binding differs.")
    contents = reader(directory / (role + ".ubj"), expected_model_hash)
    return contents, metadata


def native_from_bytes(contents, factory=None):
    if factory is None:
        from xgboost import XGBClassifier
        factory = XGBClassifier
    model = factory()
    model.load_model(bytearray(contents))
    d.require(model.get_booster().feature_names == h.FEATURES, "Loaded native model schema differs.")
    return model


def validate_checkpoint(checkpoint, *, sid, group, freeze_hash, outcome_hash,
                        model_hash, population_rows):
    expected = dict(scenario_id=sid, group_id=group, decision_freeze_sha256=freeze_hash,
                    original_outcomes_sha256=outcome_hash, candidate_model_sha256=model_hash,
                    external_file_sha256=h.EXTERNAL_SHA, population_rows=population_rows,
                    scope="forecast_train_external_development_only")
    d.require(all(checkpoint.get(k) == v for k, v in expected.items()), "Checkpoint provenance differs.")
    values = checkpoint.get("primary_targets", {})
    d.require(set(values) == set(d.TARGETS), "Checkpoint target schema differs.")
    for target, count in zip(d.TARGETS, population_rows):
        value = values[target]
        d.require(type(value) in (int, float) and np.isfinite(value) and abs(value) <= 1,
                  "Invalid checkpoint target.")
        d.require(abs(value * count - round(value * count)) < 1e-7,
                  "Checkpoint target is not a paired accuracy margin.")
    overall = (population_rows[1] * values[d.TARGETS[1]] +
               population_rows[2] * values[d.TARGETS[2]]) / population_rows[0]
    d.require(abs(overall - values[d.TARGETS[0]]) < 1e-14,
              "Checkpoint population aggregation differs.")
    return values


def save_new_checkpoint(path, record):
    contents = e.encoded(record)
    d.require(not path.exists(), "Recovery checkpoint already exists; do not overwrite.")
    path.write_bytes(contents)
    path.with_suffix(".sha256").write_text(d.sha(contents) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-id", default="adult-external-development-v3-01")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    d.require(args.resume, "Pass --resume to continue the existing frozen experiment.")
    value = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(value), "Set C4_DATA_ROOT.")
    root = Path(value)
    d.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id) is not None, "Unsafe run ID.")
    c4 = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(c4 / "src"))
    old_run = root / "c4_artifacts/forecasting_experiments/adult" / args.run_id
    original_spec, original_hash = d.checked_json(old_run, "specification")
    d.require(original_spec["version"] == "adult-external-development-v3"
              and original_spec["environment"] == h.environment(), "Frozen scope/environment differs.")
    reader = VerifiedBytes()
    for path, digest in original_spec["inputs"].items():
        reader(Path(path), digest)
    d.require(original_spec["inputs"].get(str(Path(h.__file__).resolve())) == d.sha(Path(h.__file__).read_bytes()),
              "Original runner is not bound by the frozen specification. Restore it; do not overwrite it.")
    x, old_targets, index, _, _, _ = d.load_inputs(root, c4)
    legacy = old_run / "results"
    external_bytes = reader(legacy / "adult.test", h.EXTERNAL_SHA)
    raw_path = next(Path(p) for p, digest in original_spec["inputs"].items() if digest == h.RAW_SHA)
    clean, cohort_info = h.prepare_cohort(h.raw_table(reader(raw_path, h.RAW_SHA)), h.raw_table(external_bytes),
        pd.read_csv(root / "uci_adult_data/reference.csv"), pd.read_csv(root / "uci_adult_data/test.csv"))
    clean_bytes = clean.to_csv(index_label="source_row_id").encode()
    d.require((legacy / "external_clean.csv").read_bytes() == clean_bytes,
              "Reconstructed cohort differs from the original partial run.")
    legacy_files = sorted((legacy / "cases").glob("*.json"))
    allowed = set(index.scenario_id)
    d.require(all(p.stem in allowed for p in legacy_files), "Unexpected legacy checkpoint.")
    legacy_hashes = {str(p): d.sha(p.read_bytes()) for p in legacy_files}
    recovery = old_run / "portable-resume-v1"
    claim = {"purpose": "engineering_recovery_only_scientific_recipe_unchanged",
             "original_specification_sha256": original_hash,
             "adapter_sha256": d.sha(Path(__file__).read_bytes()),
             "environment": h.environment(), "cohort_csv_sha256": d.sha(clean_bytes),
             "legacy_checkpoints": legacy_hashes,
             "legacy_checkpoint_note": "Legacy files lacked sidecars; their byte hashes are first pinned by this recovery claim. Per-case source bindings, counts and accuracy-margin identities are checked before reuse.",
             "changes": ["Verified bytearray native model loading instead of long filenames.",
                         "Reuse validated checkpoints; checksum new recovery checkpoints.",
                         "Cache frozen verified bytes within execution; recheck unique source files at end."],
             "scientific_parameters_changed": False, "production_release_authorized": False}
    claim_bytes = e.encoded(claim)
    if recovery.exists():
        existing, _ = d.checked_json(recovery, "recovery_claim")
        d.require(e.encoded(existing) == claim_bytes, "Recovery code/inputs changed; preserve this recovery directory.")
    else:
        recovery.mkdir()
        (recovery / "recovery_claim.json").write_bytes(claim_bytes)
        (recovery / "recovery_claim.sha256").write_text(d.sha(claim_bytes) + "\n", encoding="utf-8")
        (recovery / "cases").mkdir()
    if (recovery / "metadata.json").exists():
        meta, _ = d.checked_json(recovery)
        for name, digest in meta["files"].items():
            d.checked(recovery / name, digest)
        print("RECOVERY ALREADY COMPLETE:", recovery)
        return
    with (recovery / "progress.log").open("a", encoding="utf-8") as log:
        def progress(message):
            line = datetime.now(timezone.utc).isoformat() + " " + message
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        progress(f"RESUME READY: {len(legacy_files)} original checkpoints found; old results preserved")
        import _future_research as future_helpers
        from _all_mechanism_transforms import transform_future_role
        from c4.scenarios.config import ScenarioConfig
        from c4.scenarios.slices import SliceRule
        from c4.scenarios.adapter import _load_shared_functions
        from c4.scenarios.feature_metadata import load_adult_feature_metadata
        protocol_dir = root / "c4_artifacts/benchmark_protocols/adult" / d.PROTOCOL
        frozen, protocol_hash = d.checked_json(protocol_dir, "freeze")
        protocol = json.loads(reader(protocol_dir / "protocol.json", frozen["protocol_sha256"]))
        rule = protocol["true_slice"]
        true_rule = SliceRule(name=rule["name"], feature=rule["feature"], operator=rule["operator"], value=rule["resolved_threshold"])
        true_mask = true_rule.mask(clean).to_numpy()
        masks = (np.ones(len(clean), bool), true_mask, ~true_mask)
        d.require(all(mask.any() for mask in masks), "Empty target population.")
        population_rows = [int(mask.sum()) for mask in masks]
        shared, drift_hash = _load_shared_functions()
        feature_metadata = load_adult_feature_metadata()
        champion_dir = root / "c4_artifacts/champion/adult" / protocol["source_versions"]["champion"]
        champion_data, champion_meta = model_bytes(champion_dir, "champion", frozen["champion_model_sha256"], reader)
        d.require(champion_meta["reference_csv_sha256"] == h.REFERENCE_SHA, "Champion reference differs.")
        champion = None
        target_rows, records = [], []
        old_checker = future_helpers.checked_bytes
        future_helpers.checked_bytes = reader
        try:
            for number, row in enumerate(index.itertuples(index=False), 1):
                sid = row.scenario_id
                run = root / "c4_artifacts/benchmark_runs/adult" / d.PROTOCOL / sid
                snapshot = json.loads(reader(run / "pre_release_freeze-v1/freeze.json", row.decision_freeze_sha256))
                scenario = snapshot["scenario"]
                d.require(scenario["scenario_id"] == sid and scenario["group_id"] == row.group_id
                          and scenario["split_role"] == "forecast_train"
                          and snapshot["protocol_freeze_sha256"] == protocol_hash, "Frozen scenario differs.")
                future_helpers.verify_freeze_dependencies(snapshot)
                old = json.loads(reader(run / "future_research-v1/outcomes.json", row.outcomes_sha256))
                d.require(old["scenario"]["scenario_id"] == sid and old["scenario"]["split_role"] == "forecast_train"
                          and old["decision_freeze_sha256"] == row.decision_freeze_sha256
                          and old["shared_drift_types_sha256"] == drift_hash
                          and old["feature_metadata_sha256"] == feature_metadata.source_sha256
                          and old["champion_model_sha256"] == champion_meta["model_sha256"], "Original outcome binding differs.")
                candidate_data, candidate_meta = model_bytes(run / "scoped_candidate_validation-v1", "candidate", old["candidate_model_sha256"], reader)
                d.require(candidate_meta["champion_model_sha256"] == champion_meta["model_sha256"], "Candidate/champion binding differs.")
                new_path = recovery / "cases" / (sid + ".json")
                old_path = legacy / "cases" / (sid + ".json")
                if new_path.exists():
                    contents = d.checked(new_path, new_path.with_suffix(".sha256").read_text().strip())
                    record = json.loads(contents)
                    action = "REUSED recovery checkpoint"
                elif str(old_path) in legacy_hashes:
                    record = json.loads(reader(old_path, legacy_hashes[str(old_path)]))
                    action = "REUSED original checkpoint"
                else:
                    if champion is None:
                        champion = native_from_bytes(champion_data)
                    candidate = native_from_bytes(candidate_data)
                    config = ScenarioConfig(mechanism=scenario["mechanism"], magnitude=scenario["magnitude"],
                        features=tuple(scenario["features"]), interaction_pairs=tuple(tuple(v) for v in scenario["interaction_pairs"]),
                        seed=scenario["drift_seeds"]["sealed_future"])
                    transformed, actual_mask = transform_future_role(clean, config=config, true_rule=true_rule,
                        shared=shared, feature_types=feature_metadata.feature_types)
                    d.require(np.array_equal(actual_mask.to_numpy(), true_mask), "Population membership changed.")
                    truth = transformed.target.to_numpy()
                    a, b = np.asarray(champion.predict(transformed[h.FEATURES])), np.asarray(candidate.predict(transformed[h.FEATURES]))
                    d.require(a.shape == b.shape == truth.shape and np.isin(a, [0, 1]).all()
                              and np.isin(b, [0, 1]).all(), "Invalid native predictions.")
                    paired = (b == truth).astype(float) - (a == truth).astype(float)
                    record = dict(scenario_id=sid, group_id=row.group_id,
                        decision_freeze_sha256=row.decision_freeze_sha256, original_outcomes_sha256=row.outcomes_sha256,
                        candidate_model_sha256=candidate_meta["model_sha256"], population_rows=population_rows,
                        primary_targets={t: float(paired[mask].mean()) for t, mask in zip(d.TARGETS, masks)},
                        external_file_sha256=h.EXTERNAL_SHA, scope="forecast_train_external_development_only")
                    action = "COMPUTED with verified native bytes"
                values = validate_checkpoint(record, sid=sid, group=row.group_id,
                    freeze_hash=row.decision_freeze_sha256, outcome_hash=row.outcomes_sha256,
                    model_hash=candidate_meta["model_sha256"], population_rows=population_rows)
                if not new_path.exists():
                    save_new_checkpoint(new_path, record)
                target_rows.append(dict(scenario_id=sid, **values))
                records.append(record)
                progress(f"{number}/106 {action}: {sid}")
        finally:
            future_helpers.checked_bytes = old_checker
        targets = pd.DataFrame(target_rows).set_index("scenario_id").loc[x.index, list(d.TARGETS)]
        d.validate_tables(x, targets, index)
        targets.to_csv(recovery / "targets.csv", index_label="scenario_id")
        (recovery / "case_bindings.json").write_bytes(e.encoded(records))
        old_summary, _, _ = h.compare(x, old_targets, index, progress)
        old_summary["cohort"] = "original_development"
        new_summary, fold_metrics, forecasts = h.compare(x, targets, index, progress)
        new_summary["cohort"] = "external_development"
        combined = pd.concat([old_summary, new_summary], ignore_index=True)
        combined.to_csv(recovery / "metric_summary.csv", index=False)
        fold_metrics.to_csv(recovery / "external_fold_metrics.csv", index=False)
        for method, forecast in forecasts.items():
            forecast.to_csv(recovery / (method + "_grouped_oof.csv"), index_label="scenario_id")
        progress("RECHECKING unique frozen source files")
        reader.recheck()
        d.require(_load_shared_functions()[1] == drift_hash, "Shared drift implementation changed.")
        progress("COMPLETE. Original runner/specification/results/models unchanged. No TVAE/candidate refits.")
    files = {p.relative_to(recovery).as_posix(): d.sha(p.read_bytes()) for p in sorted(recovery.rglob("*"))
             if p.is_file() and p.name not in ("metadata.json", "metadata.sha256")}
    metadata = e.encoded(dict(original_specification_sha256=original_hash, recovery_claim_sha256=d.sha(claim_bytes),
        files=files, cohort=cohort_info, reused_original_cases=len(legacy_files),
        model_loading="verified_bytearray", scientific_parameters_changed=False,
        calibration_policy_outer_cases_accessed=False, production_release_authorized=False,
        interpretation="Exploratory grouped development CV; not untouched outer evaluation."))
    (recovery / "metadata.json").write_bytes(metadata)
    (recovery / "metadata.sha256").write_text(d.sha(metadata) + "\n", encoding="utf-8")
    display = combined.copy()
    display["MAE_pp"], display["RMSE_pp"], display["Pinball_pp"] = display.mae*100, display.rmse*100, display.mean_pinball*100
    print(display[["cohort", "method", "target", "MAE_pp", "RMSE_pp", "r2", "Pinball_pp"]].to_string(index=False))
    print("Compare methods within each cohort. Development results only; original application unchanged.")
    print("OUTPUT:", recovery)
    print("METADATA SHA256:", d.sha(metadata))


if __name__ == "__main__":
    main()
