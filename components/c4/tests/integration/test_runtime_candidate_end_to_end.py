"""Actual-model integration with temporary synthetic data, not research evidence.

No upstream services or frozen benchmark artifacts are used. The preprocessing
record describes an identity transformation because fixture rows are generated
in the processed numeric feature space. This does not test raw-data translation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from c4.api.runtime_service import prepare_runtime_candidate
from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.models.champion import ChampionConfig, train_champion
from c4.models.runtime_artifacts import load_runtime_candidate


def test_actual_model_candidate_preparation_roundtrip(tmp_path):
    def put(name, contents):
        path = tmp_path / name
        path.write_bytes(contents)
        return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}

    def put_json(name, value):
        return put(name, (json.dumps(value, allow_nan=False) + "\n").encode())

    def frame(n, offset):
        ages = np.linspace(.02, .98, n)
        data = pd.DataFrame({column: np.zeros(n) for column in FEATURE_COLUMNS})
        data["age"] = ages
        data["hours_per_week"] = np.linspace(.1, .9, n)
        data[TARGET_COLUMN] = (ages > .5).astype(int)
        # Source identities are fixture declarations for disjoint synthetic roles.
        ids = [f"synthetic-fixture:{offset + i}" for i in range(n)]
        return data.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]], ids

    reference, reference_ids = frame(48, 0)
    repair, repair_ids = frame(12, 100)
    validation, validation_ids = frame(24, 200)
    config = ChampionConfig(n_estimators=4, max_depth=2, min_child_weight=1,
                            n_jobs=1, random_state=42)
    champion = train_champion(reference, config)
    model = put("champion.ubj", bytes(champion.get_booster().save_raw(raw_format="ubj")))
    ref = put("reference.csv", reference.to_csv(index=False).encode())
    repair_file = put("repair.csv", repair.to_csv(index=False).encode())
    release = put("release_validation.csv", validation.to_csv(index=False).encode())
    preprocessing = put_json("preprocessing.json", {
        "kind": "synthetic_processed_identity_fixture", "feature_columns": list(FEATURE_COLUMNS)})
    metadata = put_json("champion_metadata.json", {
        "role": "champion", "dataset": "uci_adult", "artifact_version": "fixture-v1",
        "feature_columns": list(FEATURE_COLUMNS), "target_column": TARGET_COLUMN,
        "model_sha256": model["sha256"], "reference_csv_sha256": ref["sha256"],
        "training_config": asdict(config)})
    # A distinct generator-source role is declared for this artificial package;
    # no claim about TVAE generation or privacy certification is made by this test.
    lineage = put_json("lineage.json", {
        "reference_csv_sha256": ref["sha256"],
        "release_validation_csv_sha256": release["sha256"],
        "repair_csv_sha256": repair_file["sha256"],
        "preprocessing_sha256": preprocessing["sha256"],
        "reference_row_ids": reference_ids,
        "release_validation_row_ids": validation_ids,
        "repair_source_row_ids": [f"synthetic-generator-source:{i}" for i in range(12)],
        "repair_source_role": "dedicated_repair",
        "repair_excluded_release_validation": True, "repair_excluded_future": True})
    binding = {"schema_version": "1.0", "bindings": {
        "dataset": "uci", "champion_model_id": "synthetic-fixture-champion",
        "champion_model": model, "champion_metadata": metadata,
        "preprocessing": preprocessing, "reference": ref, "release_validation": release,
        "repair": repair_file, "lineage": lineage,
        "feature_columns": list(FEATURE_COLUMNS), "target_column": TARGET_COLUMN,
        "rule_feature_space": "processed"}}
    binding_file = put_json("runtime.json", binding)["path"]
    upstream = {
        "context": {"dataset": "uci", "model_id": "synthetic-fixture-champion",
                    "current_dataset_uri": "synthetic-monitoring.csv", "run_id": "fixture-run-1"},
        "c1": {
            "evidence_version": "fixture-c1", "engine_version": "fixture",
            "model_id": "synthetic-fixture-champion", "reference_dataset_uri": ref["path"],
            "current_dataset_uri": "synthetic-monitoring.csv", "batch_size": 24,
            "drift_detected": True, "predicted_harm": .1, "harm_interval": [0, .2],
            "harm_metric": "accuracy_drop", "harm_severity": "high", "marginal_extent": "single",
            "interaction_suspected": .1, "affected_features": ["age"],
            "detector_scores": {**dict.fromkeys(["ks_mean", "ks_max", "chi2_mean", "chi2_max",
                "psi_mean", "psi_max", "mmd", "lsdd"], .1), "clf_auc": .6},
            "detectors_fired": 1, "detectors_fired_list": ["ks"],
            "prediction_stats": {"conf_mean": .8, "conf_std": .1, "conf_pct_lt_06": .1,
                "mean_pos_prob": .4, "pos_prob_std": .1, "pct_pos_prob_lt_06": .6},
            "warnings": [], "runtime_sec": .1},
        "c2_to_c4": {"root_cause_category": "marginal", "root_cause_feature": "age",
            "affected_slice_rule": "age > 0.5", "slice_support": .5, "diagnosis_stability": .9,
            "affected_features_ranked": ["age"], "diagnosis_version": "fixture-c2",
            "diagnosis_timestamp": "2026-10-18T00:00:00Z"},
        "c3": {"repair_dataset_uri": repair_file["path"], "repair_package_status": "APPROVED",
            "generator": "TVAE", "n_synthetic_rows": 12, "augmentation_ratio": .25,
            "fidelity_score": .8, "utility_score": .01, "privacy_status": "PASS",
            "target_coverage": .5, "prescription_confidence": .8, "repair_version": "fixture-c3",
            "repair_timestamp": "2026-10-18T00:01:00Z"}}
    upstream_file = put_json("upstream.json", upstream)["path"]
    before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    plan = prepare_runtime_candidate(config_path=binding_file, upstream=upstream)
    assert plan["status"] == "PLAN_ONLY" and not plan["candidate_training_executed"]
    assert not (tmp_path / "c4_artifacts").exists()
    out = prepare_runtime_candidate(config_path=binding_file, upstream=upstream,
        execute_training=True, data_root=tmp_path, version="actual-model-fixture-v1")
    assert out["status"] == "CANDIDATE_SAVED_RELEASE_PENDING"
    assert out["release_decision"] is None and not out["production_release_authorized"]
    assert not out["forecast_applicable"] and not out["N1_executed"] and not out["N2_executed"]
    candidate, manifest = load_runtime_candidate(out["artifact_directory"],
        expected_manifest_sha256=out["manifest_sha256"])
    assert manifest["artifact_persistence_completed"]
    assert candidate.report["training_rows"] == 60
    assert candidate.report["validation_rows"] == 24
    assert candidate.report["candidate_reload_verified"]
    assert candidate.report["pre_release_evaluation"]["comparisons"]["affected_slice"]["rows"] == 12
    assert all(path.read_bytes() == contents for path, contents in before.items())
    with pytest.raises(FileExistsError):
        prepare_runtime_candidate(config_path=binding_file, upstream=upstream,
            execute_training=True, data_root=tmp_path, version="actual-model-fixture-v1")
    # Exercise the actual CLI without executing a second training run.
    script = Path(__file__).resolve().parents[2] / "scripts/run_runtime_candidate.py"
    cli = subprocess.run([sys.executable, str(script), "--config", binding_file,
        "--upstream-json", upstream_file], capture_output=True, text=True, timeout=60)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert json.loads(cli.stdout)["status"] == "PLAN_ONLY"
