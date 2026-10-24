"""Checks for training-only diagnostics; no real research files are required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnose_n1_development.py"
spec = importlib.util.spec_from_file_location("n1_development_diagnostics_tested", SCRIPT)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture_tables():
    rng = np.random.default_rng(42)
    ids = [f"scenario-{i:03d}" for i in range(106)]
    index = pd.DataFrame({"scenario_id": ids, "group_id": [f"group-{i % 29:02d}" for i in range(106)],
                          "split_role": "forecast_train"})
    features = pd.DataFrame(rng.normal(size=(106, 75)), index=pd.Index(ids, name="scenario_id"),
                            columns=[f"feature_{i}" for i in range(75)])
    features.iloc[:, 52:] = 1.0
    targets = pd.DataFrame(rng.normal(0, 0.01, size=(106, 3)), index=features.index, columns=h.TARGETS)
    quantile, _, _ = h.constant_references(targets, index)
    forecast = quantile.copy()
    for target in h.TARGETS:
        for _, label in h.QUANTILES:
            forecast[target + "__raw_" + label] = forecast[target + "__" + label]
    return features, targets, index, {"lightgbm": forecast.copy(), "linear": forecast.copy()}


def test_constant_references_cannot_see_own_validation_labels():
    _, targets, index, _ = fixture_tables()
    quantile, mean, folds = h.constant_references(targets, index)
    ids = folds[0]["validation_ids"]
    changed = targets.copy()
    changed.loc[ids] += 0.5
    q2, m2, _ = h.constant_references(changed, index)
    pd.testing.assert_frame_equal(quantile.loc[ids], q2.loc[ids])
    pd.testing.assert_frame_equal(mean.loc[ids], m2.loc[ids])


def test_baselines_preserve_group_boundaries_and_cover_each_row_once():
    _, targets, index, _ = fixture_tables()
    quantile, mean, folds = h.constant_references(targets, index)
    valid_ids = []
    for fold in folds:
        assert not set(fold["training_group_ids"]) & set(fold["validation_group_ids"])
        valid_ids.extend(fold["validation_ids"])
    assert sorted(valid_ids) == sorted(index.scenario_id)
    assert not quantile.isna().any().any() and not mean.isna().any().any()


def test_weighted_quantile_uses_group_balancing():
    assert h.weighted_quantile([0, 10, 20], [8, 1, 1], 0.5) == 0
    assert h.weighted_quantile([0, 10, 20], [1, 1, 8], 0.5) == 20
    weights = h.group_weights(["a", "a", "a", "b"])
    assert np.isclose(weights[:3].sum(), weights[3])


def test_metrics_keep_negative_and_undefined_r2():
    bad = h.metrics([0, 1], [2, 2], [1, 1])
    assert bad["r2"] < 0
    constant = h.metrics([1, 1], [1, 1], [1, 1])
    assert constant["r2"] is None and constant["mae"] == 0
    perfect = h.metrics([0, 1], [0, 1], [1, 1])
    assert perfect["r2"] == 1 and perfect["rmse"] == 0


def test_nontraining_roles_and_target_leakage_rejected():
    features, targets, index, _ = fixture_tables()
    bad = index.copy()
    bad.loc[0, "split_role"] = "outer_test"
    with pytest.raises(ValueError, match="Non-training"):
        h.validate_tables(features, targets, bad)
    features = features.rename(columns={"feature_0": h.TARGETS[0]})
    with pytest.raises(ValueError, match="leakage"):
        h.validate_tables(features, targets, index)


def test_diagnostic_produces_baseline_fold_group_and_residual_outputs():
    features, targets, index, forecasts = fixture_tables()
    h.validate_tables(features, targets, index)
    tables, folds = h.diagnose(features, targets, index, forecasts)
    assert len(tables["metric_summary"]) == 4 * 3 * 2
    assert len(tables["fold_metrics"]) == 4 * 3 * 5
    assert len(tables["group_metrics"]) == 4 * 3 * 29
    assert len(tables["residuals"]) == 4 * 3 * 106
    assert len(tables["feature_inventory"]) == 75 and len(folds) == 5
    mean = tables["metric_summary"].query("method == 'constant_mean'")
    assert mean["pinball_0.5"].isna().all()
    assert "NOT NEW OUTER RESULTS" in h.report_text(tables)


def write_json(directory, stem, document):
    directory.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document, allow_nan=False).encode()
    digest = hashlib.sha256(data).hexdigest()
    (directory / (stem + ".json")).write_bytes(data)
    (directory / (stem + ".sha256")).write_text(digest)
    return digest


def disk_fixture(tmp_path):
    features, targets, index, forecasts = fixture_tables()
    root, c4 = tmp_path / "data", tmp_path / "components/c4"
    training = root / "c4_artifacts/meta_datasets/adult" / h.PROTOCOL / "forecast-training-all32-audit-v1"
    training.mkdir(parents=True)
    files = {}
    for name, frame in (("features.csv", features.reset_index()), ("targets.csv", targets.reset_index()),
                        ("scenario_index.csv", index)):
        data = frame.to_csv(index=False).encode()
        (training / name).write_bytes(data)
        files[name] = h.sha(data)
    meta_hash = write_json(training, "metadata", {"fold": "forecast_train", "rows": 106,
        "groups": [f"group-{i:02d}" for i in range(32)], "feature_columns": features.columns.tolist(), "files": files,
        "source_links": {str(root / "DO_NOT_READ_outer.json"): "deliberately_not_opened"}})
    config = {"specification_version": "adult-n1-training-v1", "training_snapshot_version": "forecast-training-all32-audit-v1",
        "training_snapshot_metadata_sha256": meta_hash, "raw_quantiles": [0.05, 0.5, 0.95], "parameters": {"cv_splits": 5}}
    (c4 / "configs").mkdir(parents=True)
    config_data = json.dumps(config).encode()
    (c4 / "configs/n1_benchmark_config.json").write_bytes(config_data)
    _, _, folds = h.constant_references(targets, index)
    fit = root / "c4_artifacts/forecasting/adult" / h.PROTOCOL / "adult-n1-training-v1"
    spec_hash = write_json(fit, "specification", {"binding": {"configuration_sha256": h.sha(config_data),
        "training_metadata_sha256": meta_hash, "internal_folds": folds}})
    (fit / "n1-fit-v1").mkdir()
    oof_files = {}
    for method, frame in forecasts.items():
        name = method + "_grouped_oof.csv"
        data = frame.to_csv(index=True).encode()
        (fit / "n1-fit-v1" / name).write_bytes(data)
        oof_files[name] = h.sha(data)
    write_json(fit / "n1-fit-v1", "metadata", {"training_metadata_sha256": meta_hash,
        "specification_sha256": spec_hash, "files": oof_files})
    return root, c4, fit


def test_loader_reads_training_only_and_no_model_binaries(tmp_path, monkeypatch):
    root, c4, _ = disk_fixture(tmp_path)
    visited = []
    original = Path.read_bytes
    def track(path):
        visited.append(str(path))
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", track)
    features, targets, index, forecasts, folds, pinned = h.load_inputs(root, c4)
    assert features.shape == (106, 75) and len(folds) == 5
    assert not any("outer" in path or path.endswith(".joblib") for path in visited)
    assert len(pinned) == 9
    h.diagnose(features, targets, index, forecasts)


def test_corrupted_oof_artifact_is_rejected(tmp_path):
    root, c4, fit = disk_fixture(tmp_path)
    path = fit / "n1-fit-v1/lightgbm_grouped_oof.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        h.load_inputs(root, c4)


def test_actual_cli_saves_hashed_reports_without_changing_inputs(tmp_path):
    root, c4, _ = disk_fixture(tmp_path)
    script = c4 / "scripts/diagnose_n1_development.py"
    script.parent.mkdir()
    script.write_bytes(SCRIPT.read_bytes())
    before = {p: h.sha(p.read_bytes()) for base in (root, c4) for p in base.rglob("*") if p.is_file()}
    cmd = [sys.executable, str(script), "--data-root", str(root), "--run-id", "synthetic-cli-check"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DIAGNOSIS COMPLETE" in result.stdout
    assert all(h.sha(p.read_bytes()) == digest for p, digest in before.items())
    output = root / "c4_artifacts/forecasting_diagnostics/adult/synthetic-cli-check"
    meta, _ = h.checked_json(output)
    assert meta["n1_refitted"] is False and meta["calibration_policy_outer_accessed"] is False
    for name, digest in meta["files"].items():
        h.checked(output / name, digest)
    rejected = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and "output exists" in rejected.stderr
