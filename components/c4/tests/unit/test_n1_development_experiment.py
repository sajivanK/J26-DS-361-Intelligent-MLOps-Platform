"""Synthetic checks for nested selection, training isolation and frozen protocol."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_n1_development as h


def tiny_data():
    rng = np.random.default_rng(17)
    x = pd.DataFrame({"c1_predicted_harm": rng.normal(size=36),
                      "c4_global_champion_accuracy": rng.normal(size=36),
                      "constant": np.ones(36)}, index=[f"s{i}" for i in range(36)])
    index = pd.DataFrame({"scenario_id": x.index, "group_id": [f"g{i % 9}" for i in range(36)],
                          "split_role": "forecast_train"})
    y = pd.DataFrame({t: 0.01*x.iloc[:, 0] + rng.normal(0, 0.001, len(x)) for t in h.d.TARGETS})
    grid = [h.candidates()[0], dict(id="ridge", kind="ridge", features="compact", parameters={"alpha": 1.0})]
    return x, y, index, grid


def test_nested_selection_cannot_see_own_heldout_labels():
    x, y, index, grid = tiny_data()
    a, selections, _, _ = h.nested_study(x, y, index, grid, 3, 2, lambda _: None)
    fold = h.split_records(index, 3)[0]
    changed = y.copy()
    changed.loc[fold["validation_ids"]] += 0.5
    b, changed_selections, _, _ = h.nested_study(x, changed, index, grid, 3, 2, lambda _: None)
    pd.testing.assert_frame_equal(a.loc[fold["validation_ids"]], b.loc[fold["validation_ids"]])
    pd.testing.assert_frame_equal(selections.query("fold == 1"), changed_selections.query("fold == 1"))


def test_constant_has_equal_weight_per_group():
    candidate = h.candidates()[0]
    x = pd.DataFrame({"x": range(11)})
    p = h.fit_predict(candidate, x, pd.Series([0.0]*10 + [1.0]), ["a"]*10 + ["b"], x.iloc[:2])
    assert np.allclose(p[:, 0], 0) and np.allclose(p[:, 2], 1)


def test_screen_requires_both_objectives_to_beat_baseline():
    x, y, index, grid = tiny_data()
    best, table, _ = h.screen(x, y.iloc[:, 0], index, grid, 3)
    baseline = table.query("candidate == 'constant'").iloc[0]
    assert best["rmse"] <= baseline.rmse + 1e-15
    assert best["mean_pinball"] <= baseline.mean_pinball + 1e-15
    assert best["score"] <= 1 + 1e-12


def test_no_varying_training_features_uses_training_constant():
    x, y, index, grid = tiny_data()
    train = x.iloc[:20].copy()
    train.loc[:, :] = 1
    a = h.fit_predict(grid[1], train, y.iloc[:20, 0], index.iloc[:20].group_id, x.iloc[20:])
    b = h.fit_predict(grid[0], train, y.iloc[:20, 0], index.iloc[:20].group_id, x.iloc[20:])
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["lightgbm", "linear_quantile", "ridge"])
def test_actual_model_families_predict_finite_quantiles(kind):
    x, y, index, _ = tiny_data()
    candidate = next(c for c in h.candidates() if c["kind"] == kind and c["features"] == "compact")
    p = h.fit_predict(candidate, x.iloc[:24], y.iloc[:24, 0], index.iloc[:24].group_id, x.iloc[24:])
    assert p.shape == (12, 3) and np.isfinite(p).all()
    assert (np.diff(h.ordered(p), axis=1) >= 0).all()


def test_scaler_fits_training_rows_only(monkeypatch):
    x, y, index, grid = tiny_data()
    original = h.StandardScaler.fit
    seen = []
    def track(self, a, *args, **kwargs):
        seen.append(a.copy())
        return original(self, a, *args, **kwargs)
    monkeypatch.setattr(h.StandardScaler, "fit", track)
    valid = x.iloc[24:].copy()
    valid.loc[:, :] = 1e12
    h.fit_predict(grid[1], x.iloc[:24], y.iloc[:24, 0], index.iloc[:24].group_id, valid)
    assert len(seen) == 1 and seen[0].shape[0] == 24 and abs(seen[0]).max() < 10


def test_protocol_pins_code_data_and_explicit_limits():
    x, _, index, grid = tiny_data()
    p = h.make_protocol(x, index, {"training.csv": "a"*64}, grid)
    assert p["scope"] == "forecast_train_only"
    assert p["fresh_final_evaluation_required"] and not p["final_models_fitted"]
    assert len(p["scripts"]) == 2 and p["inputs"]["training.csv"] == "a"*64
    assert h.encoded(p) == h.encoded(json.loads(h.encoded(p)))


def test_cli_freeze_execute_integrity_and_no_source_overwrite(tmp_path, monkeypatch):
    x, y, index, grid = tiny_data()
    q, _, folds = h.d.constant_references(y, index)
    source = tmp_path / "training-input.txt"
    source.write_text("unchanged")
    pinned = {str(source): h.d.sha(source.read_bytes())}
    monkeypatch.setattr(h.d, "load_inputs", lambda *args: (x, y, index, {"lightgbm": q, "linear": q}, folds, pinned))
    monkeypatch.setattr(h, "candidates", lambda: grid)
    args = ["experiment", "--data-root", str(tmp_path), "--run-id", "synthetic"]
    monkeypatch.setattr(sys, "argv", args + ["--freeze-spec"])
    h.main()
    directory = tmp_path / "c4_artifacts/forecasting_experiments/adult/synthetic"
    original_spec = (directory / "specification.json").read_bytes()
    (directory / "specification.json").write_bytes(original_spec + b" ")
    monkeypatch.setattr(sys, "argv", args + ["--execute"])
    with pytest.raises(ValueError, match="Checksum mismatch"):
        h.main()
    (directory / "specification.json").write_bytes(original_spec)
    original_environment = h.environment
    monkeypatch.setattr(h, "environment", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="environment changed"):
        h.main()
    monkeypatch.setattr(h, "environment", original_environment)
    h.main()
    meta, _ = h.d.checked_json(directory / "results")
    assert not meta["final_models_fitted"] and not meta["production_release_authorized"]
    for name, digest in meta["files"].items():
        h.d.checked(directory / "results" / name, digest)
    assert source.read_text() == "unchanged"
    assert "NOT NEW OUTER RESULTS" in (directory / "results/N1_DEVELOPMENT_EXPERIMENT.md").read_text()
    with pytest.raises(ValueError, match="already exist"):
        h.main()
