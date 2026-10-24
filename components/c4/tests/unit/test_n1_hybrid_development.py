"""Verify the new model matches screened recipes and preserves artifact integrity."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fit_n1_hybrid_development as h
import run_n1_fast_improvement as original


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    index = pd.Index([f"case-{i}" for i in range(48)], name="scenario_id")
    x = pd.DataFrame(rng.normal(size=(48, 4)), index=index, columns=list("abcd"))
    x["duplicate"] = x.a
    x["constant"] = 1.0
    y = pd.DataFrame({t: 0.01*x.a + rng.normal(0, 0.002, 48) for t in h.d.TARGETS}, index=index)
    groups = pd.Series(np.repeat(np.arange(8), 6), index=index)
    return x, y, groups


def test_matches_original_screened_recipes(data):
    x, y, g = data
    model = h.HybridForecaster().fit(x.iloc[:36], y.iloc[:36], g.iloc[:36])
    raw = model.predict(x.iloc[36:], raw=True)
    expected = original.global_ridge(x.iloc[:36], y.iloc[:36, 0], g.iloc[:36], x.iloc[36:])
    np.testing.assert_allclose(raw.iloc[:, :3], expected, rtol=0, atol=1e-14)
    config = next(c for c in h.e.candidates() if c["id"] == "lgb_original_all")
    for j, target in enumerate(h.d.TARGETS[1:], 1):
        expected = h.e.fit_predict(config, x.iloc[:36], y.iloc[:36][target], g.iloc[:36], x.iloc[36:])
        np.testing.assert_allclose(raw.iloc[:, j*3:(j+1)*3], expected, rtol=0, atol=1e-14)
    assert "constant" not in model.ridge["columns"]
    assert "duplicate" not in model.ridge["columns"]


def test_native_save_reload_and_no_overwrite(data, tmp_path):
    x, y, g = data
    model = h.HybridForecaster().fit(x, y, g)
    directory = tmp_path / "fit"
    model.save(directory)
    loaded = h.HybridForecaster.load(directory)
    np.testing.assert_allclose(loaded.predict(x), model.predict(x), rtol=0, atol=1e-14)
    with pytest.raises(FileExistsError):
        model.save(directory)
    state, _ = h.d.checked_json(directory, "model")
    assert state["production_release_authorized"] is False
    assert state["conformal_calibrated"] is False


def test_corrupted_model_rejected(data, tmp_path):
    model = h.HybridForecaster().fit(*data)
    directory = tmp_path / "fit"
    model.save(directory)
    next(directory.glob("*.txt")).write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError):
        h.HybridForecaster.load(directory)


@pytest.mark.parametrize("problem", ["target_order", "target_index", "groups_index", "missing_group", "nan", "infinity", "out_of_range"])
def test_bad_training_data_rejected(data, problem):
    x, y, g = (v.copy() for v in data)
    if problem == "target_order":
        y = y.iloc[:, ::-1]
    elif problem == "target_index":
        y = y.iloc[::-1]
    elif problem == "groups_index":
        g = g.iloc[::-1]
    elif problem == "missing_group":
        g = g.astype(float)
        g.iloc[0] = np.nan
    elif problem == "nan":
        x.iloc[0, 0] = np.nan
    elif problem == "infinity":
        x.iloc[0, 0] = np.inf
    else:
        y.iloc[0, 0] = 1.1
    with pytest.raises(ValueError):
        h.HybridForecaster().fit(x, y, g)


def test_prediction_schema_and_ordered_bands(data):
    x, y, g = data
    model = h.HybridForecaster().fit(x, y, g)
    with pytest.raises(ValueError):
        model.predict(x.iloc[:, ::-1])
    with pytest.raises(ValueError):
        model.predict(x.drop(columns="a"))
    predictions = model.predict(x)
    assert predictions.index.equals(x.index)
    for target in h.d.TARGETS:
        a = predictions[[target+"__"+label for _, label in h.d.QUANTILES]].to_numpy()
        assert np.isfinite(a).all() and (np.abs(a) <= 1).all()
        assert (np.diff(a, axis=1) >= 0).all()


def test_plan_freeze_execute_and_source_change(data, tmp_path, monkeypatch):
    x, y, g = data
    index = pd.DataFrame({"scenario_id": x.index, "group_id": g.to_numpy()})
    oof = h.HybridForecaster().fit(x, y, g).predict(x)
    source = tmp_path / "source.txt"
    source.write_text("fixed", encoding="utf-8")
    pins = {str(source): h.d.sha(source.read_bytes())}
    monkeypatch.setattr(h, "prepare", lambda *_: (x, y, index, oof, dict(pins)))
    root_args = [str(h.__file__), "--data-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", root_args)
    h.main()
    output = tmp_path / "c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1"
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", root_args + ["--freeze-spec"])
    h.main()
    assert (output / "specification.sha256").is_file()
    assert not (output / "fit-v1").exists()
    monkeypatch.setattr(sys, "argv", root_args + ["--execute"])
    h.main()
    metadata, _ = h.d.checked_json(output / "fit-v1")
    assert metadata["production_release_authorized"] is False
    assert metadata["model_roundtrip_max_absolute_error"] < 1e-12
    with pytest.raises(ValueError):
        h.main()


def test_changed_spec_rejected_before_fit(data, tmp_path, monkeypatch):
    x, y, g = data
    index = pd.DataFrame({"scenario_id": x.index, "group_id": g.to_numpy()})
    monkeypatch.setattr(h, "prepare", lambda *_: (x, y, index, None, {}))
    args = [str(h.__file__), "--data-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args + ["--freeze-spec"])
    h.main()
    monkeypatch.setattr(h.e, "environment", lambda: {"changed": True})
    monkeypatch.setattr(sys, "argv", args + ["--execute"])
    with pytest.raises(ValueError, match="Frozen inputs/code/environment differ"):
        h.main()
    assert not (tmp_path / "c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1/fit-v1").exists()
