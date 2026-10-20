"""Fixed same-cohort recipe equivalence and persisted baseline integrity."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import fit_n1_validation_baselines as b


@pytest.fixture
def data():
    rng = np.random.default_rng(54)
    ids = pd.Index([f"s{i}" for i in range(40)], name="scenario_id")
    x = pd.DataFrame(rng.normal(size=(40, 4)), index=ids, columns=list("abcd"))
    x["constant"] = 1.
    targets = pd.DataFrame({t: .02*x.a + rng.normal(0, .004, 40) for t in b.d.TARGETS}, index=ids)
    groups = pd.Series(np.repeat(range(8), 5), index=ids)
    return x, targets, groups


@pytest.mark.parametrize("kind,recipe", [("lightgbm", "lgb_original_all"),
    ("linear", "quantile_0.001_all"), ("constant", "constant")])
def test_exact_screened_recipe_equivalence(data, kind, recipe):
    x, y, g = data
    model = b.FixedBaseline(kind).fit(x.iloc[:30], y.iloc[:30], g.iloc[:30])
    actual = model.predict(x.iloc[30:], raw=True)
    candidate = next(c for c in b.e.candidates() if c["id"] == recipe)
    for j, target in enumerate(b.d.TARGETS):
        expected = b.e.fit_predict(candidate, x.iloc[:30], y.iloc[:30][target], g.iloc[:30], x.iloc[30:])
        np.testing.assert_allclose(actual.iloc[:, j*3:(j+1)*3], expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("kind", ["lightgbm", "linear", "constant"])
def test_save_reload_no_overwrite_and_corruption(data, tmp_path, kind):
    x, y, g = data
    model = b.FixedBaseline(kind).fit(x, y, g)
    model.save(tmp_path / "model")
    loaded = b.FixedBaseline.load(tmp_path / "model")
    np.testing.assert_allclose(model.predict(x), loaded.predict(x), rtol=0, atol=1e-13)
    with pytest.raises(FileExistsError):
        model.save(tmp_path / "model")
    path = next((tmp_path / "model").glob("*.txt"), tmp_path / "model/model.json")
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        b.FixedBaseline.load(tmp_path / "model")


@pytest.mark.parametrize("problem", ["target_order", "target_identity", "group_identity", "nonfinite", "target_bounds"])
def test_training_rejects_bad_inputs(data, problem):
    x, y, g = (a.copy() for a in data)
    if problem == "target_order":
        y = y.iloc[:, ::-1]
    elif problem == "target_identity":
        y = y.iloc[::-1]
    elif problem == "group_identity":
        g = g.iloc[::-1]
    elif problem == "nonfinite":
        x.iloc[0, 0] = np.inf
    else:
        y.iloc[0, 0] = 1.2
    with pytest.raises(ValueError):
        b.FixedBaseline("linear").fit(x, y, g)


def test_prediction_schema_and_quantile_order(data):
    x, y, g = data
    model = b.FixedBaseline("linear").fit(x, y, g)
    with pytest.raises(ValueError, match="feature names/order"):
        model.predict(x.iloc[:, ::-1])
    pred = model.predict(x)
    assert pred.index.equals(x.index)
    for target in b.d.TARGETS:
        a = pred[[target+"__"+label for _, label in b.d.QUANTILES]].to_numpy()
        assert np.isfinite(a).all() and (np.abs(a) <= 1).all()
        assert (np.diff(a, axis=1) >= 0).all()


def test_full_baseline_fit_binds_four_models_and_preserves_original(data, tmp_path):
    x, y, g = data
    index = pd.DataFrame({"scenario_id": x.index, "group_id": g.to_numpy()})
    original = tmp_path / "hybrid-placeholder"
    original.write_bytes(b"unchanged")
    pins = {str(original): b.d.sha(original.read_bytes())}
    primary = {"directory": "existing-hybrid", "metadata_sha256": "frozen-by-protocol"}
    b.run_fit(tmp_path, x, y, index, pins, primary)
    meta, _ = b.d.checked_json(tmp_path / "baseline-fit-v1")
    assert len(meta["models"]) == 4
    assert meta["new_validation_cases_accessed"] is False
    assert meta["production_release_authorized"] is False
    assert meta["fresh_validation_executor_implemented"] is False
    assert all(error <= 1e-12 for error in meta["model_reload_max_absolute_errors"].values())
    for name, digest in meta["files"].items():
        b.d.checked(tmp_path / "baseline-fit-v1" / name, digest)
    assert original.read_bytes() == b"unchanged"
    with pytest.raises(ValueError, match="output exists"):
        b.run_fit(tmp_path, x, y, index, pins, primary)


def test_source_change_rejected_before_fitting(data, tmp_path):
    x, y, g = data
    source = tmp_path / "source"
    source.write_bytes(b"changed")
    index = pd.DataFrame({"scenario_id": x.index, "group_id": g.to_numpy()})
    with pytest.raises(ValueError, match="Checksum"):
        b.run_fit(tmp_path, x, y, index, {str(source): b.d.sha(b"original")}, {})
    assert not (tmp_path / "baseline-fit-v1").exists()


@pytest.mark.parametrize("kind", ["constant", "linear", "lightgbm"])
def test_no_varying_feature_fallback(data, kind):
    x, y, g = data
    x.loc[:, :] = 1.
    model = b.FixedBaseline(kind).fit(x, y, g)
    assert np.isfinite(model.predict(x).to_numpy()).all()


def test_sorted_protocol_json_preserves_frozen_manifest_order():
    original_bytes = (C4 / "configs/benchmark_protocol.json").read_bytes()
    original = json.loads(original_bytes)
    manifest = b.v.create_design(original)
    saved_protocol = json.loads(b.e.encoded({"factorial_source_protocol": original}))
    # Reproduce the reported bug: dictionary equality survives serialization,
    # but strength traversal changes the order inside the scenario list.
    assert saved_protocol["factorial_source_protocol"] == original
    assert b.v.create_design(saved_protocol["factorial_source_protocol"]) != manifest
    b.verify_manifest(saved_protocol, manifest, original_bytes)


@pytest.mark.parametrize("change", ["scenario_order", "seed", "source_factor"])
def test_manifest_fix_still_rejects_changes(change):
    original_bytes = (C4 / "configs/benchmark_protocol.json").read_bytes()
    original = json.loads(original_bytes)
    manifest = b.v.create_design(original)
    protocol = json.loads(b.e.encoded({"factorial_source_protocol": original}))
    if change == "scenario_order":
        manifest["scenarios"] = manifest["scenarios"][::-1]
    elif change == "seed":
        manifest["scenarios"][0]["generator_seed"] += 1
    else:
        protocol["factorial_source_protocol"]["injection_strengths"]["high"] = .99
    with pytest.raises(ValueError, match="differs"):
        b.verify_manifest(protocol, manifest, original_bytes)
