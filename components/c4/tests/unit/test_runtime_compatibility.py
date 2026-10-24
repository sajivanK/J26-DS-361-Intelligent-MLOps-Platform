"""Pinned runtime binding checks with temporary synthetic files only."""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from c4.contracts.runtime_compatibility import (
    PinnedFile, RuntimeBindings, verify_runtime_compatibility,
)


def fixture(tmp_path):
    def pin(name, data):
        data = data if isinstance(data, bytes) else json.dumps(data).encode()
        path = tmp_path / name; path.write_bytes(data)
        return PinnedFile(path, hashlib.sha256(data).hexdigest())
    model = pin("model.ubj", b"test-model-bytes-not-deserialized")
    prep = pin("preprocessing.bin", b"test-preprocessing-bytes-not-deserialized")
    reference = pin("reference.csv", b"age,target\n0.2,0\n0.8,1\n")
    validation = pin("validation.csv", b"age,target\n0.4,0\n0.9,1\n")
    repair = pin("repair.csv", b"target,age\n1,0.7\n")
    metadata = pin("metadata.json", {
        "role": "champion", "dataset": "uci_adult", "feature_columns": ["age"],
        "target_column": "target", "model_sha256": model.sha256,
        "reference_csv_sha256": reference.sha256,
    })
    lineage_data = {
        "reference_csv_sha256": reference.sha256,
        "release_validation_csv_sha256": validation.sha256,
        "repair_csv_sha256": repair.sha256,
        "preprocessing_sha256": prep.sha256,
        "reference_row_ids": ["source:0", "source:1"],
        "release_validation_row_ids": ["source:2", "source:3"],
        "repair_source_row_ids": ["source:4", "source:5"],
        "repair_source_role": "dedicated_repair",
        "repair_excluded_release_validation": True, "repair_excluded_future": True,
    }
    lineage = pin("lineage.json", lineage_data)
    bindings = RuntimeBindings("uci", "adult-xgb-v1", model, metadata, prep,
        reference, validation, repair, lineage, ("age",), "target", "processed")
    adapted = SimpleNamespace(prerequisite_failures=(),
        context=SimpleNamespace(dataset="uci", model_id="adult-xgb-v1"),
        inputs=SimpleNamespace(c3_repair_package=SimpleNamespace(
            repair_dataset_uri=str(repair.path), n_synthetic_rows=1, augmentation_ratio=.5)))
    return adapted, bindings, lineage_data, pin


def test_bindings_pass_without_authorizing_models_forecasts_or_release(tmp_path):
    adapted, bindings, _, _ = fixture(tmp_path)
    out = verify_runtime_compatibility(adapted, bindings)
    assert out.artifact_bindings_verified and out.declared_lineage_checked
    assert out.row_counts == {"reference": 2, "release_validation": 2, "repair": 1}
    assert not out.model_runtime_validated
    assert not out.forecast_applicable
    assert not out.production_release_authorized
    assert "true injected" in out.forecast_reason
    bindings.repair.path.write_bytes(b"changed-after-check")
    assert out.verified_bytes["repair"] == b"target,age\n1,0.7\n"


@pytest.mark.parametrize("name", ["champion_model", "champion_metadata", "preprocessing",
    "reference", "release_validation", "repair", "lineage"])
def test_tampered_file_rejected(tmp_path, name):
    adapted, bindings, _, _ = fixture(tmp_path)
    getattr(bindings, name).path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_runtime_compatibility(adapted, bindings)


@pytest.mark.parametrize("field,value", [
    ("dataset", "ieee"), ("champion_model_id", "wrong"),
    ("rule_feature_space", "raw"), ("feature_columns", ("wrong",))])
def test_configuration_mismatch(tmp_path, field, value):
    adapted, bindings, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        verify_runtime_compatibility(adapted, replace(bindings, **{field: value}))


@pytest.mark.parametrize("key,value", [
    ("reference_row_ids", ["source:0", "source:0"]),
    ("release_validation_row_ids", ["source:0", "source:3"]),
    ("repair_source_row_ids", ["source:2"]),
    ("repair_excluded_future", False),
    ("repair_excluded_release_validation", False),
    ("preprocessing_sha256", "0" * 64),
    ("reference_csv_sha256", "0" * 64),
    ("release_validation_row_ids", ["source:2"]),
])
def test_bad_lineage_rejected(tmp_path, key, value):
    adapted, bindings, data, pin = fixture(tmp_path)
    data[key] = value
    bindings = replace(bindings, lineage=pin("lineage.json", data))
    with pytest.raises(ValueError): verify_runtime_compatibility(adapted, bindings)


@pytest.mark.parametrize("csv", [b"age,target\nNaN,1\n", b"age,target\n0.7,2\n",
    b"age,target\n", b"age,other\n0.7,1\n", b"age,target\n0.7\n"])
def test_bad_csv_rejected_even_if_pinned(tmp_path, csv):
    adapted, bindings, _, pin = fixture(tmp_path)
    repair = pin("repair.csv", csv)
    with pytest.raises(ValueError):
        verify_runtime_compatibility(adapted, replace(bindings, repair=repair))


def test_failed_prerequisites(tmp_path):
    adapted, bindings, _, _ = fixture(tmp_path)
    adapted.prerequisite_failures = ("privacy_check",)
    with pytest.raises(ValueError, match="prerequisites"):
        verify_runtime_compatibility(adapted, bindings)


@pytest.mark.parametrize("field,value", [
    ("n_synthetic_rows", 2), ("augmentation_ratio", .3),
    ("repair_dataset_uri", "repair.csv"), ("repair_dataset_uri", "/different.csv")])
def test_c3_mismatch_rejected(tmp_path, field, value):
    adapted, bindings, _, _ = fixture(tmp_path)
    setattr(adapted.inputs.c3_repair_package, field, value)
    with pytest.raises(ValueError): verify_runtime_compatibility(adapted, bindings)
