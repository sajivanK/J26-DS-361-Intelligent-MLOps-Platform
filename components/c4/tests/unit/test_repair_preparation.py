import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.contracts.models import C2Diagnosis
from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import save_adult_split_manifest
from c4.experiments.repair_preparation import prepare_repair_data
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.preview import save_role_preview
from c4.scenarios.slices import SliceRule


@pytest.fixture
def preparation_inputs(tmp_path):
    directory = tmp_path / "uci_adult_data"
    directory.mkdir()

    for filename, start, rows in [
        ("reference.csv", 0, 40),
        ("test.csv", 100, 80),
    ]:
        frame = pd.DataFrame({
            feature: np.arange(start, start + rows, dtype=float)
            for feature in FEATURE_COLUMNS
        })
        frame["target"] = np.arange(rows) % 2
        frame.to_csv(directory / filename, index=False)

    manifest = save_adult_split_manifest(data_root=tmp_path)
    preview = save_role_preview(
        manifest_directory=manifest.manifest_path.parent,
        config=ScenarioConfig(
            mechanism="marginal_multi",
            features=["education_num", "hours_per_week"],
            magnitude=0.5,
            seed=42,
        ),
        true_slice=SliceRule(
            name="test_true_slice",
            feature="age",
            operator="le",
            value=140.0,
        ),
        role="repair",
        preview_id="test-preparation",
        data_root=tmp_path,
    )

    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture.read_text(encoding="utf-8")
    )["c2_diagnosis"]

    # Broad test rule ensures both classes are available for fitting.
    fields["affected_slice_rule"] = "age <= 179"
    diagnosis = C2Diagnosis.model_validate(fields)

    return {
        "preview_directory": preview.directory,
        "manifest_directory": manifest.manifest_path.parent,
        "diagnosis": diagnosis,
        "rule_feature_space": "processed",
        "data_root": tmp_path,
    }


def test_generator_fit_excludes_internal_validation(preparation_inputs):
    prepared = prepare_repair_data(**preparation_inputs)

    fit_ids = set(prepared.generator_fit.index)
    validation_ids = set(prepared.internal_validation.index)
    train_ids = set(prepared.provenance["repair_train_row_ids"])

    assert fit_ids.isdisjoint(validation_ids)
    assert fit_ids <= train_ids
    assert len(train_ids | validation_ids) == 32
    assert prepared.provenance["source_role"] == "repair"
    assert prepared.provenance["release_validation_used"] is False
    assert prepared.provenance["sealed_future_used"] is False


def test_internal_split_is_reproducible(preparation_inputs):
    first = prepare_repair_data(**preparation_inputs)
    second = prepare_repair_data(**preparation_inputs)

    pd.testing.assert_frame_equal(first.generator_fit, second.generator_fit)
    pd.testing.assert_frame_equal(
        first.internal_validation,
        second.internal_validation,
    )
    assert first.provenance == second.provenance


def test_changed_preview_data_is_rejected(preparation_inputs):
    path = preparation_inputs["preview_directory"] / "transformed.csv"
    path.write_bytes(b"altered data")

    with pytest.raises(ValueError, match="Preview file checksum"):
        prepare_repair_data(**preparation_inputs)


def test_empty_diagnosed_fit_population_is_rejected(preparation_inputs):
    fields = preparation_inputs["diagnosis"].model_dump()
    fields["affected_slice_rule"] = "age <= -999"
    preparation_inputs["diagnosis"] = C2Diagnosis.model_validate(fields)

    with pytest.raises(ValueError, match="Insufficient"):
        prepare_repair_data(**preparation_inputs)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_invalid_internal_fraction_is_rejected(
    preparation_inputs, fraction
):
    with pytest.raises(ValueError, match="Validation fraction"):
        prepare_repair_data(
            **preparation_inputs,
            validation_fraction=fraction,
        )