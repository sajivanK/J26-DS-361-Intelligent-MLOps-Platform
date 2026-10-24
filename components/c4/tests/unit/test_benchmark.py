from pathlib import Path

import pytest
import yaml

from c4.scenarios.benchmark import load_draft_benchmark


@pytest.fixture
def draft_document():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs" / "scenario_config.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_draft(tmp_path, document):
    path = tmp_path / "draft.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_valid_draft_builds_injection_grid(tmp_path, draft_document):
    draft = load_draft_benchmark(write_draft(tmp_path, draft_document))

    # 3 mechanisms × 3 strengths × 4 seeds.
    assert len(draft.injection_configs) == 36

    # Then × 2 diagnosis conditions × 2 repair conditions.
    assert draft.proposed_scenario_count == 144
    assert draft.document["execution_enabled"] is False
    assert len(draft.yaml_sha256) == 64


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "draft.yaml"
    path.write_text("status: draft\nstatus: frozen\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_draft_benchmark(path)


def test_invalid_mechanism_is_rejected(tmp_path, draft_document):
    draft_document["drift_conditions"][0]["mechanism"] = "unknown"

    with pytest.raises(ValueError):
        load_draft_benchmark(write_draft(tmp_path, draft_document))


def test_duplicate_seeds_are_rejected(tmp_path, draft_document):
    draft_document["seed_families"] = [42, 42, 2026, 3407]

    with pytest.raises(ValueError, match="unique integers"):
        load_draft_benchmark(write_draft(tmp_path, draft_document))


def test_invalid_strength_order_is_rejected(tmp_path, draft_document):
    draft_document["injection_strengths"]["low"] = 0.9

    with pytest.raises(ValueError, match="Injection strengths"):
        load_draft_benchmark(write_draft(tmp_path, draft_document))


def test_incorrect_scenario_count_is_rejected(tmp_path, draft_document):
    draft_document["design"]["proposed_factorial_combinations"] = 288

    with pytest.raises(ValueError, match="scenario count"):
        load_draft_benchmark(write_draft(tmp_path, draft_document))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "frozen"),
        ("execution_enabled", True),
    ],
)
def test_execution_cannot_be_enabled(
    tmp_path, draft_document, field, value
):
    draft_document[field] = value

    with pytest.raises(ValueError, match="non-executable drafts"):
        load_draft_benchmark(write_draft(tmp_path, draft_document))