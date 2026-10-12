import hashlib
import json

import pytest

from c4.data import FEATURE_COLUMNS
from c4.scenarios.feature_metadata import load_adult_feature_metadata


def write_metadata(tmp_path, document):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    return path


def test_shared_metadata_loads_with_exact_checksum():
    metadata = load_adult_feature_metadata()

    assert set(metadata.feature_types) == set(FEATURE_COLUMNS)
    assert metadata.feature_types["age"] == "continuous"
    assert metadata.feature_types["native_country"] == "categorical"
    assert metadata.source_sha256 == hashlib.sha256(
        metadata.source_path.read_bytes()
    ).hexdigest()


def test_missing_feature_is_rejected(tmp_path):
    types = load_adult_feature_metadata().feature_types.copy()
    types.pop("age")
    path = write_metadata(tmp_path, {"feature_types": types})

    with pytest.raises(ValueError, match="feature schema"):
        load_adult_feature_metadata(path)


def test_unsupported_type_is_rejected(tmp_path):
    types = load_adult_feature_metadata().feature_types.copy()
    types["age"] = "unknown"
    path = write_metadata(tmp_path, {"feature_types": types})

    with pytest.raises(ValueError, match="unsupported feature type"):
        load_adult_feature_metadata(path)


def test_type_conflicting_with_slice_schema_is_rejected(tmp_path):
    types = load_adult_feature_metadata().feature_types.copy()
    types["native_country"] = "continuous"
    path = write_metadata(tmp_path, {"feature_types": types})

    with pytest.raises(ValueError, match="conflict"):
        load_adult_feature_metadata(path)


def test_missing_type_mapping_is_rejected(tmp_path):
    path = write_metadata(tmp_path, {"dataset": "Adult"})

    with pytest.raises(ValueError, match="contain feature_types"):
        load_adult_feature_metadata(path)