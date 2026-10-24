"""Path adaptation must preserve model bytes and enforce original schema/hash checks."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import _portable_model_io as io_module
from c4.data import FEATURE_COLUMNS


@pytest.fixture
def artifact(tmp_path):
    data = b'verified-test-model-bytes'
    metadata = {'role': 'candidate', 'feature_columns': list(FEATURE_COLUMNS),
                'model_sha256': hashlib.sha256(data).hexdigest()}
    (tmp_path / 'candidate.ubj').write_bytes(data)
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    return tmp_path, data, metadata


def test_native_loader_receives_verified_bytes_instead_of_path(artifact, monkeypatch):
    directory, data, metadata = artifact
    received = []
    class Model:
        def load_model(self, contents):
            received.append(contents)
        def get_booster(self):
            return SimpleNamespace(feature_names=list(FEATURE_COLUMNS))
    monkeypatch.setattr(io_module, 'XGBClassifier', Model)
    _, actual = io_module.load_candidate(directory)
    assert actual == metadata
    assert len(received) == 1 and isinstance(received[0], bytearray)
    assert bytes(received[0]) == data


def test_corrupted_model_is_rejected_before_native_load(artifact, monkeypatch):
    directory, _, _ = artifact
    (directory / 'candidate.ubj').write_bytes(b'changed')
    def unexpected():
        pytest.fail('Native model constructed before checksum validation.')
    monkeypatch.setattr(io_module, 'XGBClassifier', unexpected)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        io_module.load_candidate(directory)


def test_wrong_role_metadata_is_rejected(artifact):
    directory, _, metadata = artifact
    metadata['role'] = 'champion'
    (directory / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='wrong model role'):
        io_module.load_candidate(directory)


def test_short_mirror_is_identical_temporary_and_preserves_originals(artifact):
    directory, data, _ = artifact
    original_metadata = (directory / 'metadata.json').read_bytes()
    with io_module.short_model_mirror(directory, 'candidate.ubj', 'candidate') as mirror:
        assert mirror != directory
        assert (mirror / 'candidate.ubj').read_bytes() == data
        assert (mirror / 'metadata.json').read_bytes() == original_metadata
    assert not mirror.exists()
    assert (directory / 'candidate.ubj').read_bytes() == data
    assert (directory / 'metadata.json').read_bytes() == original_metadata
