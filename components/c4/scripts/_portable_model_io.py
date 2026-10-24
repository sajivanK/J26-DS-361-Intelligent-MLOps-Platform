"""Verified model I/O avoiding long native filenames; core files stay unchanged."""
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from xgboost import XGBClassifier

from _common import require, sha, checked_bytes
from c4.data import FEATURE_COLUMNS
from c4.models import save_pre_release_evidence as _save_pre_release_evidence


def _verified_model(directory, filename, role):
    directory = Path(directory)
    metadata_bytes = (directory / 'metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    require(metadata.get('role') == role, 'Artifact metadata has the wrong model role.')
    require(metadata.get('feature_columns') == list(FEATURE_COLUMNS), 'Model metadata feature schema differs.')
    model_bytes = checked_bytes(directory / filename, metadata.get('model_sha256'))
    return model_bytes, metadata_bytes, metadata


def _load_bytes(directory, filename, role):
    model_bytes, _, metadata = _verified_model(directory, filename, role)
    model = XGBClassifier()
    model.load_model(bytearray(model_bytes))
    require(model.get_booster().feature_names == list(FEATURE_COLUMNS), 'Model feature schema differs.')
    return model, metadata


def load_candidate(directory):
    return _load_bytes(directory, 'candidate.ubj', 'candidate')


def load_champion(directory):
    return _load_bytes(directory, 'champion.ubj', 'champion')


@contextmanager
def short_model_mirror(directory, filename, role):
    """Supply unmodified core loaders with byte-identical short temporary files."""
    directory = Path(directory)
    model_bytes, metadata_bytes, _ = _verified_model(directory, filename, role)
    with tempfile.TemporaryDirectory(prefix='c4io-') as temporary:
        mirror = Path(temporary)
        require(len(str(mirror / filename)) < 240, 'System temporary model path is unexpectedly long.')
        (mirror / filename).write_bytes(model_bytes)
        (mirror / 'metadata.json').write_bytes(metadata_bytes)
        checked_bytes(mirror / filename, sha(model_bytes))
        checked_bytes(mirror / 'metadata.json', sha(metadata_bytes))
        yield mirror
        # The original model and metadata must remain byte-identical.
        checked_bytes(directory / filename, sha(model_bytes))
        checked_bytes(directory / 'metadata.json', sha(metadata_bytes))


def save_pre_release_evidence(**kwargs):
    """Preserve original scoring/provenance logic; only native load paths change.

    Persistent evidence identifies models by their original version and SHA256.
    The caller records original directories and this adapter's source checksum.
    No original artifact or core implementation is modified.
    """
    original_candidate = kwargs['candidate_directory']
    original_champion = kwargs['champion_directory']
    with short_model_mirror(original_candidate, 'candidate.ubj', 'candidate') as candidate:
        with short_model_mirror(original_champion, 'champion.ubj', 'champion') as champion:
            return _save_pre_release_evidence(**{**kwargs,
                'candidate_directory': candidate, 'champion_directory': champion})
