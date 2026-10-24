import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import resume_n1_fast_improvement as r


def record():
    return dict(scenario_id="case-1", group_id="g1", decision_freeze_sha256="f"*64,
        original_outcomes_sha256="o"*64, candidate_model_sha256="m"*64,
        external_file_sha256=r.h.EXTERNAL_SHA, population_rows=[4, 2, 2],
        scope="forecast_train_external_development_only",
        primary_targets=dict(zip(r.d.TARGETS, [.25, 0., .5])))


def kwargs():
    return dict(sid="case-1", group="g1", freeze_hash="f"*64, outcome_hash="o"*64,
                model_hash="m"*64, population_rows=[4, 2, 2])


def test_checkpoint_reuse_preserves_targets():
    value = record()
    assert r.validate_checkpoint(value, **kwargs()) == value["primary_targets"]


@pytest.mark.parametrize("key,value", [("scenario_id", "other"), ("group_id", "other"),
    ("external_file_sha256", "bad"), ("candidate_model_sha256", "bad"),
    ("population_rows", [8, 4, 4])])
def test_foreign_checkpoint_rejected(key, value):
    case = record()
    case[key] = value
    with pytest.raises(ValueError, match="provenance"):
        r.validate_checkpoint(case, **kwargs())


def test_impossible_or_inconsistent_paired_targets_rejected():
    case = record()
    case["primary_targets"][r.d.TARGETS[0]] = .123
    with pytest.raises(ValueError, match="paired accuracy"):
        r.validate_checkpoint(case, **kwargs())
    case["primary_targets"][r.d.TARGETS[0]] = .5
    with pytest.raises(ValueError, match="aggregation"):
        r.validate_checkpoint(case, **kwargs())


def test_native_model_receives_bytearray_not_filename():
    calls = []
    class Model:
        def load_model(self, value): calls.append(value)
        def get_booster(self): return SimpleNamespace(feature_names=r.h.FEATURES)
    r.native_from_bytes(b"native-test-payload", Model)
    assert len(calls) == 1 and isinstance(calls[0], bytearray)
    assert bytes(calls[0]) == b"native-test-payload"


def test_bad_native_feature_schema_rejected():
    class Model:
        def load_model(self, value): pass
        def get_booster(self): return SimpleNamespace(feature_names=["wrong"])
    with pytest.raises(ValueError, match="native model schema"):
        r.native_from_bytes(b"test", Model)


def test_native_bytes_preserve_actual_predictions(tmp_path):
    from xgboost import XGBClassifier
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(40, len(r.h.FEATURES))), columns=r.h.FEATURES)
    y = (x.age > 0).astype(int)
    original = XGBClassifier(n_estimators=5, max_depth=2, random_state=42, n_jobs=1).fit(x, y)
    path = tmp_path / "candidate.ubj"
    original.save_model(path)
    restored = r.native_from_bytes(path.read_bytes())
    np.testing.assert_array_equal(restored.predict(x), original.predict(x))
    np.testing.assert_array_equal(restored.predict_proba(x), original.predict_proba(x))


def test_model_payload_hash_and_role_verified_before_native_loading(tmp_path):
    payload = b"verified-payload"
    digest = r.d.sha(payload)
    metadata = dict(role="candidate", feature_columns=r.h.FEATURES, model_sha256=digest)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "candidate.ubj").write_bytes(payload)
    value, actual = r.model_bytes(tmp_path, "candidate", digest, r.VerifiedBytes())
    assert value == payload and actual == metadata
    (tmp_path / "candidate.ubj").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        r.model_bytes(tmp_path, "candidate", digest, r.VerifiedBytes())


def test_cache_avoids_repeated_reads_but_detects_source_change_on_final_recheck(tmp_path, monkeypatch):
    source = tmp_path / "data"
    source.write_bytes(b"original")
    calls = []
    original = r.d.checked
    def checked(path, expected):
        calls.append(str(path))
        return original(path, expected)
    monkeypatch.setattr(r.d, "checked", checked)
    cache = r.VerifiedBytes()
    digest = r.d.sha(b"original")
    assert cache(source, digest) == cache(source, digest) == b"original"
    assert len(calls) == 1
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        cache.recheck()


def test_new_checkpoints_are_hashed_and_never_overwritten(tmp_path):
    path = tmp_path / "case.json"
    value = record()
    r.save_new_checkpoint(path, value)
    digest = path.with_suffix(".sha256").read_text().strip()
    assert json.loads(r.d.checked(path, digest)) == value
    with pytest.raises(ValueError, match="already exists"):
        r.save_new_checkpoint(path, value)


def test_missing_source_has_clear_python_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Google Drive availability"):
        r.VerifiedBytes()(tmp_path / "absent.ubj", "f"*64)
