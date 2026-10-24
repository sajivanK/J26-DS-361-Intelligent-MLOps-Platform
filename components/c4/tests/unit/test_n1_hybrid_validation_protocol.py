"""Read-only design, role isolation and immutable protocol checks."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / "scripts"))
import freeze_n1_hybrid_validation as v
from _benchmark_manifest import build_manifest


@pytest.fixture
def original():
    return json.loads((C4 / "configs/benchmark_protocol.json").read_text(encoding="utf-8"))


def test_fixed_counts_and_no_training_role(original):
    m = v.create_design(original)
    assert len(m["groups"]) == 52
    assert len(m["scenarios"]) == 1924
    assert Counter(g["split_role"] for g in m["groups"]) == v.COUNTS
    assert sum(s["repair_condition"] == "targeted_repair" for s in m["scenarios"]) == 988
    assert all(s["split_role"] != "forecast_train" for s in m["scenarios"])


def test_deterministic_and_original_unchanged(original):
    saved = copy.deepcopy(original)
    assert v.encoded(v.create_design(original)) == v.encoded(v.create_design(original))
    assert original == saved


def test_group_and_child_seed_disjointness(original):
    old = build_manifest(original)
    new = v.create_design(original)
    assert not {g["group_id"] for g in old["groups"]} & {g["group_id"] for g in new["groups"]}
    assert not v.child_seeds(old) & v.child_seeds(new)


def test_all_related_cases_keep_same_role(original):
    m = v.create_design(original)
    mapping = {g["group_id"]: g["split_role"] for g in m["groups"]}
    for g in mapping:
        rows = [s for s in m["scenarios"] if s["group_id"] == g]
        assert len(rows) == 37
        assert sum(s["is_no_drift_control"] for s in rows) == 1
        assert {s["split_role"] for s in rows} == {mapping[g]}
        bases = {}
        for row in rows:
            base = row["shared_base_package_id"]
            bases.setdefault(base, set()).add(row["split_role"])
        assert all(len(roles) == 1 for roles in bases.values())


def test_original_design_factors_unchanged(original):
    old, new = build_manifest(original), v.create_design(original)
    def factors(m):
        group = m["groups"][0]["group_id"]
        return {(s["mechanism"], s["magnitude"], s["diagnosis_condition"], s["repair_condition"],
                 s["label_flip_fraction"], tuple(s["features"]), tuple(map(tuple, s["interaction_pairs"])))
                for s in m["scenarios"] if s["group_id"] == group}
    assert factors(old) == factors(new)


def test_overlap_namespace_rejected(original):
    original["replication"]["seed_namespaces"].append(v.NAMESPACES[0])
    with pytest.raises(ValueError, match="overlap"):
        v.create_design(original)


def test_frozen_files_roundtrip_corruption_and_no_overwrite(original, tmp_path):
    out = tmp_path / "new"
    m = v.create_design(original)
    hashes = v.freeze_bundle(out, {"scenario_manifest": m})
    loaded, digest = v.checked_json(out, "scenario_manifest")
    assert loaded == m and digest == hashes["scenario_manifest"]
    with pytest.raises(FileExistsError):
        v.freeze_bundle(out, {"scenario_manifest": m})
    (out / "scenario_manifest.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Checksum"):
        v.checked_json(out, "scenario_manifest")
