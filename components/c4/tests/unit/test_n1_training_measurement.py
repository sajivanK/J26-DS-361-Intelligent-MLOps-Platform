import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_n1_training_measurement as h


def example():
    prediction = pd.DataFrame({"source_row_id": [1, 2, 3, 4], "target": [0, 0, 1, 1],
                               "champion_prediction": [0, 1, 1, 0], "candidate_prediction": [1, 0, 1, 1]})
    membership = pd.DataFrame({"source_row_id": [1, 2, 3, 4], "true_slice_member": [True, True, False, False]})
    targets = dict(zip(h.d.TARGETS, [0.25, 0.0, 0.5]))
    return prediction, membership, targets


def test_paired_difference_statistics_and_population_semantics():
    p, m, t = example()
    rows = h.paired_statistics(p, m, t, [1, 2, 3, 4])
    np.testing.assert_allclose([r["margin"] for r in rows], [0.25, 0.0, 0.5])
    assert rows[0]["candidate_only_correct"] == 2 and rows[0]["champion_only_correct"] == 1
    assert rows[1]["population_rows"] == 2
    difference = np.array([-1, 1, 0, 1])
    assert rows[0]["paired_iid_standard_error"] == pytest.approx(difference.std(ddof=1)/2)


def test_target_mismatch_rejected():
    p, m, t = example()
    t[h.d.TARGETS[0]] += 0.1
    with pytest.raises(ValueError, match="does not match"):
        h.paired_statistics(p, m, t, [1, 2, 3, 4])


def test_source_order_mismatch_rejected():
    p, m, t = example()
    m = m.iloc[::-1]
    with pytest.raises(ValueError, match="identities/order"):
        h.paired_statistics(p, m, t, [1, 2, 3, 4])


def test_nonbinary_prediction_rejected():
    p, m, t = example()
    p.loc[0, "candidate_prediction"] = 2
    with pytest.raises(ValueError, match="Nonbinary"):
        h.paired_statistics(p, m, t, [1, 2, 3, 4])


def test_nonboolean_membership_rejected():
    p, m, t = example()
    m["true_slice_member"] = m.true_slice_member.astype(int)
    with pytest.raises(ValueError, match="Invalid true-slice"):
        h.paired_statistics(p, m, t, [1, 2, 3, 4])


def test_duplicate_source_ids_rejected():
    p, m, t = example()
    p.loc[0, "source_row_id"] = 2
    with pytest.raises(ValueError, match="Duplicate"):
        h.paired_statistics(p, m, t, [2, 2, 3, 4])


def test_nontraining_role_rejected_before_any_outcome_read(tmp_path, monkeypatch):
    spec = __import__("importlib.util", fromlist=["util"])
    file = Path(__file__).with_name("test_n1_development_diagnosis.py")
    s = spec.spec_from_file_location("diagnosis_fixture_noise", file)
    fixture = spec.module_from_spec(s)
    s.loader.exec_module(fixture)
    root, _, _ = fixture.disk_fixture(tmp_path)
    folder = root / "c4_artifacts/meta_datasets/adult" / h.d.PROTOCOL / "forecast-training-all32-audit-v1"
    index = pd.read_csv(folder / "scenario_index.csv")
    index.loc[0, "split_role"] = "outer_test"
    contents = index.to_csv(index=False).encode()
    (folder / "scenario_index.csv").write_bytes(contents)
    import json
    metadata = json.loads((folder / "metadata.json").read_bytes())
    metadata["files"]["scenario_index.csv"] = h.d.sha(contents)
    payload = json.dumps(metadata).encode()
    (folder / "metadata.json").write_bytes(payload)
    monkeypatch.setattr(h, "TRAINING_HASH", h.d.sha(payload))
    with pytest.raises(ValueError, match="Non-training"):
        h.load_training(root)
