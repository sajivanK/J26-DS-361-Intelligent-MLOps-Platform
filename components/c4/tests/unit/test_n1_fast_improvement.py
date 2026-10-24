import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_n1_fast_improvement as h


def raw_record(label="<=50K"):
    return f"39, Private, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, {label}\n"


def test_headerless_first_record_preserved():
    frame = h.raw_table((raw_record() + raw_record(">50K")).encode())
    assert len(frame) == 2 and frame.income.tolist() == [0, 1]


def test_official_external_comment_and_trailing_target_period():
    frame = h.raw_table(("|1x3 Cross validator\n\n" + raw_record(">50K.")).encode())
    assert len(frame) == 1 and frame.income.iloc[0] == 1


def test_named_header_preserved_without_data_loss():
    frame = h.raw_table((",".join(h.RAW_NAMES) + "\n" + raw_record()).encode())
    assert len(frame) == 1


@pytest.mark.parametrize("label", ["bad", "", "nan"])
def test_invalid_target_rejected(label):
    with pytest.raises(ValueError, match="income label"):
        h.raw_table(raw_record(label).encode())


@pytest.mark.parametrize("value", ["inf", "-inf", "NaN"])
def test_nonfinite_numeric_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite|nonfinite"):
        h.raw_table(raw_record().replace("39,", value + ",", 1).encode())


def synthetic_raw(n, offset=0):
    j = np.arange(offset, offset+n)
    values = {c: ["Private"]*n for c in h.RAW_NAMES}
    for c in [*h.CONTINUOUS, "fnlwgt"]:
        values[c] = j + 10
    values["income"] = (j % 3 == 0).astype(int)
    return pd.DataFrame(values)


def manually_processed(raw, ref):
    result = raw[h.FEATURES].copy()
    for c in h.CATEGORICAL:
        result[c] = 0
    for c in h.CONTINUOUS:
        result[c] = (raw[c] - ref[c].mean()) / ref[c].std(ddof=0)
    result["target"] = raw.income
    return result


def test_compatibility_reconstruction_and_feature_only_overlap_exclusion():
    raw = synthetic_raw(500)
    ref, test = train_test_split(raw, test_size=.2, stratify=raw.income, random_state=42)
    external = synthetic_raw(2005, offset=10000)
    external.iloc[0] = raw.iloc[0]
    external.loc[0, "income"] = 1 - raw.income.iloc[0]
    external.iloc[2] = external.iloc[1]
    fresh, info = h.prepare_cohort(raw, external, manually_processed(ref, ref), manually_processed(test, ref))
    assert info["old_feature_overlap_rows"] == 1
    assert info["external_exact_duplicate_extras"] == 1
    assert len(fresh) == 2003 and fresh.index.is_unique


def test_reconstruction_mismatch_stops_before_external_use():
    raw = synthetic_raw(500)
    ref, test = train_test_split(raw, test_size=.2, stratify=raw.income, random_state=42)
    processed = manually_processed(ref, ref)
    processed.iloc[0, 0] += 0.1
    with pytest.raises(ValueError, match="reconstruction failed"):
        h.prepare_cohort(raw, synthetic_raw(2000, 10000), processed, manually_processed(test, ref))


def test_external_unknowns_reported_and_encoder_not_refitted():
    raw = synthetic_raw(500)
    ref, test = train_test_split(raw, test_size=.2, stratify=raw.income, random_state=42)
    external = synthetic_raw(2000, offset=10000)
    external.loc[0, "workclass"] = "UNSEEN"
    fresh, info = h.prepare_cohort(raw, external, manually_processed(ref, ref), manually_processed(test, ref))
    assert info["unknown_category_rows_before_filtering"]["workclass"] == 1
    assert fresh.workclass.nunique() == 1 and fresh.workclass.iloc[0] == 0


def test_fixed_global_ridge_finite_outputs_and_train_only_feature_selection():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    x["duplicate_a"] = x.a * 2
    y = x.a * .01 + rng.normal(0, .001, 30)
    p = h.global_ridge(x.iloc[:20], y.iloc[:20], [f"g{i%5}" for i in range(20)], x.iloc[20:])
    assert p.shape == (10, 3) and np.isfinite(p).all()


def test_grouped_predictions_do_not_use_own_heldout_labels():
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(36, 5)), index=[f"s{i}" for i in range(36)])
    x.columns = [f"f{i}" for i in range(5)]
    targets = pd.DataFrame(rng.normal(0, .01, size=(36, 3)), index=x.index, columns=h.d.TARGETS)
    index = pd.DataFrame({"scenario_id": x.index, "group_id": [f"g{i%9}" for i in range(36)],
                          "split_role": "forecast_train"})
    _, _, a = h.compare(x, targets, index, lambda _: None)
    _, valid = next(h.GroupKFold(5).split(index, groups=index.group_id))
    changed = targets.copy()
    changed.iloc[valid] += .1
    _, _, b = h.compare(x, changed, index, lambda _: None)
    for method in a:
        pd.testing.assert_frame_equal(a[method].iloc[valid], b[method].iloc[valid])
