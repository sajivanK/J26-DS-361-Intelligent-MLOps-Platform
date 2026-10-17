"""Platform mapping tests; no dataset access or model training."""
from copy import deepcopy
import pytest
from c4.contracts.platform_adapter import adapt_platform_upstream


def payload():
    return {
        "context": {"dataset": "uci", "model_id": "adult-xgb-v1",
                    "current_dataset_uri": "current.csv", "run_id": "run-1",
                    "label_status": "UNAVAILABLE"},
        "c1": {
            "evidence_version": "c1-1", "engine_version": "1",
            "model_id": "adult-xgb-v1", "reference_dataset_uri": "reference.csv",
            "current_dataset_uri": "current.csv", "batch_size": 10,
            "drift_detected": True, "predicted_harm": .1,
            "harm_interval": [0, .2], "harm_metric": "accuracy_drop",
            "harm_severity": "high", "marginal_extent": "single",
            "interaction_suspected": .1, "affected_features": ["age"],
            "detector_scores": {**dict.fromkeys(
                ["ks_mean", "ks_max", "chi2_mean", "chi2_max", "psi_mean",
                 "psi_max", "mmd", "lsdd"], .1), "clf_auc": .6},
            "detectors_fired": 1, "detectors_fired_list": ["ks"],
            "prediction_stats": {"conf_mean": .8, "conf_std": .1,
                "conf_pct_lt_06": .1, "mean_pos_prob": .4,
                "pos_prob_std": .1, "pct_pos_prob_lt_06": .6},
            "warnings": [], "runtime_sec": .1,
        },
        "c2_to_c4": {"root_cause_category": "marginal",
            "root_cause_feature": "age", "affected_slice_rule": "age > 45",
            "slice_support": .2, "diagnosis_stability": .8,
            "affected_features_ranked": ["age"], "diagnosis_version": "c2-1",
            "diagnosis_timestamp": "2026-10-18T00:00:00Z"},
        "c3": {"repair_dataset_uri": "repair.csv",
            "repair_package_status": "APPROVED", "generator": "TVAE",
            "n_synthetic_rows": 10, "augmentation_ratio": .1,
            "fidelity_score": .8, "utility_score": -.01,
            "privacy_status": "PASS", "target_coverage": .9,
            "prescription_confidence": .8, "repair_version": "c3-1",
            "repair_timestamp": "2026-10-18T00:01:00Z"},
    }


def test_mapping_preserves_audit_and_does_not_mutate_input():
    raw = payload(); original = deepcopy(raw)
    result = adapt_platform_upstream(raw)
    assert raw == original
    assert result.audit_envelope == raw
    assert result.inputs.c3_repair_package.utility_score == -.01
    assert not result.prerequisite_failures
    result.audit_envelope["c3"]["repair_version"] = "changed"
    assert raw == original


@pytest.mark.parametrize("key", ["context", "c1", "c2_to_c4", "c3"])
def test_missing_required_record(key):
    raw = payload(); del raw[key]
    with pytest.raises(ValueError): adapt_platform_upstream(raw)


@pytest.mark.parametrize("field", ["model_id", "current_dataset_uri"])
def test_context_mismatch_rejected(field):
    raw = payload(); raw["context"][field] = "different"
    with pytest.raises(ValueError): adapt_platform_upstream(raw)


@pytest.mark.parametrize("key,field", [
    ("c2_to_c4", "diagnosis_timestamp"), ("c3", "repair_timestamp")])
def test_timestamp_required_and_timezone_required(key, field):
    raw = payload(); del raw[key][field]
    with pytest.raises(ValueError): adapt_platform_upstream(raw)
    raw[key][field] = "2026-10-18T00:00:00"
    with pytest.raises(ValueError): adapt_platform_upstream(raw)


@pytest.mark.parametrize("key,field,value,failure", [
    ("c2_to_c4", "diagnosis_stability", .49, "diagnosis_stability"),
    ("c3", "privacy_status", "FAIL", "privacy_check"),
    ("c3", "repair_package_status", "REJECTED", "repair_approved")])
def test_failed_prerequisites_preserved(key, field, value, failure):
    raw = payload(); raw[key][field] = value
    assert failure in adapt_platform_upstream(raw).prerequisite_failures


def test_stability_boundary():
    raw = payload(); raw["c2_to_c4"]["diagnosis_stability"] = .5
    assert not adapt_platform_upstream(raw).prerequisite_failures


def test_unknown_contract_field_rejected():
    raw = payload(); raw["c3"]["future_accuracy"] = .9
    with pytest.raises(ValueError): adapt_platform_upstream(raw)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_rejected(bad):
    raw = payload(); raw["c1"]["runtime_sec"] = bad
    with pytest.raises(ValueError): adapt_platform_upstream(raw)
