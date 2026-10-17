"""
Fake C1 output (contract v1.1) built from inject_drift ground truth.
J26-DS-361 | maintained by Sajivan K (IT23172296)

For independent development of C2, C3 and C4 before the real C1 API is connected.
Field names, types and order match shared/contracts/c1_output_schema.json v1.1 exactly.

The fake is built from the injection ANSWER KEY, so it is cleaner than the real
C1 output. In particular, the real engine:
  - predicts harm with an error of about 0.7 pp (the fake uses the true drop),
  - gives interaction_suspected about 0.46 on clean batches (the fake uses 0.2),
  - misses most interaction-only drift in drift_detected (the fake follows the rule below,
    which also misses it, because interaction rarely makes 3 detectors fire).
Do not report research results measured on fake C1 values as C1 results.

Usage:
    from contracts.c1_fake_output import build_fake_c1_output   # shared/ on sys.path
    c1_output = build_fake_c1_output(result["ground_truth"], seed=42)
"""
import math

_SEVERITY = [(1.0, "none"), (3.0, "low"), (7.0, "moderate"), (15.0, "high")]  # pp, else critical
_EXTENT = {"none": "none", "interaction": "none", "marginal_single": "single",
           "combined": "single", "marginal_multi": "multi"}
_P = 0.8744    # UCI Adult fixed baseline accuracy
_N = 6513      # UCI Adult test pool (finite-population correction)
_Q = 3.0369    # C1 engine 1.2.1 conformal quantile


def _harm_severity(drop_pp):
    for limit, name in _SEVERITY:
        if drop_pp < limit:          # negative drops are 'none' (not 'critical')
            return name
    return "critical"


def build_fake_c1_output(ground_truth, seed=42, model_id="adult-xgb-v1",
                         reference_dataset_uri="shared/uci_adult/data/reference.csv",
                         current_dataset_uri=None):
    gt = ground_truth
    dtype = gt["drift_type"]
    drop_pp = float(gt["actual_accuracy_drop"])               # percentage points
    bs = int(gt.get("batch_size") or 1000)
    extent = _EXTENT[dtype]
    interaction = dtype in ("interaction", "combined")

    # interval with the same width rule as the real engine (lower bound not clipped)
    bs_c = min(max(bs, 500), 3000)
    sigma = 100 * math.sqrt(_P * (1 - _P) / bs_c * (_N - bs_c) / (_N - 1))
    half = _Q * sigma

    ks = gt.get("per_feature_ks") or {}
    ks_vals = list(ks.values()) or [0.02]
    marginal = extent != "none"
    affected = list(gt["injected_features"]) if marginal else []

    fired = (["ks", "chi2", "psi", "mmd", "lsdd", "classifier"] if marginal
             else (["mmd", "classifier"] if interaction else []))

    return {
        "evidence_version": f"c1-{seed:08x}",
        "engine_version": "fake-1.1",
        "model_id": model_id,
        "reference_dataset_uri": reference_dataset_uri,
        "current_dataset_uri": current_dataset_uri or f"data/scenarios/batch_{seed}.csv",
        "batch_size": bs,
        "drift_detected": bool(marginal or len(fired) >= 3),
        "predicted_harm": round(drop_pp / 100, 4),
        "harm_interval": [round((drop_pp - half) / 100, 4), round((drop_pp + half) / 100, 4)],
        "harm_metric": "accuracy_drop",
        "harm_severity": _harm_severity(drop_pp),
        "marginal_extent": extent,
        "interaction_suspected": 0.85 if interaction else 0.2,
        "affected_features": affected,
        "detector_scores": {
            "ks_mean": round(sum(ks_vals) / len(ks_vals), 4),
            "ks_max": round(max(ks_vals), 4),
            "chi2_mean": 0.2 if marginal else 0.02,
            "chi2_max": 0.6 if marginal else 0.05,
            "psi_mean": 0.3 if marginal else 0.01,
            "psi_max": 2.5 if marginal else 0.04,
            "mmd": 0.03 if (marginal or interaction) else 0.001,
            "lsdd": 0.08 if marginal else 0.001,
            "clf_auc": 0.86 if marginal else (0.55 if interaction else 0.5),
        },
        "detectors_fired": len(fired),
        "detectors_fired_list": fired,
        "prediction_stats": {
            "conf_mean": 0.87, "conf_std": 0.15, "conf_pct_lt_06": 0.08,
            "mean_pos_prob": 0.24, "pos_prob_std": 0.30, "pct_pos_prob_lt_06": 0.83,
        },
        "warnings": [],
        "runtime_sec": 0.0,
    }
