"""Live evidence mapping to the frozen benchmark feature schema.

Reports unavailable inputs explicitly. Does not perform model inference.
"""
import math


def build_feature_inventory(*, upstream, evaluation, columns):
    if len(columns) != 75 or len(set(columns)) != 75:
        raise ValueError("Expected the unique frozen 75-feature schema.")

    values = {}

    def add(name, value):
        if value is None:
            values[name] = None
            return
        if type(value) not in (int, float, bool):
            raise ValueError(f"Non-numeric feature: {name}")
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite feature: {name}")
        values[name] = float(value)

    c1 = upstream["c1"]
    c2 = upstream["c2_to_c4"]
    c3 = upstream["c3"]

    for key in (
        "batch_size", "drift_detected", "predicted_harm",
        "interaction_suspected", "detectors_fired",
    ):
        add("c1_" + key, c1[key])

    lower, upper = c1["harm_interval"]
    add("c1_harm_lower", lower)
    add("c1_harm_upper", upper)
    add("c1_harm_width", upper - lower)

    for section in ("detector_scores", "prediction_stats"):
        for key in sorted(c1[section]):
            add(f"c1_{section}_{key}", c1[section][key])

    add("c1_affected_feature_count", len(c1["affected_features"]))
    add("c1_warning_count", len(c1["warnings"]))

    for key in ("slice_support", "diagnosis_stability"):
        add("c2_" + key, c2[key])
    add("c2_ranked_feature_count", len(c2["affected_features_ranked"]))

    for key in (
        "n_synthetic_rows", "augmentation_ratio", "fidelity_score",
        "utility_score", "target_coverage", "prescription_confidence",
    ):
        add("c3_" + key, c3[key])
    add("c3_repair_approved", c3["repair_package_status"] == "APPROVED")
    add("c3_privacy_screen_pass", c3["privacy_status"] == "PASS")

    comparisons = evaluation["pre_release_report"]["comparisons"]
    for population in ("global", "affected_slice", "non_affected"):
        report = comparisons[population]
        add(f"c4_{population}_rows", report["rows"])

        for model in ("champion", "candidate", "candidate_minus_champion"):
            for metric in ("accuracy", "macro_f1", "auc"):
                value = report[model][metric]
                name = f"c4_{population}_{model}_{metric}"

                if metric == "auc":
                    # Reproduce the original explicit missing-AUC encoding.
                    add(name + "_missing", value is None)
                    add(name, 0.0 if value is None else value)
                else:
                    if value is None:
                        raise ValueError(f"Required metric unavailable: {name}")
                    add(name, value)

    missing_names = sorted(set(columns) - set(values))
    extra_names = sorted(set(values) - set(columns))
    if missing_names or extra_names:
        raise ValueError(
            f"Feature-name mismatch: missing={missing_names}; "
            f"extra={extra_names}"
        )

    unavailable = [name for name in columns if values[name] is None]

    return {
        "status": (
            "FEATURE_VALUES_AVAILABLE"
            if not unavailable else "FEATURE_VALUES_INCOMPLETE"
        ),
        "columns": list(columns),
        "values": [values[name] for name in columns],
        "available_count": len(columns) - len(unavailable),
        "unavailable_features": unavailable,
        "repair_strategy": c3.get("repair_strategy"),
        "model_applicability_validated": False,
        "live_calibration_applicability_validated": False,
        "N1_executed": False,
        "N2_executed": False,
        "production_release_authorized": False,
    }