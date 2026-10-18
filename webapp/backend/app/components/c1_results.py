"""
Recorded C1 research results shown on the C1 page ("Why trust it", "Model and engine").
These are NOT computed by the backend: they are copied from the committed engine 1.2.3 results
(c1/results/, project records C1_v12_*_Record.md). Update here if the engine is retrained.
Live values (baseline accuracy/AUC, batch sizes, interval quantiles) come from the loaded engine instead.
"""

RECORDED = {
    "uci": {
        "title": "UCI Adult · income above 50K",
        "model": "XGBoost",
        "test_rows": 6513,
        "positive_rate": "about 24%",
        "runtime_laptop": "about 0.2 s",
        "fresh_test": {"batches": 114, "mae_pp": 0.708, "coverage": 0.974, "extent_accuracy": 0.974,
                       "interaction_auc": 0.779, "false_alarm_rate": 0.033},
        "benchmark_batches": 1600,
        "n1_mae": [["ATC", 3.375], ["CBPE", 2.436], ["Constant", 2.052], ["DoC", 1.846], ["C1 (N1)", 0.656]],
        "n1_headline": {"value": "81%", "text": "lower harm error than ATC",
                        "detail": "68% lower than a constant guess; 81% lower than ATC."},
        "n2_coverage": {"types": ["none", "single", "multi", "interaction", "combined"],
                        "global": [1.0, 0.95, 0.892, 0.982, 0.973],
                        "mondrian": [0.989, 0.949, 0.972, 0.945, 0.973], "overall": 0.957},
        "n3": {"extent_accuracy": 0.974, "labels": ["none", "single", "multi"],
               "matrix": [[0.992, 0.008, 0.0], [0.020, 0.978, 0.001], [0.0, 0.065, 0.935]]},
    },
    "ieee": {
        "title": "IEEE-CIS · card fraud",
        "model": "LightGBM",
        "test_rows": 40000,
        "positive_rate": "3.0% fraud",
        "runtime_laptop": "about 1.6 s",
        "fresh_test": {"batches": 114, "mae_pp": 0.788, "coverage": 0.974, "extent_accuracy": 0.982,
                       "interaction_auc": 0.696, "false_alarm_rate": 0.0},
        "benchmark_batches": 1600,
        "n1_mae": [["DoC", 1.049], ["ATC", 1.043], ["Constant", 1.008], ["CBPE", 0.687], ["C1 (N1)", 0.354]],
        "n1_headline": {"value": "66%", "text": "lower harm error than ATC",
                        "detail": "65% lower than a constant guess; 66% lower than ATC."},
        "n2_coverage": {"types": ["none", "single", "multi", "interaction", "combined"],
                        "global": [0.98, 0.954, 0.916, 0.969, 0.961],
                        "mondrian": [0.966, 0.956, 0.974, 0.957, 0.963], "overall": 0.959},
        "n3": {"extent_accuracy": 0.993, "labels": ["none", "single", "multi"], "matrix": None,
               "note": "F1 none 0.991, single 0.993, multi 0.996"},
    },
}
