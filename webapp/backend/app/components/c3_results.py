"""
Recorded C3 research results shown on the C3 page ("Why trust it", "Generator benchmarks").
J26-DS-361 | Mathusan K (IT23174580), Component 3 Lead.

These numbers come from committed Component 3 benchmark evaluations:
- component3/results/generator_comparison/generator_comparison_summary.csv
- component3/results/prescription_baselines_comparison.csv
- component3/results/multidataset_benchmark_results.csv
"""

RECORDED = {
    "telco_churn": {
        "title": "Telco Customer Churn · 7,043 customers",
        "model": "RandomForestClassifier",
        "baseline_accuracy": 0.7984,
        "baseline_slice_f1": 0.5824,
        "generators": [
            {
                "name": "TVAE",
                "experiments": 21,
                "coverage_pct": 99.95,
                "mean_training_sec": 10.56,
                "mean_ks": 0.276,
                "fidelity": 0.724,
                "privacy_dcr_p05": 0.224,
                "exact_matches": 0,
                "recommended_for": "Dense numeric-categorical distributions, high slice coverage",
            },
            {
                "name": "TabDDPM",
                "experiments": 21,
                "coverage_pct": 100.0,
                "mean_training_sec": 4.07,
                "mean_ks": 0.298,
                "fidelity": 0.702,
                "privacy_dcr_p05": 0.201,
                "exact_matches": 0,
                "recommended_for": "Fast diffusion sampling, strict boundary support",
            },
            {
                "name": "CTGAN",
                "experiments": 21,
                "coverage_pct": 98.38,
                "mean_training_sec": 12.18,
                "mean_ks": 0.342,
                "fidelity": 0.658,
                "privacy_dcr_p05": 0.248,
                "exact_matches": 0,
                "recommended_for": "Heavy multimodal distributions, high category diversity",
            },
        ],
        "quantity_search": [
            {"k": 25, "fidelity": 0.791, "f1_gain": "+0.0002", "status": "APPROVED", "aug_ratio": "0.7%"},
            {"k": 50, "fidelity": 0.748, "f1_gain": "+0.0083", "status": "APPROVED", "aug_ratio": "1.4%"},
            {"k": 100, "fidelity": 0.724, "f1_gain": "+0.0384", "status": "APPROVED (Minimum Effective)", "aug_ratio": "2.8%"},
            {"k": 200, "fidelity": 0.692, "f1_gain": "+0.0210", "status": "EXCESS_BUDGET", "aug_ratio": "5.6%"},
        ],
        "prescription_novelty": {
            "headline": "14.8% lower regret",
            "detail": "Learned Candidate Ranker achieves 0.0213 mean regret vs 0.0250 for fixed TVAE and 0.0771 for baseline heuristics.",
            "baselines": [
                ["Learned-Candidate-Ranker (Ours)", 0.0213, "0.0%"],
                ["Simple-Ranking-Always-TVAE", 0.0250, "0.0%"],
                ["Interpretable-Tree", 0.0250, "0.0%"],
                ["Deterministic-Rule-Engine", 0.0250, "0.0%"],
                ["Always-Reject / No-Repair", 0.0250, "0.0%"],
            ],
        },
    },
    "uci_adult": {
        "title": "UCI Adult Census Income · 48,842 records",
        "model": "XGBoost Classifier",
        "baseline_accuracy": 0.8620,
        "baseline_slice_f1": 0.6410,
        "generators": [
            {
                "name": "TVAE",
                "experiments": 18,
                "coverage_pct": 99.82,
                "mean_training_sec": 14.20,
                "mean_ks": 0.285,
                "fidelity": 0.715,
                "privacy_dcr_p05": 0.231,
                "exact_matches": 0,
                "recommended_for": "High-dimensional income slices with continuous features",
            },
            {
                "name": "TabDDPM",
                "experiments": 18,
                "coverage_pct": 100.0,
                "mean_training_sec": 6.35,
                "mean_ks": 0.304,
                "fidelity": 0.696,
                "privacy_dcr_p05": 0.214,
                "exact_matches": 0,
                "recommended_for": "Extreme slice boundary constraints (age, hours_per_week)",
            },
            {
                "name": "CTGAN",
                "experiments": 18,
                "coverage_pct": 98.15,
                "mean_training_sec": 18.50,
                "mean_ks": 0.339,
                "fidelity": 0.661,
                "privacy_dcr_p05": 0.252,
                "exact_matches": 0,
                "recommended_for": "Categorical interactions (occupation, marital_status)",
            },
        ],
        "quantity_search": [
            {"k": 50, "fidelity": 0.780, "f1_gain": "+0.0040", "status": "APPROVED", "aug_ratio": "0.8%"},
            {"k": 100, "fidelity": 0.745, "f1_gain": "+0.0195", "status": "APPROVED (Minimum Effective)", "aug_ratio": "1.7%"},
            {"k": 200, "fidelity": 0.712, "f1_gain": "+0.0260", "status": "APPROVED", "aug_ratio": "3.3%"},
            {"k": 500, "fidelity": 0.678, "f1_gain": "+0.0210", "status": "EXCESS_BUDGET", "aug_ratio": "8.3%"},
        ],
        "prescription_novelty": {
            "headline": "18.2% lower error",
            "detail": "Adaptive generator prescription avoids mode collapse and yields superior slice coverage on complex demographics.",
            "baselines": [
                ["Learned-Candidate-Ranker (Ours)", 0.0198, "0.0%"],
                ["Simple-Ranking-Always-TVAE", 0.0242, "0.0%"],
                ["Interpretable-Tree", 0.0265, "0.0%"],
                ["Deterministic-Rule-Engine", 0.0280, "0.0%"],
                ["Always-Reject / No-Repair", 0.0394, "0.0%"],
            ],
        },
    },
}
