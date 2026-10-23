"""
Recorded C2 research results shown on the C2 page ("Why trust it"). Owner: Tharmika N.
Source: c2/evaluation/results/*_summary.md (UCI Adult, 158 monitoring streams of 5 cycles, onset at
cycle 2, batch 1000 from test.csv, seeds 11/22/33). Re-run the evaluation scripts to update.
"""

RECORDED = {
    "n1": {
        "title": "Novelty 1 · Temporal SHAP monitoring",
        "headline": "0%", "headline_label": "false alarms on clean data (drift-only monitor: 10%)",
        "note": "Interaction drift changes how features relate while each looks normal alone. "
                "Persistence across runs keeps clean data silent.",
        "charts": [
            {"label": "Top-1 root cause on interaction drift (%)",
             "rows": [["Drift-only monitor", 50.0], ["SHAP snapshot", 16.7], ["C2 temporal SHAP", 91.7]]},
            {"label": "False alarms on clean data (%)",
             "rows": [["Drift-only monitor", 10.0], ["C2 temporal SHAP", 0.0]]},
        ],
    },
    "n2": {
        "title": "Novelty 2 · Drift Impact Score",
        "headline": "98.6%", "headline_label": "top-1 root cause (best single signal: 95.7%)",
        "note": "Each evidence source alone against the fused score, on 138 drifted streams.",
        "charts": [
            {"label": "Top-1 root cause (%)",
             "rows": [["SliceLine", 0.7], ["Error concentration", 39.1], ["Accuracy impact", 73.9],
                      ["Data drift", 95.7], ["Explanation change", 95.7], ["C2 fused score", 98.6]]},
        ],
    },
    "n3": {
        "title": "Novelty 3 · Hierarchical localization",
        "headline": "89%", "headline_label": "of the best achievable group F1 (SliceLine: 60%)",
        "note": "F1 against the rows the drift truly broke. The dashed line is the oracle: the best "
                "rule of this shape when the true answer is known.",
        "charts": [
            {"label": "Group F1 (%)", "ref": 20.2,
             "rows": [["Guide draft", 7.9], ["Level 2 only", 11.8], ["SliceLine", 12.1],
                      ["C2 hierarchical", 18.0]]},
        ],
    },
}
