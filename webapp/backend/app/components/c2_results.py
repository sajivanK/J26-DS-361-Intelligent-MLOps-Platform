"""
Recorded C2 research results shown on the C2 page ("Why trust it"). Owner: Tharmika N.
Source: c2/evaluation/results/*_summary.md (UCI Adult, monitoring streams of 5 cycles, onset at
cycle 2, batch 1000 from test.csv). Novelty 2 and 3 come from the label-free evaluations
(novelty2_free_summary.md, novelty3_free_summary.md): production batches carry no labels, the
answer key only grades. Re-run the evaluation scripts to update.
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
        "title": "Novelty 2 · Drift Impact Score (no labels)",
        "headline": "93%", "headline_label": "right column when a harmful and a harmless feature drift together (drift monitor: 67%)",
        "note": "Held-out seeds, no labels used. The harmless feature drifts too, so a plain drift "
                "monitor often blames it; C2 asks which change actually flips the model's decisions.",
        "charts": [
            {"label": "Top-1 root cause, harmful + harmless pairs (%)",
             "rows": [["SHAP explanation only", 33.3], ["Drift monitor", 66.7], ["C2 fused score", 93.3]]},
            {"label": "Blamed the harmless drifted feature (%)",
             "rows": [["SHAP explanation only", 46.7], ["Drift monitor", 33.3], ["C2 fused score", 6.7]]},
        ],
    },
    "n3": {
        "title": "Novelty 3 · Hierarchical localization (no labels)",
        "headline": "79%", "headline_label": "of the best achievable group F1, without labels (SliceLine, with labels: 60%)",
        "note": "F1 against the rows the drift truly broke (graded afterwards with the answer key). "
                "The dashed line is the oracle: the best rule of this shape when the true answer is known.",
        "charts": [
            {"label": "Group F1 (%)", "ref": 20.2,
             "rows": [["Shift region", 8.2], ["Level 2 only", 10.9], ["SliceLine (labels)", 12.1],
                      ["C2 hierarchical", 15.9]]},
        ],
    },
}
