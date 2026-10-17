"""
End-to-end check of the C2 engine outputs.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

For every scenario in the suite (plus clean streams): warm up on 2 clean cycles,
diagnose the 3 drifted cycles with C1 output from the team's shared v1.1 fake
builder, and check:
  - contract validity of every C2 -> C3 and C2 -> C4 output (validate_output)
  - root_cause_category accuracy, with and without C1 hints
  - diagnosis latency per cycle (proposal metric: bounded time per 6-hour cycle)

Run:   python3.11 -m c2.evaluation.evaluate_outputs       (~3 min)
Writes c2/evaluation/results/outputs_summary.md (committed).
"""
import time
import warnings

import numpy as np
import pandas as pd

from c2.src.config import RESULTS_DIR  # noqa: I001  (adds shared/ to the path first)
from contracts.c1_fake_output import build_fake_c1_output
from c2.src.outputs import (C3_SCHEMA, C4_SCHEMA, C2DiagnosisEngine,
                            root_cause_category, validate_output)
from c2.src.scenarios import make_temporal_stream, scenario_suite

warnings.filterwarnings("ignore", category=UserWarning)
TRUE_CATEGORY = {"marginal_single": "marginal", "interaction": "interaction",
                 "combined": "combined", "none": "unknown"}


def main():
    engine = C2DiagnosisEngine()
    ref = engine.a["ref_df"]
    specs = scenario_suite() + [("none", [], 0.0, None, "none")] * 4
    rows, t0 = [], time.time()
    for i, (dtype, feats, mag, pairs, group) in enumerate(specs):
        stream = make_temporal_stream(dtype, feats, mag, seed=11 + i, interaction_pairs=pairs)
        engine.warmup([s["drifted_df"] for s in stream[:2]])
        for s in stream[2:]:
            c1 = build_fake_c1_output(s["ground_truth"], seed=s["spec"]["seed"])
            t = time.time()
            out = engine.diagnose(s["drifted_df"], c1)
            latency = time.time() - t
        p3 = validate_output(out["to_c3"], C3_SCHEMA, ref)
        p4 = validate_output(out["to_c4"], C4_SCHEMA, ref)
        # category without C1 hints, from exactly the evidence the engine used
        ev = engine.last_evidence
        rows.append({
            "group": group, "drift_type": dtype, "true_category": TRUE_CATEGORY[dtype],
            "category": out["to_c3"]["root_cause_category"],
            "category_c2_only": root_cause_category(c1, ev["dis"], ev["shap"], use_c1=False),
            "c3_valid": not p3, "c4_valid": not p4, "problems": "; ".join(p3 + p4),
            "latency_s": latency, "stability": out["to_c3"]["diagnosis_stability"],
            "rule": out["to_c3"]["affected_slice_rule"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(specs)} streams  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    L = ["# C2 engine output check (contracts v1.1)\n",
         f"Streams: {len(df)} (scenario suite + clean), 2 warm-up + 3 diagnosed cycles each, "
         f"C1 input from shared/contracts/c1_fake_output.py.\n",
         "| Check | Result |", "|---|---|",
         f"| C2 -> C3 outputs valid | {df['c3_valid'].sum()}/{len(df)} |",
         f"| C2 -> C4 outputs valid | {df['c4_valid'].sum()}/{len(df)} |",
         f"| root_cause_category accuracy (C2 evidence only) | "
         f"{100 * (df['category_c2_only'] == df['true_category']).mean():.1f}% |",
         f"| root_cause_category accuracy (C2 + C1 hints) | "
         f"{100 * (df['category'] == df['true_category']).mean():.1f}% |",
         f"| Diagnosis latency per cycle, median | {df['latency_s'].median():.2f} s |",
         f"| Diagnosis latency per cycle, max | {df['latency_s'].max():.2f} s |",
         "\nThe fake C1 is built from the injection answer key, so the C2-only "
         "category accuracy is the honest figure until real C1 output is connected.\n",
         "\n### Category confusion (rows = truth, columns = C2 + C1 hints)\n"]
    ct = pd.crosstab(df["true_category"], df["category"])
    L.append("| truth \\ predicted | " + " | ".join(ct.columns) + " |")
    L.append("|---|" + "---|" * len(ct.columns))
    for idx, r in ct.iterrows():
        L.append(f"| {idx} | " + " | ".join(str(v) for v in r.values) + " |")
    bad = df[~(df["c3_valid"] & df["c4_valid"])]
    if len(bad):
        L.append("\n### Invalid outputs\n")
        L += [f"- {r.drift_type}: {r.problems}" for r in bad.itertuples()]
    text = "\n".join(L) + "\n"
    (RESULTS_DIR / "outputs_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()