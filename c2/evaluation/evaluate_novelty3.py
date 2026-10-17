"""
Novelty 3 evaluation: Hierarchical Localization vs existing slice-finding.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Full C2 pipeline on the same monitoring streams as Novelties 1 and 2:
    Temporal SHAP (N1) -> Drift Impact Score (N2) -> localization (N3)
judged on the last cycle.

Ground truth: the batch is rebuilt exactly as it was BEFORE drift injection
(same seed, same rows from test.csv). A row is truly drift-harmed when the model
was right on the clean row and is wrong on the drifted row.

Methods
  SliceLine   slice search on the current batch's errors, levels 1-2, no root cause
  Guide       C2 guide draft: worst-accuracy quartile of the root feature plus a
              median split / most common value of the second-ranked feature
  N3_level2   Novelty 3 stopped at level 2 (root feature value range only)
  N3          Novelty 3, full hierarchical rule (level 3 kept only if it helps)
  Oracle      upper bound: the same rule search given the TRUE harmed rows and
              the true root feature (best rule of this shape that is findable)
  (Guide and N3 use the root cause chosen by the Drift Impact Score.)

Metrics (streams with at least 10 truly harmed rows)
  precision   share of rows in the slice that were truly drift-harmed
  recall      share of truly drift-harmed rows inside the slice
  F1, support (share of the batch the rule selects)
  lift        precision / share of harmed rows in the whole batch
              (how many times more concentrated the harm is inside the slice)
  % of oracle mean F1 relative to the Oracle

Run:   python3.11 -m c2.evaluation.evaluate_novelty3            (full, ~8 min)
       python3.11 -m c2.evaluation.evaluate_novelty3 --quick    (~2 min)
Writes c2/evaluation/results/novelty3_runs.csv (git-ignored) and
       c2/evaluation/results/novelty3_summary.md (committed).
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from c2.src.config import RESULTS_DIR, TARGET_COL
from c2.src.drift_impact import DriftImpactScorer
from c2.src.localization import (HierarchicalLocalizer, guide_localization,
                                 sliceline_localization)
from c2.src.scenarios import artifacts, make_temporal_stream, scenario_suite
from c2.src.temporal_shap import TemporalSHAPMonitor

warnings.filterwarnings("ignore", category=UserWarning)

N_CYCLES, ONSET = 5, 2
METHODS = ["SliceLine", "Guide", "N3_level2", "N3", "Oracle"]
GROUPS = ["high_importance", "low_importance", "interaction", "combined"]
MIN_HARMED = 10


def clean_batch(a, scenario):
    """Rebuild the batch exactly as inject_drift drew it, before injection."""
    spec = scenario["spec"]
    rng = np.random.default_rng(int(spec["seed"]))
    n = min(spec["batch_size"], len(a["test_df"]))
    return a["test_df"].sample(n=n, replace=False,
                               random_state=int(rng.integers(0, 99999))).reset_index(drop=True)


def rule_mask(df, rule):
    if not rule:
        return np.ones(len(df), dtype=bool)
    try:
        idx = df.query(rule).index
    except Exception:
        return np.zeros(len(df), dtype=bool)
    m = np.zeros(len(df), dtype=bool)
    m[idx] = True
    return m


def scores(mask, truth):
    tp = int((mask & truth).sum())
    prec = tp / mask.sum() if mask.sum() else 0.0
    rec = tp / truth.sum() if truth.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, float(mask.mean())


def run_stream(spec, seed, a, mon, scorer, loc):
    dtype, feats, mag, pairs, group = spec
    stream = make_temporal_stream(dtype, feats, mag, n_cycles=N_CYCLES,
                                  onset=ONSET, seed=seed, interaction_pairs=pairs)
    last = stream[-1]
    cur = last["drifted_df"]
    X = cur.drop(columns=[TARGET_COL])
    y = cur[TARGET_COL].values
    clean = clean_batch(a, last)
    truth = ((a["model"].predict(clean.drop(columns=[TARGET_COL])) == y)
             & (a["model"].predict(X) != y))

    mon.reset()
    for s in stream:
        r = mon.update(s["drifted_df"])
    scorer.calibrate_performance([s["drifted_df"] for s in stream[:ONSET]])
    dis = scorer.score(cur, r)
    ranked = list(dis["drift_impact_scores"])
    root = ranked[0]
    n3 = loc.localize(cur, root, ranked)

    rules = {
        "SliceLine": sliceline_localization(cur, a["model"], a["feature_types"]),
        "Guide": guide_localization(cur, a["model"], root, ranked[1]),
        "N3_level2": n3.get("level2_rule", n3["affected_slice_rule"]),
        "N3": n3["affected_slice_rule"],
        "Oracle": loc.localize(cur, last["true_causes"][0], harm=truth.astype(float)
                               )["affected_slice_rule"] if truth.sum() >= 3 else "",
    }
    row = {"group": group, "drift_type": dtype,
           "features": "+".join(last["true_causes"]), "magnitude": mag, "seed": seed,
           "harm_pp": last["ground_truth"]["actual_accuracy_drop"],
           "true_harmed_rows": int(truth.sum()), "root_pred": root,
           "root_correct": root in last["true_causes"],
           "N3_levels": n3["levels_used"]}
    base_rate = max(truth.mean(), 1e-9)
    for m, rule in rules.items():
        p, rc, f1, sup = scores(rule_mask(cur, rule), truth)
        row.update({f"{m}_precision": p, f"{m}_recall": rc, f"{m}_f1": f1,
                    f"{m}_support": sup, f"{m}_lift": p / base_rate, f"{m}_rule": rule})
    return row


def summarise(df):
    ev = df[df["true_harmed_rows"] >= MIN_HARMED]
    L = [f"Evaluated on {len(ev)} of {len(df)} drifted streams with at least "
         f"{MIN_HARMED} truly drift-harmed rows. Root cause from the Drift Impact "
         f"Score was correct on {100 * ev['root_correct'].mean():.1f}% of them. "
         f"Median share of harmed rows per batch: "
         f"{100 * (ev['true_harmed_rows'] / 1000).median():.1f}%. Lift is the median "
         f"over streams.\n"]

    def table(title, frame):
        L.append(f"\n### {title}\n")
        L.append("| Method | Precision | Recall | F1 | Support | Lift | % of oracle F1 |")
        L.append("|---|---|---|---|---|---|---|")
        oracle = frame["Oracle_f1"].mean()
        for m in METHODS:
            f1 = frame[f"{m}_f1"].mean()
            L.append(f"| {m} | {100 * frame[f'{m}_precision'].mean():.1f}% | "
                     f"{100 * frame[f'{m}_recall'].mean():.1f}% | "
                     f"{100 * f1:.1f}% | "
                     f"{100 * frame[f'{m}_support'].mean():.1f}% | "
                     f"{frame[f'{m}_lift'].median():.1f}x | "
                     f"{100 * f1 / max(oracle, 1e-9):.0f}% |")

    table(f"All evaluated streams (n={len(ev)})", ev)
    for g in GROUPS:
        sub = ev[ev["group"] == g]
        if len(sub):
            table(f"{g} (n={len(sub)})", sub)

    L.append("\n### Novelty 3 depth\n")
    for k, v in ev["N3_levels"].value_counts().sort_index().items():
        L.append(f"- level {k}: {v} streams")
    L.append("\n### Example rules (first stream per group)\n")
    for g in GROUPS:
        sub = ev[ev["group"] == g]
        if len(sub):
            r = sub.iloc[0]
            L.append(f"- {g} ({r['features']}, m={r['magnitude']}): "
                     f"N3 `{r['N3_rule'] or '(none)'}` vs SliceLine `{r['SliceLine_rule']}`")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [11] if args.quick else [11, 22, 33]

    a = artifacts()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    scorer = DriftImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)
    loc = HierarchicalLocalizer(a["model"], a["ref_df"], a["feature_types"])
    specs = [s for s in scenario_suite() if s[0] != "none"]
    jobs = [(s, seed) for s in specs for seed in seeds]

    rows, t0 = [], time.time()
    for i, (spec, seed) in enumerate(jobs, 1):
        rows.append(run_stream(spec, seed, a, mon, scorer, loc))
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} streams  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "novelty3_runs.csv", index=False)
    header = (f"# Novelty 3 results: Hierarchical Localization vs slice finding\n\n"
              f"Full pipeline N1 -> N2 -> N3 on {len(df)} drifted streams "
              f"({N_CYCLES} cycles, onset at cycle {ONSET}, batch 1000 from test.csv, "
              f"seeds {seeds}). Truth = rows correct before injection and wrong after.\n\n")
    text = header + summarise(df) + "\n"
    (RESULTS_DIR / "novelty3_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()