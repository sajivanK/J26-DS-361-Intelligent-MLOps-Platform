"""
Novelty 3 evaluation, label-free: does C2 find the responsible value range without labels?
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Full label-free C2 pipeline on the same monitoring streams as the other C2
evaluations (5 cycles, drift onset at cycle 2, batch 1000 from unseen test.csv):
    Temporal SHAP (N1) -> label-free Drift Impact Score (N2) -> localization (N3)
Production batches carry NO labels. Labels and the pre-injection batch are used
only afterwards, to grade.

Label-free localization target ("rows at risk", C1 label-free plan, 23 Oct 2026):
for every row, how far the model's predicted probability moves when the root
feature is restored to its normal distribution (rank-preserving restore). The
hierarchical search itself is unchanged (feature -> value range -> one extra
condition, kept only if it adds >= 0.02 F1; no group under 20 rows).
The default target (probability movement) was fixed before this run; the
decision-flip target is reported as an alternative, not used to pick a winner.

Two kinds of truth, both from the injection answer key:
  harmed rows    the model was right on the clean row and is wrong on the drifted
                 row (the same truth as the labelled Novelty 3 evaluation)
  changed rows   rows whose drifted feature value(s) differ from the clean batch
                 ("did C2 find where the drift is?")

Methods
  label-free (usable in production)
    ShiftRegion      range of the root feature that gained the most share against
                     the reference data (no model, no labels)
    N3_free_level2   Novelty 3 on rows at risk, stopped at level 2
    N3_free          Novelty 3 on rows at risk, full rule (default)
    N3_free_flips    alternative target: rows whose decision flips
  need labels (reference only, not usable in production)
    SliceLine, Guide, N3_labelled (your labelled Novelty 3), Oracle (best rule of
    this shape given the true harmed rows)

Metrics, per stream with at least 10 truly harmed rows: precision, recall, F1,
support and lift against harmed rows; precision, recall and F1 against changed
rows. Seeds 11 and 22 = development, 33 = held out.

Run:   python3.11 -m c2.evaluation.evaluate_novelty3_free            (full, ~25 min)
       python3.11 -m c2.evaluation.evaluate_novelty3_free --quick    (~5 min)
Writes c2/evaluation/results/novelty3_free_runs.csv (git-ignored) and
       c2/evaluation/results/novelty3_free_summary.md (committed).
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from c2.evaluation.evaluate_novelty3 import clean_batch, rule_mask, scores
from c2.src.config import RESULTS_DIR, TARGET_COL
from c2.src.drift_impact import DriftImpactScorer
from c2.src.drift_impact_free import LabelFreeImpactScorer
from c2.src.localization import (HierarchicalLocalizer, guide_localization,
                                 shift_region_localization, sliceline_localization)
from c2.src.scenarios import artifacts, make_temporal_stream, scenario_suite
from c2.src.temporal_shap import TemporalSHAPMonitor

warnings.filterwarnings("ignore", category=UserWarning)

N_CYCLES, ONSET = 5, 2
FREE = ["ShiftRegion", "N3_free_level2", "N3_free", "N3_free_flips"]
LABELLED = ["SliceLine", "Guide", "N3_labelled", "Oracle"]
METHODS = FREE + LABELLED
GROUPS = ["high_importance", "low_importance", "interaction", "combined"]
MIN_HARMED = 10


def run_stream(spec, seed, a, mon, free, labelled, loc):
    dtype, feats, mag, pairs, group = spec
    stream = make_temporal_stream(dtype, feats, mag, n_cycles=N_CYCLES,
                                  onset=ONSET, seed=seed, interaction_pairs=pairs)
    last = stream[-1]
    cur = last["drifted_df"]
    X, y = cur.drop(columns=[TARGET_COL]), cur[TARGET_COL].values
    unlabelled = [s["drifted_df"].drop(columns=[TARGET_COL]) for s in stream]

    # ── grading truth (answer key, never given to C2) ──────────────────────
    clean = clean_batch(a, last)
    harmed = ((a["model"].predict(clean.drop(columns=[TARGET_COL])) == y)
              & (a["model"].predict(X) != y))
    changed = np.zeros(len(cur), dtype=bool)
    for f in last["true_causes"]:
        changed |= ~np.isclose(clean[f].to_numpy(float), cur[f].to_numpy(float))

    # ── label-free C2 ──────────────────────────────────────────────────────
    mon.reset()
    for b in unlabelled:
        r = mon.update(b)
    free.calibrate(unlabelled[:ONSET])
    dis = free.score(unlabelled[-1], r)
    ranked = list(dis["drift_impact_scores"])
    root = ranked[0]
    moves, flips = free.rows_at_risk(unlabelled[-1], root)
    n3 = loc.localize(unlabelled[-1], root, ranked, harm=moves)
    n3_flip = loc.localize(unlabelled[-1], root, ranked, harm=flips)

    # ── labelled references (not usable in production) ──────────────────────
    labelled.calibrate_performance([s["drifted_df"] for s in stream[:ONSET]])
    lab = labelled.score(cur, r)
    lab_ranked = list(lab["drift_impact_scores"])
    n3_lab = loc.localize(cur, lab_ranked[0], lab_ranked)

    rules = {
        "ShiftRegion": shift_region_localization(unlabelled[-1], a["ref_df"], root, a["feature_types"]),
        "N3_free_level2": n3.get("level2_rule", n3["affected_slice_rule"]),
        "N3_free": n3["affected_slice_rule"],
        "N3_free_flips": n3_flip["affected_slice_rule"],
        "SliceLine": sliceline_localization(cur, a["model"], a["feature_types"]),
        "Guide": guide_localization(cur, a["model"], lab_ranked[0], lab_ranked[1]),
        "N3_labelled": n3_lab["affected_slice_rule"],
        "Oracle": loc.localize(cur, last["true_causes"][0], harm=harmed.astype(float)
                               )["affected_slice_rule"] if harmed.sum() >= 3 else "",
    }
    row = {"group": group, "drift_type": dtype, "features": "+".join(last["true_causes"]),
           "magnitude": mag, "seed": seed, "split": "held-out" if seed == 33 else "development",
           "harm_pp": last["ground_truth"]["actual_accuracy_drop"],
           "true_harmed_rows": int(harmed.sum()), "changed_rows": int(changed.sum()),
           "root_pred": root, "root_correct": root in last["true_causes"],
           "N3_free_levels": n3["levels_used"],
           "risk_rows_flipped": int(round(flips.sum()))}
    base_rate = max(harmed.mean(), 1e-9)
    for m, rule in rules.items():
        mask = rule_mask(cur, rule)
        p, rc, f1, sup = scores(mask, harmed)
        pc, rcc, f1c, _ = scores(mask, changed)
        row.update({f"{m}_precision": p, f"{m}_recall": rc, f"{m}_f1": f1, f"{m}_support": sup,
                    f"{m}_lift": p / base_rate, f"{m}_chg_precision": pc, f"{m}_chg_recall": rcc,
                    f"{m}_chg_f1": f1c, f"{m}_rule": rule})
    return row


def pct(v):
    return "n/a" if pd.isna(v) else f"{100 * v:.1f}%"


def summarise(df):
    ev = df[df["true_harmed_rows"] >= MIN_HARMED]
    L = [f"Evaluated on {len(ev)} of {len(df)} drifted streams with at least {MIN_HARMED} truly "
         f"drift-harmed rows. The label-free root cause was correct on "
         f"{100 * ev['root_correct'].mean():.1f}% of them. Median share of harmed rows per batch: "
         f"{100 * (ev['true_harmed_rows'] / 1000).median():.1f}%; median share of changed rows: "
         f"{100 * (ev['changed_rows'] / 1000).median():.1f}%. Lift is the median over streams.\n"]

    def table(title, frame):
        L.append(f"\n### {title}\n")
        L.append("| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | "
                 "% of oracle F1 | vs changed rows: precision | recall | F1 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        oracle = frame["Oracle_f1"].mean()
        for m in METHODS:
            L.append(f"| {m} | {'no' if m in FREE else 'yes'} | {pct(frame[f'{m}_precision'].mean())} | "
                     f"{pct(frame[f'{m}_recall'].mean())} | {pct(frame[f'{m}_f1'].mean())} | "
                     f"{pct(frame[f'{m}_support'].mean())} | {frame[f'{m}_lift'].median():.1f}x | "
                     f"{100 * frame[f'{m}_f1'].mean() / max(oracle, 1e-9):.0f}% | "
                     f"{pct(frame[f'{m}_chg_precision'].mean())} | {pct(frame[f'{m}_chg_recall'].mean())} | "
                     f"{pct(frame[f'{m}_chg_f1'].mean())} |")

    table(f"All evaluated streams (n={len(ev)})", ev)
    held = ev[ev["split"] == "held-out"]
    table(f"Held-out seed 33 (n={len(held)})", held)
    for g in GROUPS:
        sub = ev[ev["group"] == g]
        if len(sub):
            table(f"{g} (n={len(sub)})", sub)
    lv = ev["N3_free_levels"].value_counts().sort_index()
    L.append("\n### Label-free Novelty 3 depth\n")
    L += [f"- level {k}: {v} streams" for k, v in lv.items()]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [11, 33] if args.quick else [11, 22, 33]

    a = artifacts()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    free = LabelFreeImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)
    labelled = DriftImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)
    loc = HierarchicalLocalizer(a["model"], a["ref_df"], a["feature_types"])
    specs = [s for s in scenario_suite() if s[0] != "none"]
    jobs = [(s, seed) for s in specs for seed in seeds]

    rows, t0 = [], time.time()
    for i, (spec, seed) in enumerate(jobs, 1):
        rows.append(run_stream(spec, seed, a, mon, free, labelled, loc))
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} streams  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "novelty3_free_runs.csv", index=False)
    text = ("# Novelty 3 without labels: finding the responsible value range\n\n"
            f"Full label-free pipeline N1 -> N2 -> N3 on {len(df)} drifted streams ({N_CYCLES} cycles, "
            f"onset at cycle {ONSET}, batch 1000 from test.csv, seeds {seeds}). Production batches carry "
            "no labels; the answer key grades afterwards. Truth 1, harmed rows: right before injection, "
            "wrong after. Truth 2, changed rows: the drifted feature's value differs from the clean "
            "batch.\n\n" + summarise(df) + "\n")
    (RESULTS_DIR / "novelty3_free_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
