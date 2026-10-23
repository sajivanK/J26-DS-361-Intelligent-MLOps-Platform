"""
Quick check that the team's inject_drift works on this machine (not committed).
Run from the repo root:  python3.11 check_injection.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from c2.src.config import load_artifacts, TARGET_COL
from injection.inject_drift import inject_drift

a = load_artifacts()
ref, test, model, types = a["ref_df"], a["test_df"], a["model"], a["feature_types"]
feats = a["feature_names"]
print(f"reference {ref.shape}, test {test.shape}, features {len(feats)}\n")

cases = [
    ("none",            [],                            None),
    ("marginal_single", ["capital_gain"],              None),
    ("marginal_single", ["marital_status"],            None),
    ("marginal_multi",  ["age", "hours_per_week"],     None),
    ("interaction",     ["age", "marital_status"],     None),
    ("combined",        ["capital_gain"],              [("age", "marital_status")]),
]
# With the same seed and source_df the injector picks the same test rows, so a "none"
# run gives the exact batch before injection, to compare against.
before = inject_drift(reference_df=ref, model=model, drift_type="none", features=[], magnitude=0.0,
                      feature_types=types, batch_size=1000, seed=42, source_df=test, paired=True)["drifted_df"]
ok = True
for dtype, fs, pairs in cases:
    r = inject_drift(reference_df=ref, model=model, drift_type=dtype, features=fs,
                     magnitude=0.6, feature_types=types, interaction_pairs=pairs,
                     batch_size=1000, seed=42, source_df=test, paired=True)
    d, gt = r["drifted_df"], r["ground_truth"]
    changed = [f for f in feats if not np.allclose(before[f].values, d[f].values)]
    labels_kept = bool((before[TARGET_COL].values == d[TARGET_COL].values).all())
    expected = set(fs) | {x for p in (pairs or []) for x in p}
    good = len(d) == 1000 and labels_kept and set(changed) <= expected and (dtype == "none") == (changed == [])
    ok &= good
    print(f"{dtype:<16} {'+'.join(fs) or '-':<22} rows={len(d)}  accuracy drop={gt['actual_accuracy_drop']:6.2f} pp  "
          f"malignant={gt['is_malignant']!s:<5}  columns changed={changed}  labels kept={labels_kept}  {'OK' if good else 'CHECK'}")
print("\nINJECTION OK" if ok else "\nSOMETHING LOOKS WRONG: send me this output")