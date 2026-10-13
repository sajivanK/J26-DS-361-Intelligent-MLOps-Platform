"""
Sprint 02 — Task 03: Comprehensive Model Performance Evaluation Under Drift.
Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix across all scenarios.
Run from repository root:
    .\.venv\Scripts\python.exe component3/src/experiments/evaluate_drift_performance.py
"""

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "component3/configs/drift_scenarios.json"
OUTPUT_DIR = ROOT / "component3/data/drifted"

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model = joblib.load(ROOT / config["model_path"])
reference_df = pd.read_csv(ROOT / config["reference_path"])
batch = reference_df.sample(
    n=config["batch_size"], random_state=config["seed"]
).reset_index(drop=True)
target_col = config.get("target_col", "target")


def evaluate(df):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba) if y_proba is not None else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


baseline = evaluate(batch)

print("=" * 75)
print("BASELINE PERFORMANCE (REFERENCE BATCH — 6,000 ROWS)")
print("=" * 75)
for metric, value in baseline.items():
    print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

print("\n" + "=" * 75)
print("EVALUATING DRIFTED DATASETS (S00 – S07)")
print("=" * 75)

records = []
for scenario in config["scenarios"]:
    scenario_id = scenario["id"]
    dataset_path = OUTPUT_DIR / f"{scenario_id}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    dataset = pd.read_csv(dataset_path)
    metrics = evaluate(dataset)
    record = {
        "scenario": scenario_id,
        "drift_type": scenario["drift_type"],
        "magnitude": scenario["magnitude"],
        **metrics,
    }

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        record[f"{metric}_drop_pp"] = (
            baseline[metric] - metrics[metric]
        ) * 100

    records.append(record)
    print(
        f"{scenario_id:4s} | "
        f"Accuracy={metrics['accuracy']:.4f} (drop {record['accuracy_drop_pp']:+.2f} pp) | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"AUC={metrics['roc_auc']:.4f}"
    )

results = pd.DataFrame(records)
output_path = OUTPUT_DIR / "drift_performance_metrics.csv"
results.to_csv(output_path, index=False)

print("\n" + "=" * 75)
print(f"Saved: {output_path.relative_to(ROOT)}")
print(f"Evaluated scenarios: {len(results)}")
print("=" * 75)
