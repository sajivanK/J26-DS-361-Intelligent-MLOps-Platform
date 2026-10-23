"""Reproducible Multi-Dataset Drift-Repair Benchmark for Component 3.

Executes a comprehensive, fair, and resource-bounded experimental matrix across:
- 3 Tabular Datasets: Telco Customer Churn, UCI Adult Full, Bank Marketing
- 6 Drift Conditions: Numerical shift, Categorical shift, Prior shift, Localized slice, Concept drift
- 9 Repair Strategies: NO_REPAIR, Authentic Adaptation (B=50, 100, 200), Authentic ROS,
  Authentic Weighted Adaptation, TVAE, CTGAN, TabDDPM, Weighted Hybrid, Targeted Repair
- Quantity Search: K in [25, 50, 100, 200]

Outputs:
- component3/results/multidataset_benchmark_results.json
- component3/results/multidataset_benchmark_results.csv
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from scipy.spatial.distance import cdist

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.drift.drift_config import DriftCategory, get_drift_registry
from component3.src.drift.drift_injector import DriftInjector
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.tvae_generator import TVAEGenerator
from component3.src.models.model_registry import get_model_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.prescription.prescription_models import RepairStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "component3" / "results"
SPLITS_DIR = RESULTS_DIR / "adaptation_splits"


def compute_paired_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_cand: np.ndarray,
    y_pred_ref: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calculate paired bootstrap confidence intervals for positive F1 difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        f1_c = f1_score(y_true[idx], y_pred_cand[idx], pos_label=1, zero_division=0)
        f1_r = f1_score(y_true[idx], y_pred_ref[idx], pos_label=1, zero_division=0)
        diffs[b] = f1_c - f1_r

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


def compute_dcr_privacy_metric(syn_X: np.ndarray, real_X: np.ndarray) -> float:
    """Compute 5th percentile Distance to Closest Record (DCR) as privacy risk indicator."""
    if len(syn_X) == 0 or len(real_X) == 0:
        return 0.0
    # Subsample for bounded CPU computation
    sub_syn = syn_X[:min(len(syn_X), 100)]
    sub_real = real_X[:min(len(real_X), 500)]
    dists = cdist(sub_syn, sub_real, metric="euclidean")
    min_dists = dists.min(axis=1)
    return round(float(np.percentile(min_dists, 5)), 4)


@dataclass
class BenchmarkExperimentRecord:
    """Standardized record of a single benchmark experiment arm."""
    experiment_id: str
    dataset_id: str
    scenario_id: str
    drift_category: str
    strategy: str
    generator: str | None
    adaptation_budget: int
    synthetic_quantity: int
    sample_weight_policy: dict[str, float]
    # Metrics
    baseline_target_f1: float
    candidate_target_f1: float
    delta_f1_vs_baseline: float
    matched_control_f1: float
    delta_f1_vs_control: float
    baseline_source_f1: float
    candidate_source_f1: float
    source_retention_delta: float
    source_retention_preserved: bool
    bootstrap_ci_95: tuple[float, float]
    dcr_privacy_5th_pct: float | None
    training_duration_seconds: float
    # Decision
    is_acceptable: bool
    package_status: str
    rejection_reasons: list[str]
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiDatasetBenchmarkRunner:
    """Orchestrates the multi-dataset drift repair scientific benchmark."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model_reg = get_model_registry()
        self.dataset_reg = get_dataset_registry()
        self.drift_reg = get_drift_registry()
        self.injector = DriftInjector(seed=seed)

    def run_benchmark(self) -> list[BenchmarkExperimentRecord]:
        """Execute the multi-dataset benchmark matrix."""
        records: list[BenchmarkExperimentRecord] = []

        # Benchmark dataset configurations
        dataset_configs = [
            {
                "dataset_id": "telco_churn",
                "model_id": "telco_churn_random_forest",
                "target_col": "Churn",
                "scenarios": ["TELCO_NUM", "TELCO_CAT", "TELCO_CONCEPT"],
                "source_eval_path": SPLITS_DIR / "telco_churn_source_eval.csv" if (SPLITS_DIR / "telco_churn_source_eval.csv").is_file() else None,
                "adapt_pool_path": SPLITS_DIR / "telco_churn_target_adapt_pool.csv" if (SPLITS_DIR / "telco_churn_target_adapt_pool.csv").is_file() else None,
            },
            {
                "dataset_id": "uci_adult_full",
                "model_id": "uci_adult_full_random_forest",
                "target_col": "target",
                "scenarios": ["ADULT_NUM", "ADULT_PRIOR", "ADULT_CONCEPT"],
                "source_eval_path": SPLITS_DIR / "uci_adult_full_source_eval.csv",
                "adapt_pool_path": SPLITS_DIR / "uci_adult_full_target_adapt_pool.csv",
            },
            {
                "dataset_id": "bank_marketing",
                "model_id": "bank_marketing_random_forest",
                "target_col": "y",
                "scenarios": ["BANK_NUM", "BANK_CAT", "BANK_CONCEPT"],
                "source_eval_path": SPLITS_DIR / "bank_marketing_source_eval.csv",
                "adapt_pool_path": SPLITS_DIR / "bank_marketing_target_adapt_pool.csv",
            },
        ]

        exp_counter = 1

        for ds_cfg in dataset_configs:
            ds_id = ds_cfg["dataset_id"]
            model_id = ds_cfg["model_id"]
            target_col = ds_cfg["target_col"]
            logger.info("Starting Multi-Dataset Benchmark for: %s", ds_id)

            # Load baseline model and preprocessor
            base_model = self.model_reg.load_model(model_id)
            preprocessor = self.model_reg.load_preprocessor(model_id)
            metadata = self.dataset_reg.get(ds_id)

            # Load or partition data
            if ds_cfg["source_eval_path"] and Path(ds_cfg["source_eval_path"]).is_file():
                df_source_eval = pd.read_csv(ds_cfg["source_eval_path"])
                df_source_train = pd.read_csv(SPLITS_DIR / f"{ds_id}_source_train.csv")
                df_adapt_pool = pd.read_csv(ds_cfg["adapt_pool_path"])
            else:
                # Fallback to direct raw split
                raw_df = self.dataset_reg.load_dataset(ds_id)
                n_total = len(raw_df)
                df_source_train = raw_df.iloc[: int(0.7 * n_total)].copy().reset_index(drop=True)
                df_source_eval = raw_df.iloc[int(0.7 * n_total) : int(0.8 * n_total)].copy().reset_index(drop=True)
                df_adapt_pool = raw_df.iloc[int(0.8 * n_total) : int(0.9 * n_total)].copy().reset_index(drop=True)

            # Preprocess source eval
            bundle_seval = preprocessor.transform(df_source_eval)
            base_pred_seval = base_model.predict(bundle_seval.X)
            base_source_f1 = float(f1_score(bundle_seval.y, base_pred_seval, pos_label=1, zero_division=0))

            # Run each scenario
            for sc_id in ds_cfg["scenarios"]:
                logger.info("  Running Scenario: %s", sc_id)
                sc_cfg = self.drift_reg.get(sc_id)

                # Inject drift into adaptation pool to construct authentic post-drift adaptation & dev_val
                drift_res = self.injector.inject(df_adapt_pool, sc_cfg, metadata=metadata)
                drifted_pool = drift_res.drifted_df

                # Partition drifted pool into adaptation subset (B=200 max) and dev_val (clean evaluation target)
                n_adapt_max = min(200, len(drifted_pool) // 2)
                df_adapt_b200 = drifted_pool.iloc[:n_adapt_max].copy().reset_index(drop=True)
                df_adapt_b100 = df_adapt_b200.iloc[: min(100, len(df_adapt_b200))].copy().reset_index(drop=True)
                df_adapt_b50 = df_adapt_b200.iloc[: min(50, len(df_adapt_b200))].copy().reset_index(drop=True)
                df_dev_val = drifted_pool.iloc[n_adapt_max:].copy().reset_index(drop=True)

                # Baseline performance on drifted dev_val
                bundle_dev = preprocessor.transform(df_dev_val)
                base_pred_dev = base_model.predict(bundle_dev.X)
                base_dev_f1 = float(f1_score(bundle_dev.y, base_pred_dev, pos_label=1, zero_division=0))

                # -------------------------------------------------------------
                # ARM 0: NO_REPAIR (Control)
                # -------------------------------------------------------------
                rec_no_repair = BenchmarkExperimentRecord(
                    experiment_id=f"EXP_{exp_counter:04d}",
                    dataset_id=ds_id,
                    scenario_id=sc_id,
                    drift_category=sc_cfg.category.value,
                    strategy=RepairStrategy.NO_REPAIR.value,
                    generator=None,
                    adaptation_budget=0,
                    synthetic_quantity=0,
                    sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                    baseline_target_f1=base_dev_f1,
                    candidate_target_f1=base_dev_f1,
                    delta_f1_vs_baseline=0.0,
                    matched_control_f1=base_dev_f1,
                    delta_f1_vs_control=0.0,
                    baseline_source_f1=base_source_f1,
                    candidate_source_f1=base_source_f1,
                    source_retention_delta=0.0,
                    source_retention_preserved=True,
                    bootstrap_ci_95=(0.0, 0.0),
                    dcr_privacy_5th_pct=None,
                    training_duration_seconds=0.0,
                    is_acceptable=False,
                    package_status="REJECTED",
                    rejection_reasons=["Baseline control: no repair applied."],
                )
                records.append(rec_no_repair)
                exp_counter += 1

                # -------------------------------------------------------------
                # ARM 1: AUTHENTIC_ADAPTATION (B=50, 100, 200)
                # -------------------------------------------------------------
                ctrl_f1_by_budget: dict[int, float] = {}

                for b_size, df_adapt in [(50, df_adapt_b50), (100, df_adapt_b100), (200, df_adapt_b200)]:
                    t0 = time.perf_counter()
                    # Train adapted candidate
                    bundle_src = preprocessor.transform(df_source_train)
                    bundle_ad = preprocessor.transform(df_adapt)
                    X_tr = np.vstack([bundle_src.X, bundle_ad.X])
                    y_tr = np.concatenate([bundle_src.y, bundle_ad.y])

                    cand = clone(base_model)
                    cand.fit(X_tr, y_tr)
                    dur = round(time.perf_counter() - t0, 3)

                    cand_dev_f1 = float(f1_score(bundle_dev.y, cand.predict(bundle_dev.X), pos_label=1, zero_division=0))
                    cand_src_f1 = float(f1_score(bundle_seval.y, cand.predict(bundle_seval.X), pos_label=1, zero_division=0))
                    delta_dev = round(cand_dev_f1 - base_dev_f1, 4)
                    delta_src = round(cand_src_f1 - base_source_f1, 4)
                    ci_95 = compute_paired_bootstrap_ci(bundle_dev.y, cand.predict(bundle_dev.X), base_pred_dev)

                    ctrl_f1_by_budget[b_size] = cand_dev_f1

                    reasons = []
                    is_acc = True
                    if delta_src < -0.05:
                        is_acc = False
                        reasons.append("Source retention degraded.")
                    if delta_dev < 0.01:
                        is_acc = False
                        reasons.append("Utility gain below practical margin (+0.01).")
                    if ci_95[1] <= 0:
                        is_acc = False
                        reasons.append("Bootstrap 95% CI upper bound <= 0.")

                    rec_auth = BenchmarkExperimentRecord(
                        experiment_id=f"EXP_{exp_counter:04d}",
                        dataset_id=ds_id,
                        scenario_id=sc_id,
                        drift_category=sc_cfg.category.value,
                        strategy=RepairStrategy.AUTHENTIC_ONLY_ADAPTATION.value,
                        generator=None,
                        adaptation_budget=b_size,
                        synthetic_quantity=0,
                        sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                        baseline_target_f1=base_dev_f1,
                        candidate_target_f1=cand_dev_f1,
                        delta_f1_vs_baseline=delta_dev,
                        matched_control_f1=cand_dev_f1,
                        delta_f1_vs_control=0.0,
                        baseline_source_f1=base_source_f1,
                        candidate_source_f1=cand_src_f1,
                        source_retention_delta=delta_src,
                        source_retention_preserved=(delta_src >= -0.05),
                        bootstrap_ci_95=ci_95,
                        dcr_privacy_5th_pct=None,
                        training_duration_seconds=dur,
                        is_acceptable=is_acc,
                        package_status="ACCEPTED" if is_acc else "REJECTED",
                        rejection_reasons=reasons,
                    )
                    records.append(rec_auth)
                    exp_counter += 1

                # -------------------------------------------------------------
                # ARM 2: AUTHENTIC_OVERSAMPLING (B=50)
                # -------------------------------------------------------------
                t0 = time.perf_counter()
                class_counts = df_adapt_b50[target_col].value_counts()
                max_c = class_counts.max()
                resampled = []
                rng = np.random.default_rng(42)
                for c_val, cnt in class_counts.items():
                    c_df = df_adapt_b50[df_adapt_b50[target_col] == c_val]
                    if cnt < max_c:
                        idx = rng.choice(len(c_df), size=(max_c - cnt), replace=True)
                        resampled.append(c_df.iloc[idx])
                df_ros = pd.concat([df_adapt_b50] + resampled, ignore_index=True) if resampled else df_adapt_b50

                bundle_ros = preprocessor.transform(df_ros)
                X_tr = np.vstack([bundle_src.X, bundle_ros.X])
                y_tr = np.concatenate([bundle_src.y, bundle_ros.y])

                cand = clone(base_model)
                cand.fit(X_tr, y_tr)
                dur = round(time.perf_counter() - t0, 3)

                cand_dev_f1 = float(f1_score(bundle_dev.y, cand.predict(bundle_dev.X), pos_label=1, zero_division=0))
                cand_src_f1 = float(f1_score(bundle_seval.y, cand.predict(bundle_seval.X), pos_label=1, zero_division=0))
                delta_dev = round(cand_dev_f1 - base_dev_f1, 4)
                delta_src = round(cand_src_f1 - base_source_f1, 4)
                ci_95 = compute_paired_bootstrap_ci(bundle_dev.y, cand.predict(bundle_dev.X), base_pred_dev)
                matched_ctrl = ctrl_f1_by_budget.get(50, base_dev_f1)

                reasons = []
                is_acc = True
                if delta_src < -0.05:
                    is_acc = False
                    reasons.append("Source retention degraded.")
                if delta_dev < 0.01:
                    is_acc = False
                    reasons.append("Utility gain below practical margin (+0.01).")
                if ci_95[1] <= 0:
                    is_acc = False
                    reasons.append("Bootstrap 95% CI upper bound <= 0.")

                rec_ros = BenchmarkExperimentRecord(
                    experiment_id=f"EXP_{exp_counter:04d}",
                    dataset_id=ds_id,
                    scenario_id=sc_id,
                    drift_category=sc_cfg.category.value,
                    strategy=RepairStrategy.AUTHENTIC_OVERSAMPLING.value,
                    generator=None,
                    adaptation_budget=50,
                    synthetic_quantity=0,
                    sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                    baseline_target_f1=base_dev_f1,
                    candidate_target_f1=cand_dev_f1,
                    delta_f1_vs_baseline=delta_dev,
                    matched_control_f1=matched_ctrl,
                    delta_f1_vs_control=round(cand_dev_f1 - matched_ctrl, 4),
                    baseline_source_f1=base_source_f1,
                    candidate_source_f1=cand_src_f1,
                    source_retention_delta=delta_src,
                    source_retention_preserved=(delta_src >= -0.05),
                    bootstrap_ci_95=ci_95,
                    dcr_privacy_5th_pct=None,
                    training_duration_seconds=dur,
                    is_acceptable=is_acc,
                    package_status="ACCEPTED" if is_acc else "REJECTED",
                    rejection_reasons=reasons,
                )
                records.append(rec_ros)
                exp_counter += 1

                # -------------------------------------------------------------
                # ARM 3: AUTHENTIC_WEIGHTED_ADAPTATION (B=50)
                # -------------------------------------------------------------
                t0 = time.perf_counter()
                w_tgt = 2.0
                bundle_ad50 = preprocessor.transform(df_adapt_b50)
                X_tr = np.vstack([bundle_src.X, bundle_ad50.X])
                y_tr = np.concatenate([bundle_src.y, bundle_ad50.y])
                w_tr = np.concatenate([np.ones(len(bundle_src.y)), np.full(len(bundle_ad50.y), w_tgt)])

                cand = clone(base_model)
                cand.fit(X_tr, y_tr, sample_weight=w_tr)
                dur = round(time.perf_counter() - t0, 3)

                cand_dev_f1 = float(f1_score(bundle_dev.y, cand.predict(bundle_dev.X), pos_label=1, zero_division=0))
                cand_src_f1 = float(f1_score(bundle_seval.y, cand.predict(bundle_seval.X), pos_label=1, zero_division=0))
                delta_dev = round(cand_dev_f1 - base_dev_f1, 4)
                delta_src = round(cand_src_f1 - base_source_f1, 4)
                ci_95 = compute_paired_bootstrap_ci(bundle_dev.y, cand.predict(bundle_dev.X), base_pred_dev)

                reasons = []
                is_acc = True
                if delta_src < -0.05:
                    is_acc = False
                    reasons.append("Source retention degraded.")
                if delta_dev < 0.01:
                    is_acc = False
                    reasons.append("Utility gain below practical margin (+0.01).")
                if ci_95[1] <= 0:
                    is_acc = False
                    reasons.append("Bootstrap 95% CI upper bound <= 0.")

                rec_wt = BenchmarkExperimentRecord(
                    experiment_id=f"EXP_{exp_counter:04d}",
                    dataset_id=ds_id,
                    scenario_id=sc_id,
                    drift_category=sc_cfg.category.value,
                    strategy=RepairStrategy.AUTHENTIC_WEIGHTED_ADAPTATION.value,
                    generator=None,
                    adaptation_budget=50,
                    synthetic_quantity=0,
                    sample_weight_policy={"source": 1.0, "target": w_tgt, "synth": 1.0},
                    baseline_target_f1=base_dev_f1,
                    candidate_target_f1=cand_dev_f1,
                    delta_f1_vs_baseline=delta_dev,
                    matched_control_f1=matched_ctrl,
                    delta_f1_vs_control=round(cand_dev_f1 - matched_ctrl, 4),
                    baseline_source_f1=base_source_f1,
                    candidate_source_f1=cand_src_f1,
                    source_retention_delta=delta_src,
                    source_retention_preserved=(delta_src >= -0.05),
                    bootstrap_ci_95=ci_95,
                    dcr_privacy_5th_pct=None,
                    training_duration_seconds=dur,
                    is_acceptable=is_acc,
                    package_status="ACCEPTED" if is_acc else "REJECTED",
                    rejection_reasons=reasons,
                )
                records.append(rec_wt)
                exp_counter += 1

                # -------------------------------------------------------------
                # ARM 4, 5, 6: SYNTHETIC AUGMENTATION (TVAE, CTGAN, TabDDPM)
                # -------------------------------------------------------------
                # Fit generators on adaptation data (B=50) with CPU-bounded epochs
                generators = [
                    ("tvae", TVAEGenerator(epochs=20)),
                    ("ctgan", CTGANGenerator(epochs=20)),
                    ("tabddpm", TabDDPMGenerator(epochs=5, num_timesteps=20)),
                ]

                for gen_name, gen_inst in generators:
                    logger.info("    Evaluating Generator: %s on %s", gen_name, sc_id)
                    t0_gen = time.perf_counter()
                    try:
                        # Clean adaptation data for generator
                        gen_train_data = df_adapt_b50.copy()
                        for c in gen_train_data.columns:
                            if gen_train_data[c].dtype == object:
                                gen_train_data[c] = gen_train_data[c].fillna("missing").astype(str)
                            else:
                                gen_train_data[c] = gen_train_data[c].fillna(0.0)

                        gen_inst.fit(gen_train_data)
                        gen_fit_time = round(time.perf_counter() - t0_gen, 3)

                        # Test quantity search K in [25, 50, 100, 200]
                        k_list = [25, 50, 100, 200] if gen_name != "tabddpm" else [50]

                        for k_qty in k_list:
                            t0_sample = time.perf_counter()
                            syn_df = gen_inst.sample(k_qty)
                            bundle_syn = preprocessor.transform(syn_df)

                            # Compute privacy metric (DCR)
                            dcr_val = compute_dcr_privacy_metric(bundle_syn.X, bundle_ad50.X)

                            # Retrain candidate: source + adapt_b50 + synthetic
                            X_tr = np.vstack([bundle_src.X, bundle_ad50.X, bundle_syn.X])
                            y_tr = np.concatenate([bundle_src.y, bundle_ad50.y, bundle_syn.y])

                            cand = clone(base_model)
                            cand.fit(X_tr, y_tr)
                            total_dur = round(gen_fit_time + (time.perf_counter() - t0_sample), 3)

                            cand_dev_f1 = float(f1_score(bundle_dev.y, cand.predict(bundle_dev.X), pos_label=1, zero_division=0))
                            cand_src_f1 = float(f1_score(bundle_seval.y, cand.predict(bundle_seval.X), pos_label=1, zero_division=0))
                            delta_dev = round(cand_dev_f1 - base_dev_f1, 4)
                            delta_src = round(cand_src_f1 - base_source_f1, 4)
                            ci_95 = compute_paired_bootstrap_ci(bundle_dev.y, cand.predict(bundle_dev.X), base_pred_dev)
                            delta_vs_ctrl = round(cand_dev_f1 - matched_ctrl, 4)

                            reasons = []
                            is_acc = True
                            if delta_src < -0.05:
                                is_acc = False
                                reasons.append("Source retention degraded.")
                            if delta_dev < 0.01:
                                is_acc = False
                                reasons.append("Utility gain below practical margin (+0.01).")
                            if delta_vs_ctrl < 0.005:
                                is_acc = False
                                reasons.append("Synthetic utility does not exceed matched authentic control (+0.005 margin).")
                            if ci_95[1] <= 0:
                                is_acc = False
                                reasons.append("Bootstrap 95% CI upper bound <= 0.")

                            rec_syn = BenchmarkExperimentRecord(
                                experiment_id=f"EXP_{exp_counter:04d}",
                                dataset_id=ds_id,
                                scenario_id=sc_id,
                                drift_category=sc_cfg.category.value,
                                strategy=RepairStrategy.SYNTHETIC_UNWEIGHTED.value,
                                generator=gen_name,
                                adaptation_budget=50,
                                synthetic_quantity=k_qty,
                                sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                                baseline_target_f1=base_dev_f1,
                                candidate_target_f1=cand_dev_f1,
                                delta_f1_vs_baseline=delta_dev,
                                matched_control_f1=matched_ctrl,
                                delta_f1_vs_control=delta_vs_ctrl,
                                baseline_source_f1=base_source_f1,
                                candidate_source_f1=cand_src_f1,
                                source_retention_delta=delta_src,
                                source_retention_preserved=(delta_src >= -0.05),
                                bootstrap_ci_95=ci_95,
                                dcr_privacy_5th_pct=dcr_val,
                                training_duration_seconds=total_dur,
                                is_acceptable=is_acc,
                                package_status="ACCEPTED" if is_acc else "REJECTED",
                                rejection_reasons=reasons,
                            )
                            records.append(rec_syn)
                            exp_counter += 1

                    except Exception as e:
                        logger.warning("Generator %s failed on %s: %s", gen_name, sc_id, str(e))
                        rec_err = BenchmarkExperimentRecord(
                            experiment_id=f"EXP_{exp_counter:04d}",
                            dataset_id=ds_id,
                            scenario_id=sc_id,
                            drift_category=sc_cfg.category.value,
                            strategy=RepairStrategy.SYNTHETIC_UNWEIGHTED.value,
                            generator=gen_name,
                            adaptation_budget=50,
                            synthetic_quantity=50,
                            sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                            baseline_target_f1=base_dev_f1,
                            candidate_target_f1=base_dev_f1,
                            delta_f1_vs_baseline=0.0,
                            matched_control_f1=matched_ctrl,
                            delta_f1_vs_control=0.0,
                            baseline_source_f1=base_source_f1,
                            candidate_source_f1=base_source_f1,
                            source_retention_delta=0.0,
                            source_retention_preserved=True,
                            bootstrap_ci_95=(0.0, 0.0),
                            dcr_privacy_5th_pct=None,
                            training_duration_seconds=0.0,
                            is_acceptable=False,
                            package_status="FAILED",
                            rejection_reasons=[f"Generator execution error: {str(e)[:100]}"],
                        )
                        records.append(rec_err)
                        exp_counter += 1

                # -------------------------------------------------------------
                # ARM 7: WEIGHTED_SYNTHETIC_HYBRID (TVAE K=50, w_synth=0.75, w_target=2.0)
                # -------------------------------------------------------------
                t0 = time.perf_counter()
                try:
                    tvae_gen = TVAEGenerator(epochs=20)
                    tvae_gen.fit(gen_train_data)
                    syn_df = tvae_gen.sample(50)
                    bundle_syn = preprocessor.transform(syn_df)
                    dcr_val = compute_dcr_privacy_metric(bundle_syn.X, bundle_ad50.X)

                    w_s, w_t, w_y = 1.0, 2.0, 0.75
                    X_tr = np.vstack([bundle_src.X, bundle_ad50.X, bundle_syn.X])
                    y_tr = np.concatenate([bundle_src.y, bundle_ad50.y, bundle_syn.y])
                    w_tr = np.concatenate([
                        np.full(len(bundle_src.y), w_s),
                        np.full(len(bundle_ad50.y), w_t),
                        np.full(len(bundle_syn.y), w_y),
                    ])

                    cand = clone(base_model)
                    cand.fit(X_tr, y_tr, sample_weight=w_tr)
                    dur = round(time.perf_counter() - t0, 3)

                    cand_dev_f1 = float(f1_score(bundle_dev.y, cand.predict(bundle_dev.X), pos_label=1, zero_division=0))
                    cand_src_f1 = float(f1_score(bundle_seval.y, cand.predict(bundle_seval.X), pos_label=1, zero_division=0))
                    delta_dev = round(cand_dev_f1 - base_dev_f1, 4)
                    delta_src = round(cand_src_f1 - base_source_f1, 4)
                    ci_95 = compute_paired_bootstrap_ci(bundle_dev.y, cand.predict(bundle_dev.X), base_pred_dev)
                    delta_vs_ctrl = round(cand_dev_f1 - matched_ctrl, 4)

                    reasons = []
                    is_acc = True
                    if delta_src < -0.05:
                        is_acc = False
                        reasons.append("Source retention degraded.")
                    if delta_dev < 0.01:
                        is_acc = False
                        reasons.append("Utility gain below practical margin (+0.01).")
                    if ci_95[1] <= 0:
                        is_acc = False
                        reasons.append("Bootstrap 95% CI upper bound <= 0.")

                    rec_hyb = BenchmarkExperimentRecord(
                        experiment_id=f"EXP_{exp_counter:04d}",
                        dataset_id=ds_id,
                        scenario_id=sc_id,
                        drift_category=sc_cfg.category.value,
                        strategy=RepairStrategy.SYNTHETIC_WEIGHTED_HYBRID.value,
                        generator="tvae",
                        adaptation_budget=50,
                        synthetic_quantity=50,
                        sample_weight_policy={"source": w_s, "target": w_t, "synth": w_y},
                        baseline_target_f1=base_dev_f1,
                        candidate_target_f1=cand_dev_f1,
                        delta_f1_vs_baseline=delta_dev,
                        matched_control_f1=matched_ctrl,
                        delta_f1_vs_control=delta_vs_ctrl,
                        baseline_source_f1=base_source_f1,
                        candidate_source_f1=cand_src_f1,
                        source_retention_delta=delta_src,
                        source_retention_preserved=(delta_src >= -0.05),
                        bootstrap_ci_95=ci_95,
                        dcr_privacy_5th_pct=dcr_val,
                        training_duration_seconds=dur,
                        is_acceptable=is_acc,
                        package_status="ACCEPTED" if is_acc else "REJECTED",
                        rejection_reasons=reasons,
                    )
                    records.append(rec_hyb)
                    exp_counter += 1

                except Exception as e:
                    logger.warning("Hybrid repair arm failed on %s: %s", sc_id, str(e))

                # -------------------------------------------------------------
                # ARM 8: TARGETED_SYNTHETIC_REPAIR (on diagnosed slice)
                # -------------------------------------------------------------
                if sc_cfg.slice_rule and not sc_cfg.is_concept_drift:
                    t0 = time.perf_counter()
                    try:
                        # Extract slice
                        slice_cond = df_adapt_b50.eval(sc_cfg.slice_rule) if hasattr(df_adapt_b50, "eval") else None
                        if slice_cond is not None and slice_cond.sum() >= 5:
                            slice_data = df_adapt_b50[slice_cond.values].copy()
                            tvae_tgt = TVAEGenerator(epochs=20)
                            tvae_tgt.fit(slice_data)
                            syn_tgt = tvae_tgt.sample(50)
                            bundle_tgt = preprocessor.transform(syn_tgt)

                            X_tr = np.vstack([bundle_src.X, bundle_ad50.X, bundle_tgt.X])
                            y_tr = np.concatenate([bundle_src.y, bundle_ad50.y, bundle_tgt.y])

                            cand = clone(base_model)
                            cand.fit(X_tr, y_tr)
                            dur = round(time.perf_counter() - t0, 3)

                            cand_dev_f1 = float(f1_score(bundle_dev.y, cand.predict(bundle_dev.X), pos_label=1, zero_division=0))
                            cand_src_f1 = float(f1_score(bundle_seval.y, cand.predict(bundle_seval.X), pos_label=1, zero_division=0))
                            delta_dev = round(cand_dev_f1 - base_dev_f1, 4)
                            delta_src = round(cand_src_f1 - base_source_f1, 4)
                            ci_95 = compute_paired_bootstrap_ci(bundle_dev.y, cand.predict(bundle_dev.X), base_pred_dev)
                            delta_vs_ctrl = round(cand_dev_f1 - matched_ctrl, 4)

                            reasons = []
                            is_acc = True
                            if delta_src < -0.05:
                                is_acc = False
                                reasons.append("Source retention degraded.")
                            if delta_dev < 0.01:
                                is_acc = False
                                reasons.append("Utility gain below practical margin (+0.01).")
                            if ci_95[1] <= 0:
                                is_acc = False
                                reasons.append("Bootstrap 95% CI upper bound <= 0.")

                            rec_tgt = BenchmarkExperimentRecord(
                                experiment_id=f"EXP_{exp_counter:04d}",
                                dataset_id=ds_id,
                                scenario_id=sc_id,
                                drift_category=sc_cfg.category.value,
                                strategy=RepairStrategy.TARGETED_SYNTHETIC_REPAIR.value,
                                generator="tvae",
                                adaptation_budget=50,
                                synthetic_quantity=50,
                                sample_weight_policy={"source": 1.0, "target": 1.0, "synth": 1.0},
                                baseline_target_f1=base_dev_f1,
                                candidate_target_f1=cand_dev_f1,
                                delta_f1_vs_baseline=delta_dev,
                                matched_control_f1=matched_ctrl,
                                delta_f1_vs_control=delta_vs_ctrl,
                                baseline_source_f1=base_source_f1,
                                candidate_source_f1=cand_src_f1,
                                source_retention_delta=delta_src,
                                source_retention_preserved=(delta_src >= -0.05),
                                bootstrap_ci_95=ci_95,
                                dcr_privacy_5th_pct=None,
                                training_duration_seconds=dur,
                                is_acceptable=is_acc,
                                package_status="ACCEPTED" if is_acc else "REJECTED",
                                rejection_reasons=reasons,
                            )
                            records.append(rec_tgt)
                            exp_counter += 1
                    except Exception as e:
                        logger.warning("Targeted repair arm failed on %s: %s", sc_id, str(e))

        logger.info("Multi-Dataset Benchmark Completed! Total experimental arms executed: %d", len(records))
        return records

    def save_records(self, records: list[BenchmarkExperimentRecord]) -> tuple[Path, Path]:
        """Serialize benchmark records to JSON and CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RESULTS_DIR / "multidataset_benchmark_results.json"
        csv_path = RESULTS_DIR / "multidataset_benchmark_results.csv"

        dict_list = [r.to_dict() for r in records]
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(dict_list, f, indent=2)

        df = pd.DataFrame(dict_list)
        df.to_csv(csv_path, index=False)

        logger.info("Saved %d benchmark records to:", len(records))
        logger.info("  JSON: %s", json_path)
        logger.info("  CSV:  %s", csv_path)
        return json_path, csv_path


def main() -> None:
    runner = MultiDatasetBenchmarkRunner(seed=42)
    records = runner.run_benchmark()
    json_path, csv_path = runner.save_records(records)
    print(f"Benchmark finished with {len(records)} records.")


if __name__ == "__main__":
    main()
