"""Unified Baseline Model Training Framework for Component 3.

Supports XGBoost, Random Forest, and LightGBM with explicit configuration tracking,
class imbalance handling, and standardized metric evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class ModelFamily(str, Enum):
    """Supported baseline model families."""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"


@dataclass
class ModelMetrics:
    """Standardized performance metrics for binary classification models."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    support: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class BaselineExperimentRecord:
    """Provenance and audit record for a baseline model training experiment."""
    experiment_id: str
    dataset_id: str
    model_family: str
    model_config: dict[str, Any]
    training_seed: int
    train_metrics: ModelMetrics
    val_metrics: ModelMetrics
    test_metrics: ModelMetrics
    dependency_versions: dict[str, str]
    model_path: str | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dictionary."""
        d = asdict(self)
        d["train_metrics"] = self.train_metrics.to_dict()
        d["val_metrics"] = self.val_metrics.to_dict()
        d["test_metrics"] = self.test_metrics.to_dict()
        return d

    def save(self, filepath: Path | str) -> None:
        """Save experiment record to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaselineModelTrainer:
    """Orchestrates model creation, training, evaluation, and artifact logging."""

    @staticmethod
    def get_dependencies() -> dict[str, str]:
        """Collect current versions of installed modeling dependencies."""
        versions = {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lgb.__version__ if HAS_LIGHTGBM else "NOT_INSTALLED",
        }
        return versions

    @classmethod
    def create_model(
        cls,
        family: ModelFamily | str,
        hyperparams: dict[str, Any] | None = None,
        seed: int = 42,
        class_weights: dict[int, float] | None = None,
    ) -> Any:
        """Instantiate a baseline classifier with specified parameters."""
        fam = ModelFamily(family) if isinstance(family, str) else family
        params = dict(hyperparams or {})

        if fam == ModelFamily.XGBOOST:
            # Default balanced hyperparameters
            cfg: dict[str, Any] = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": seed,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            }
            if class_weights and 1 in class_weights and 0 in class_weights:
                # scale_pos_weight = count(negative) / count(positive)
                cfg["scale_pos_weight"] = class_weights[1] / max(class_weights[0], 1e-6)
            cfg.update(params)
            # Remove use_label_encoder if xgboost version deprecated it
            cfg.pop("use_label_encoder", None)
            return xgboost.XGBClassifier(**cfg)

        elif fam == ModelFamily.RANDOM_FOREST:
            cfg = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": seed,
                "class_weight": "balanced" if class_weights else None,
            }
            cfg.update(params)
            return RandomForestClassifier(**cfg)

        elif fam == ModelFamily.LIGHTGBM:
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM is not installed in the current environment.")
            cfg = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": seed,
                "verbose": -1,
            }
            if class_weights and 1 in class_weights and 0 in class_weights:
                cfg["scale_pos_weight"] = class_weights[1] / max(class_weights[0], 1e-6)
            cfg.update(params)
            return lgb.LGBMClassifier(**cfg)

        else:
            raise ValueError(f"Unsupported model family '{fam}'")

    @classmethod
    def evaluate_model(cls, model: Any, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Calculate standardized classification metrics."""
        y_pred = model.predict(X)

        acc = float(accuracy_score(y, y_pred))
        prec = float(precision_score(y, y_pred, zero_division=0))
        rec = float(recall_score(y, y_pred, zero_division=0))
        f1 = float(f1_score(y, y_pred, zero_division=0))

        # Check probability prediction for ROC-AUC
        roc_auc = 0.5
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)
                if y_prob.shape[1] >= 2:
                    roc_auc = float(roc_auc_score(y, y_prob[:, 1]))
            except Exception:
                pass

        cm = confusion_matrix(y, y_pred).tolist()

        return ModelMetrics(
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            roc_auc=round(roc_auc, 4),
            confusion_matrix=cm,
            support=int(len(y)),
        )

    @classmethod
    def train_and_evaluate(
        cls,
        dataset_id: str,
        family: ModelFamily | str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hyperparams: dict[str, Any] | None = None,
        seed: int = 42,
        class_weights: dict[int, float] | None = None,
        model_save_path: Path | str | None = None,
    ) -> tuple[Any, BaselineExperimentRecord]:
        """Fit model on train set, evaluate on train/val/test, and log experiment record."""
        model = cls.create_model(family=family, hyperparams=hyperparams, seed=seed, class_weights=class_weights)
        model.fit(X_train, y_train)

        train_m = cls.evaluate_model(model, X_train, y_train)
        val_m = cls.evaluate_model(model, X_val, y_val)
        test_m = cls.evaluate_model(model, X_test, y_test)

        save_str: str | None = None
        if model_save_path:
            p = Path(model_save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, p)
            save_str = str(p)

        fam = ModelFamily(family)
        record = BaselineExperimentRecord(
            experiment_id=f"{dataset_id}_{fam.value}_seed{seed}",
            dataset_id=dataset_id,
            model_family=fam.value,
            model_config=dict(hyperparams or {}),
            training_seed=seed,
            train_metrics=train_m,
            val_metrics=val_m,
            test_metrics=test_m,
            dependency_versions=cls.get_dependencies(),
            model_path=save_str,
        )

        return model, record
