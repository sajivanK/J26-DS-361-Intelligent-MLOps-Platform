"""Baseline Model Adapter for Component 3.

Enables seamless loading and evaluation of pre-existing or freshly trained baseline models
without requiring redundant retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from component3.src.baselines.baseline_trainer import BaselineModelTrainer, ModelMetrics
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor


class BaselineModelAdapter:
    """Adapter wrapping scikit-learn, XGBoost, or LightGBM models with standardized evaluation."""

    def __init__(
        self,
        model: Any,
        model_family: str = "xgboost",
        preprocessor: Any | None = None,
    ) -> None:
        self.model = model
        self.model_family = model_family
        self.preprocessor = preprocessor

    @classmethod
    def load(
        cls,
        model_path: Path | str,
        model_family: str = "xgboost",
        preprocessor: Any | None = None,
        preprocessor_path: Path | str | None = None,
    ) -> BaselineModelAdapter:
        """Load an existing serialized model artifact from disk."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        try:
            from component3.src.core.model_loader import load_trusted_pickle
            model = load_trusted_pickle(path, trusted=True)
        except Exception:
            model = joblib.load(path)

        prep = preprocessor
        if prep is None and preprocessor_path:
            prep_p = Path(preprocessor_path)
            if prep_p.is_file():
                try:
                    from component3.src.core.model_loader import load_trusted_pickle
                    prep = load_trusted_pickle(prep_p, trusted=True)
                except Exception:
                    prep = joblib.load(prep_p)

        return cls(model=model, model_family=model_family, preprocessor=prep)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict binary class labels."""
        features = self._prepare_features(X)
        return self.model.predict(features)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        features = self._prepare_features(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)
        # Fallback if model only supports predict
        preds = self.model.predict(features)
        return np.column_stack([1.0 - preds, preds])

    def evaluate(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ModelMetrics:
        """Calculate standardized classification metrics."""
        features = self._prepare_features(X)
        targets = np.asarray(y).astype(int)
        return BaselineModelTrainer.evaluate_model(self.model, features, targets)

    def evaluate_slice(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        slice_mask: np.ndarray | pd.Series,
    ) -> tuple[ModelMetrics, ModelMetrics]:
        """Evaluate performance partitioned into affected slice vs non-affected population."""
        features = self._prepare_features(X)
        targets = np.asarray(y).astype(int)
        mask = np.asarray(slice_mask).astype(bool)

        if not np.any(mask):
            raise ValueError("Slice mask matched 0 records; cannot evaluate slice metrics.")
        if np.all(mask):
            raise ValueError("Slice mask matched 100% of records; non-affected population is empty.")

        slice_metrics = BaselineModelTrainer.evaluate_model(
            self.model, features[mask], targets[mask]
        )
        non_affected_metrics = BaselineModelTrainer.evaluate_model(
            self.model, features[~mask], targets[~mask]
        )
        return slice_metrics, non_affected_metrics

    def _prepare_features(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Ensure input is converted into model-compatible 2D numpy array."""
        if isinstance(X, pd.DataFrame):
            df_in = X.copy()
            if self.preprocessor is not None:
                is_fitted = getattr(self.preprocessor, "is_fitted", True)
                if is_fitted:
                    res = self.preprocessor.transform(df_in)
                    if hasattr(res, "X"):
                        return res.X
                    elif isinstance(res, tuple):
                        # e.g. UCIAdultPreprocessor returns (df_transformed, y)
                        feat_df = res[0]
                        return feat_df.to_numpy() if isinstance(feat_df, pd.DataFrame) else np.asarray(feat_df)
                    elif isinstance(res, pd.DataFrame):
                        return res.to_numpy()
                    else:
                        return np.asarray(res)

            # If no preprocessor is attached, drop known target columns if present
            for target_col in ("target", "income", "TARGET", "Churn", "isFraud", "y"):
                if target_col in df_in.columns:
                    df_in = df_in.drop(columns=[target_col])
            return df_in.to_numpy()
        return np.asarray(X)
