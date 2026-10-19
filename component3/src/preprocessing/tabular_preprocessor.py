"""Configurable Tabular Preprocessing Framework for Component 3.

Maintains 3 distinct data representations:
1. Original Raw Tabular Data (untransformed, native types)
2. Generator-Compatible Data (clean tabular frame, discrete features identified, no one-hot explosion)
3. Model-Compatible Transformed Data (imputed, scaled, encoded, ready for XGBoost/LightGBM/RF)

Strictly prevents target leakage, identifier leakage, and evaluation contamination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from component3.src.datasets.dataset_registry import DatasetMetadata


@dataclass
class PreprocessedDataBundle:
    """Encapsulates the 3 distinct representations produced by the preprocessor."""
    raw_df: pd.DataFrame
    generator_df: pd.DataFrame
    generator_discrete_columns: list[str]
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    target_name: str | None


class TabularDataPreprocessor:
    """Configurable preprocessor separating generator and model feature spaces."""

    def __init__(
        self,
        numerical_features: list[str],
        categorical_features: list[str],
        target_column: str | None = None,
        target_mapping: dict[Any, int] | None = None,
        drop_columns: list[str] | None = None,
        id_columns: list[str] | None = None,
        leakage_prone_columns: list[str] | None = None,
        missing_representations: list[Any] | None = None,
        scale_numerical: bool = True,
    ) -> None:
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)
        self.target_column = target_column
        self.target_mapping = target_mapping or {}
        self.drop_columns = list(drop_columns or [])
        self.id_columns = list(id_columns or [])
        self.leakage_prone_columns = list(leakage_prone_columns or [])
        self.missing_representations = list(missing_representations or ["?", "", "NA", "null", "NaN"])
        self.scale_numerical = scale_numerical

        # Exclusions: columns that must never become model/generator features
        self.excluded_columns = set(
            self.drop_columns + self.id_columns + self.leakage_prone_columns
        )
        if self.target_column:
            self.excluded_columns.add(self.target_column)

        # Active features in deterministic ordering
        self.active_numerical = [c for c in self.numerical_features if c not in self.excluded_columns]
        self.active_categorical = [c for c in self.categorical_features if c not in self.excluded_columns]
        self.ordered_features = self.active_numerical + self.active_categorical

        # Fitted estimators
        self._num_imputer: SimpleImputer | None = None
        self._cat_imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._encoder: OrdinalEncoder | None = None
        self._is_fitted: bool = False
        self._class_weights: dict[int, float] = {}

    @classmethod
    def from_dataset_metadata(
        cls,
        metadata: Any,
        scale_numerical: bool = True,
    ) -> TabularDataPreprocessor:
        """Instantiate preprocessor configured directly from dataset registry metadata."""
        return cls(
            numerical_features=metadata.numerical_features,
            categorical_features=metadata.categorical_features,
            target_column=metadata.target_column,
            target_mapping=metadata.target_mapping,
            drop_columns=metadata.drop_columns,
            id_columns=metadata.id_columns,
            leakage_prone_columns=metadata.leakage_prone_columns,
            missing_representations=metadata.missing_representations,
            scale_numerical=scale_numerical,
        )

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted on training data."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """Names of transformed model features in strict column order."""
        return list(self.ordered_features)

    def _clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize missing representations to np.nan without altering native dtypes."""
        cleaned = df.copy()
        # Clean string whitespace if column is object or string
        for col in cleaned.columns:
            if cleaned[col].dtype == object or str(cleaned[col].dtype) == "string":
                cleaned[col] = cleaned[col].astype(str).str.strip()
                # Replace specified missing tokens
                for token in self.missing_representations:
                    cleaned.loc[cleaned[col] == str(token).strip(), col] = np.nan

        # Replace non-string missing tokens across all columns
        for token in self.missing_representations:
            cleaned = cleaned.replace(token, np.nan)

        return cleaned

    def fit(self, df: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> TabularDataPreprocessor:
        """Fit imputers, scalers, and encoders exclusively on training data."""
        df_clean = self._clean_missing_values(df)

        # 1. Fit Numerical Pipeline
        if self.active_numerical:
            # Coerce numerical columns to float
            num_data = df_clean[self.active_numerical].apply(pd.to_numeric, errors="coerce")
            self._num_imputer = SimpleImputer(strategy="median")
            num_imputed = self._num_imputer.fit_transform(num_data)

            if self.scale_numerical:
                self._scaler = StandardScaler()
                self._scaler.fit(num_imputed)

        # 2. Fit Categorical Pipeline
        if self.active_categorical:
            cat_data = df_clean[self.active_categorical].astype(str)
            self._cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
            cat_imputed = self._cat_imputer.fit_transform(cat_data)

            # OrdinalEncoder with safe unknown category handling
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            self._encoder.fit(cat_imputed)

        # 3. Compute Class Weights on training target (if provided)
        target_series = None
        if y is not None:
            target_series = pd.Series(y)
        elif self.target_column and self.target_column in df_clean.columns:
            raw_target = df_clean[self.target_column]
            if self.target_mapping:
                target_series = raw_target.map(self.target_mapping)
            else:
                target_series = raw_target

        if target_series is not None:
            clean_targets = target_series.dropna()
            counts = clean_targets.value_counts().to_dict()
            total = len(clean_targets)
            n_classes = len(counts)
            if n_classes > 0 and total > 0:
                self._class_weights = {
                    cls: float(total / (n_classes * count)) for cls, count in counts.items()
                }

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> PreprocessedDataBundle:
        """Transform input data into the 3 standardized representations."""
        if not self._is_fitted:
            raise RuntimeError("TabularDataPreprocessor must be fitted on training data before calling transform.")

        df_clean = self._clean_missing_values(df)

        # -------------------------------------------------------------
        # Representation 1: Original Tabular Data (raw copy)
        # -------------------------------------------------------------
        raw_df = df.copy()

        # -------------------------------------------------------------
        # Representation 2: Generator-Compatible Tabular Data
        # - Target and identifiers removed
        # - Discrete features preserved as clean string / integer types
        # - Numerical features preserved in continuous domain
        # - NO one-hot encoding (prevents dimensionality explosion)
        # -------------------------------------------------------------
        generator_cols = [c for c in self.ordered_features if c in df_clean.columns]
        gen_df = df_clean[generator_cols].copy()

        # Fill missing values for generator training using fitted training statistics
        if self.active_numerical and self._num_imputer:
            num_clean = gen_df[self.active_numerical].apply(pd.to_numeric, errors="coerce")
            num_filled = self._num_imputer.transform(num_clean)
            gen_df[self.active_numerical] = num_filled

        if self.active_categorical and self._cat_imputer:
            cat_clean = gen_df[self.active_categorical].astype(str)
            cat_filled = self._cat_imputer.transform(cat_clean)
            gen_df[self.active_categorical] = cat_filled

        # -------------------------------------------------------------
        # Representation 3: Model-Compatible Transformed Data (X, y)
        # - Numerical scaled
        # - Categorical ordinally encoded with unknown token handling
        # - Strict column order matching self.ordered_features
        # -------------------------------------------------------------
        transformed_parts: list[np.ndarray] = []

        if self.active_numerical:
            num_sub = df_clean[self.active_numerical].apply(pd.to_numeric, errors="coerce")
            num_imp = self._num_imputer.transform(num_sub)
            if self.scale_numerical and self._scaler:
                num_transformed = self._scaler.transform(num_imp)
            else:
                num_transformed = num_imp
            transformed_parts.append(num_transformed.astype(np.float32))

        if self.active_categorical:
            cat_sub = df_clean[self.active_categorical].astype(str)
            cat_imp = self._cat_imputer.transform(cat_sub)
            cat_encoded = self._encoder.transform(cat_imp)
            transformed_parts.append(cat_encoded.astype(np.float32))

        if transformed_parts:
            X = np.hstack(transformed_parts)
        else:
            X = np.empty((len(df), 0), dtype=np.float32)

        # Extract target vector if available
        y: np.ndarray | None = None
        if self.target_column and self.target_column in df.columns:
            target_raw = df[self.target_column]
            if self.target_mapping:
                y = target_raw.map(self.target_mapping).to_numpy()
            else:
                try:
                    num_target = pd.to_numeric(target_raw, errors="raise")
                    y = num_target.to_numpy()
                except Exception:
                    y = target_raw.to_numpy()

        return PreprocessedDataBundle(
            raw_df=raw_df,
            generator_df=gen_df,
            generator_discrete_columns=list(self.active_categorical),
            X=X,
            y=y,
            feature_names=list(self.ordered_features),
            target_name=self.target_column,
        )

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> PreprocessedDataBundle:
        """Fit on training data and return the 3 representations."""
        return self.fit(df, y=y).transform(df)

    def transform_synthetic(self, synthetic_df: pd.DataFrame) -> np.ndarray:
        """Transform synthetic records directly into the model-compatible feature matrix X."""
        if not self._is_fitted:
            raise RuntimeError("TabularDataPreprocessor must be fitted on training data before transforming synthetic data.")

        # Ensure all required features are present
        missing = [c for c in self.ordered_features if c not in synthetic_df.columns]
        if missing:
            raise ValueError(f"Synthetic dataframe missing required feature columns: {missing}")

        bundle = self.transform(synthetic_df)
        return bundle.X

    def get_class_weights(self) -> dict[int, float]:
        """Retrieve balanced class weights computed from the training split."""
        return dict(self._class_weights)

    def save(self, filepath: Path | str) -> None:
        """Serialize fitted preprocessor to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Path | str) -> TabularDataPreprocessor:
        """Load serialized preprocessor from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
        preprocessor = joblib.load(path)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"Loaded object is of type {type(preprocessor)}, expected {cls.__name__}")
        return preprocessor


def create_preprocessor_from_metadata(metadata: DatasetMetadata, scale_numerical: bool = True) -> TabularDataPreprocessor:
    """Convenience factory creating a preprocessor initialized directly from DatasetMetadata."""
    return TabularDataPreprocessor.from_dataset_metadata(metadata, scale_numerical=scale_numerical)


def get_preprocessor_for_dataset(dataset_id: str, scale_numerical: bool = True) -> TabularDataPreprocessor:
    """Retrieve and configure a preprocessor for any dataset registered in DatasetRegistry."""
    from component3.src.datasets.dataset_registry import get_dataset_registry
    meta = get_dataset_registry().get(dataset_id)
    return create_preprocessor_from_metadata(meta, scale_numerical=scale_numerical)
