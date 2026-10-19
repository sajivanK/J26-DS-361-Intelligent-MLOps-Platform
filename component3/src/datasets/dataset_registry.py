"""Multi-Dataset Registry for Component 3 Experimental Platform.

Provides a standardized catalog of candidate binary-classification tabular datasets
with explicit column types, missingness semantics, leakage rules, and splitting strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


class DatasetStatus(str, Enum):
    """Availability status of dataset in local repository."""
    VERIFIED = "VERIFIED"
    PENDING_DOWNLOAD = "PENDING_DOWNLOAD"


class SplitStrategy(str, Enum):
    """Reproducible partition strategy."""
    STRATIFIED = "STRATIFIED"
    TIME_BASED = "TIME_BASED"
    GROUP_AWARE = "GROUP_AWARE"


@dataclass
class DatasetMetadata:
    """Configurable metadata schema for a tabular research dataset."""
    dataset_id: str
    display_name: str
    version: str
    source: str
    license_info: str
    status: DatasetStatus
    target_column: str
    target_mapping: dict[Any, int]
    numerical_features: list[str]
    categorical_features: list[str]
    missing_representations: list[Any] = field(default_factory=lambda: ["?", "", "NA", "null"])
    id_columns: list[str] = field(default_factory=list)
    leakage_prone_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    class_distribution: dict[int, float] = field(default_factory=dict)
    split_strategy: SplitStrategy = SplitStrategy.STRATIFIED
    split_column: str | None = None
    label_availability: str = "fully_labeled"
    label_provenance: str = "empirical_ground_truth"
    preprocessing_notes: str = ""
    local_path: str | None = None

    @property
    def feature_columns(self) -> list[str]:
        """All usable feature columns (excluding target, IDs, and dropped columns)."""
        excluded = set(self.id_columns + self.drop_columns + self.leakage_prone_columns + [self.target_column])
        return [c for c in (self.numerical_features + self.categorical_features) if c not in excluded]

    @property
    def is_local_available(self) -> bool:
        """Check if dataset file currently exists locally."""
        if not self.local_path:
            return False
        p = ROOT / self.local_path if not Path(self.local_path).is_absolute() else Path(self.local_path)
        return p.is_file() or p.is_dir()

    def get_absolute_path(self) -> Path | None:
        """Resolve absolute filesystem path if configured."""
        if not self.local_path:
            return None
        p = Path(self.local_path)
        return p if p.is_absolute() else (ROOT / p)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "version": self.version,
            "source": self.source,
            "license_info": self.license_info,
            "status": self.status.value,
            "target_column": self.target_column,
            "target_mapping": self.target_mapping,
            "numerical_features_count": len(self.numerical_features),
            "categorical_features_count": len(self.categorical_features),
            "total_features_count": len(self.feature_columns),
            "missing_representations": self.missing_representations,
            "id_columns": self.id_columns,
            "leakage_prone_columns": self.leakage_prone_columns,
            "drop_columns": self.drop_columns,
            "class_distribution": self.class_distribution,
            "split_strategy": self.split_strategy.value,
            "split_column": self.split_column,
            "label_availability": self.label_availability,
            "label_provenance": self.label_provenance,
            "is_local_available": self.is_local_available,
        }


class DatasetRegistry:
    """Central catalog and lifecycle manager for multi-dataset experiments."""

    def __init__(self) -> None:
        self._registry: dict[str, DatasetMetadata] = {}
        self._register_default_datasets()

    def register(self, metadata: DatasetMetadata) -> None:
        """Register or update a dataset specification."""
        self._registry[metadata.dataset_id] = metadata

    def get(self, dataset_id: str) -> DatasetMetadata:
        """Retrieve dataset metadata by identifier."""
        if dataset_id not in self._registry:
            supported = sorted(list(self._registry.keys()))
            raise KeyError(f"Dataset '{dataset_id}' not found in registry. Supported datasets: {supported}")
        return self._registry[dataset_id]

    def list_datasets(self, status: DatasetStatus | None = None) -> list[str]:
        """List dataset identifiers, optionally filtered by status."""
        if status is None:
            return sorted(list(self._registry.keys()))
        return sorted([k for k, v in self._registry.items() if v.status == status])

    def is_available(self, dataset_id: str) -> bool:
        """Check if dataset exists and is verified locally."""
        meta = self.get(dataset_id)
        return meta.status == DatasetStatus.VERIFIED and meta.is_local_available

    def load_dataset(self, dataset_id: str, nrows: int | None = None) -> pd.DataFrame:
        """Load dataset DataFrame if locally available, otherwise raise descriptive error."""
        meta = self.get(dataset_id)
        path = meta.get_absolute_path()

        if not path or not path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_id}' ({meta.display_name}) is marked as {meta.status.value}. "
                f"Expected file at: '{meta.local_path}'. "
                f"To use this dataset in Sprint 06 Batch 2, please download from {meta.source} "
                f"or utilize the Colab execution runner."
            )

        # Dataset specific loaders
        if meta.dataset_id == "uci_adult":
            # Handles header vs non-header format
            df = pd.read_csv(path, nrows=nrows)
            return df

        elif meta.dataset_id == "ieee_cis":
            df = pd.read_csv(path, nrows=nrows)
            return df

        else:
            df = pd.read_csv(path, nrows=nrows)
            return df

    def _register_default_datasets(self) -> None:
        """Initialize the 5 standard candidate tabular datasets."""

        # 1. UCI Adult Census Income (Verified reference in repo)
        self.register(
            DatasetMetadata(
                dataset_id="uci_adult",
                display_name="UCI Adult Census Income",
                version="1.0.0",
                source="https://archive.ics.uci.edu/ml/datasets/adult",
                license_info="CC BY 4.0 / Open Access",
                status=DatasetStatus.VERIFIED,
                target_column="target",
                target_mapping={"<=50K": 0, ">50K": 1, 0: 0, 1: 1},
                numerical_features=[
                    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
                ],
                categorical_features=[
                    "workclass", "marital_status", "occupation",
                    "relationship", "race", "sex", "native_country"
                ],
                missing_representations=["?", " ?", "", "NA", "null"],
                id_columns=[],
                leakage_prone_columns=[],
                drop_columns=["fnlwgt", "education"],
                class_distribution={0: 0.759, 1: 0.241},
                split_strategy=SplitStrategy.STRATIFIED,
                split_column=None,
                label_availability="fully_labeled",
                label_provenance="census_survey_ground_truth",
                preprocessing_notes="Standardized reference benchmark in shared/uci_adult/data/reference.csv",
                local_path="shared/uci_adult/data/reference.csv",
            )
        )

        # 2. Telco Customer Churn (IBM / Kaggle)
        self.register(
            DatasetMetadata(
                dataset_id="telco_churn",
                display_name="Telco Customer Churn",
                version="1.0.0",
                source="https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
                license_info="IBM Community Data License Agreement / Public Domain",
                status=DatasetStatus.PENDING_DOWNLOAD,
                target_column="Churn",
                target_mapping={"No": 0, "Yes": 1, 0: 0, 1: 1},
                numerical_features=["tenure", "MonthlyCharges", "TotalCharges"],
                categorical_features=[
                    "gender", "SeniorCitizen", "Partner", "Dependents",
                    "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod"
                ],
                missing_representations=[" ", "", "NA", "null", "NaN"],
                id_columns=["customerID"],
                leakage_prone_columns=[],
                drop_columns=["customerID"],
                class_distribution={0: 0.735, 1: 0.265},
                split_strategy=SplitStrategy.STRATIFIED,
                split_column=None,
                label_availability="fully_labeled",
                label_provenance="telecom_operational_logs",
                preprocessing_notes="TotalCharges contains whitespace for new customers with tenure=0 (impute as 0.0).",
                local_path="component3/data/datasets/telco_churn.csv",
            )
        )

        # 3. Home Credit Default Risk (Kaggle)
        self.register(
            DatasetMetadata(
                dataset_id="home_credit",
                display_name="Home Credit Default Risk",
                version="1.0.0",
                source="https://www.kaggle.com/c/home-credit-default-risk",
                license_info="Kaggle Competition License / Research Only",
                status=DatasetStatus.PENDING_DOWNLOAD,
                target_column="TARGET",
                target_mapping={0: 0, 1: 1},
                numerical_features=[
                    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
                    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "EXT_SOURCE_1",
                    "EXT_SOURCE_2", "EXT_SOURCE_3"
                ],
                categorical_features=[
                    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
                    "FLAG_OWN_REALTY", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE",
                    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                    "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
                ],
                missing_representations=["", "XNA", "NA", "null", "NaN"],
                id_columns=["SK_ID_CURR"],
                leakage_prone_columns=[],
                drop_columns=["SK_ID_CURR"],
                class_distribution={0: 0.919, 1: 0.081},
                split_strategy=SplitStrategy.STRATIFIED,
                split_column=None,
                label_availability="fully_labeled",
                label_provenance="credit_bureau_default_outcomes",
                preprocessing_notes="Severe class imbalance (~8% positive default). DAYS_EMPLOYED contains 365243 anomaly.",
                local_path="component3/data/datasets/home_credit.csv",
            )
        )

        # 4. IEEE-CIS Fraud Detection (Vesta / Kaggle)
        self.register(
            DatasetMetadata(
                dataset_id="ieee_cis",
                display_name="IEEE-CIS Fraud Detection",
                version="1.0.0",
                source="https://www.kaggle.com/c/ieee-fraud-detection",
                license_info="IEEE / Kaggle Competition Research License",
                status=DatasetStatus.PENDING_DOWNLOAD,
                target_column="isFraud",
                target_mapping={0: 0, 1: 1},
                numerical_features=[
                    "TransactionAmt", "dist1", "dist2",
                    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                    "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"
                ],
                categorical_features=[
                    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
                    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
                    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
                ],
                missing_representations=["", "NA", "null", "NaN"],
                id_columns=["TransactionID"],
                leakage_prone_columns=[],
                drop_columns=["TransactionID"],
                class_distribution={0: 0.965, 1: 0.035},
                split_strategy=SplitStrategy.TIME_BASED,
                split_column="TransactionDT",
                label_availability="fully_labeled",
                label_provenance="transaction_chargeback_verification",
                preprocessing_notes="High missingness (>50% on V and D columns). Split must be temporal on TransactionDT.",
                local_path=None,
            )
        )

        # 5. Bank Marketing (UCI)
        self.register(
            DatasetMetadata(
                dataset_id="bank_marketing",
                display_name="Bank Marketing (UCI)",
                version="1.0.0",
                source="https://archive.ics.uci.edu/ml/datasets/bank+marketing",
                license_info="CC BY 4.0 / Open Access",
                status=DatasetStatus.PENDING_DOWNLOAD,
                target_column="y",
                target_mapping={"no": 0, "yes": 1, 0: 0, 1: 1},
                numerical_features=[
                    "age", "campaign", "pdays", "previous", "emp_var_rate",
                    "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"
                ],
                categorical_features=[
                    "job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "day_of_week", "poutcome"
                ],
                missing_representations=["unknown", "", "NA", "null"],
                id_columns=[],
                leakage_prone_columns=["duration"],
                drop_columns=["duration"],
                class_distribution={0: 0.887, 1: 0.113},
                split_strategy=SplitStrategy.STRATIFIED,
                split_column=None,
                label_availability="fully_labeled",
                label_provenance="telemarketing_campaign_outcomes",
                preprocessing_notes="duration column must be dropped to prevent target leakage in realistic prospective deployment.",
                local_path="component3/data/raw/bank_marketing.csv",
            )
        )


_GLOBAL_REGISTRY: DatasetRegistry | None = None


def get_dataset_registry() -> DatasetRegistry:
    """Singleton provider for global dataset registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = DatasetRegistry()
    return _GLOBAL_REGISTRY
