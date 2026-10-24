"""Explicit model identities. Missing artifacts are never silently substituted."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ModelEntry:
    model_id: str
    dataset_id: str
    model_path: str
    preprocessor_path: str
    target_column: str = "Churn"
    algorithm: str = "RandomForestClassifier"
    model_sha256: str | None = None
    preprocessor_sha256: str | None = None


class ModelRegistry:
    def __init__(self):
        self.entries = {}
        self._init_default_entries()

    def _init_default_entries(self):
        m_path = "component3/models/experimental/telco_rf_experimental_source.joblib"
        p_path = "component3/models/experimental/telco_preprocessor_experimental_source.joblib"
        if (ROOT / m_path).is_file() and (ROOT / p_path).is_file():
            self.register(
                ModelEntry(
                    model_id="telco_rf_experimental_source",
                    dataset_id="telco_churn",
                    model_path=m_path,
                    preprocessor_path=p_path,
                    target_column="Churn",
                    algorithm="RandomForestClassifier",
                    model_sha256="256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0",
                    preprocessor_sha256="a188f0285e6cf07741bbb69be8b3df44eece492e69166c78d92353f1f4d41224",
                )
            )

        # Experimental baseline has a separate identity from the team model.
        m_adult = "component3/models/baselines/uci_adult_full_xgboost.joblib"
        p_adult = "component3/models/baselines/uci_adult_full_preprocessor.joblib"
        if (ROOT / m_adult).is_file() and (ROOT / p_adult).is_file():
            self.register(
                ModelEntry(
                    model_id="uci_adult_full_experimental",
                    dataset_id="uci_adult_full",
                    model_path=m_adult,
                    preprocessor_path=p_adult,
                    target_column="target",
                    algorithm="XGBClassifier",
                    model_sha256="4e02e7a6362f5b4a64b3cc5f994326d554e870eec406487944d56b20af3e22e1",
                    preprocessor_sha256="01c5935e9c98ff148a0086f1c5a251472396af23d1759ef62fc28ec2c6bbcb4c",
                )
            )


    def register(self, entry):
        if entry.model_id in self.entries and self.entries[entry.model_id] != entry:
            raise ValueError(f"Model {entry.model_id!r} is already registered differently.")
        self.entries[entry.model_id] = entry

    def has(self, model_id):
        return model_id in self.entries

    def get(self, model_id):
        if not self.has(model_id):
            raise KeyError(f"Model {model_id!r} is not registered.")
        return self.entries[model_id]

    def list_models(self):
        return list(self.entries.values())

    def verify_integrity(self, model_id):
        entry = self.get(model_id)
        for name in ("model", "preprocessor"):
            path = (ROOT / getattr(entry, name + "_path")).resolve()
            expected = getattr(entry, name + "_sha256")
            if not path.is_file() or not expected:
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        return True

    def load_model(self, model_id):
        return self._load(model_id, "model")

    def load_preprocessor(self, model_id):
        return self._load(model_id, "preprocessor")

    def _load(self, model_id, name):
        if not self.verify_integrity(model_id):
            raise FileNotFoundError(f"Model {model_id}: artifacts or their verified hashes are missing.")
        return joblib.load(ROOT / getattr(self.get(model_id), name + "_path"))


_registry = ModelRegistry()


def get_model_registry():
    return _registry
