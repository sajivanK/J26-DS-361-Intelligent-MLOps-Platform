"""Component 3 Inter-Component Integration Adapters.

Provides typed adapters connecting Component 3 to:
- Component 1: Drift Detection
- Component 2: Drift Diagnosis & Root Cause Analysis
- Component 4: Agentic Decision, Retraining, and Deployment Gatekeeper
"""

from component3.src.adapters.c1_drift_adapter import (
    C1DriftAdapter,
    C1DriftPayload,
    normalize_c1_drift_type,
    normalize_c1_severity,
)
from component3.src.adapters.c2_diagnosis_adapter import (
    C2DiagnosisAdapter,
    C2DiagnosisPayload,
    normalize_c2_provenance,
)
from component3.src.adapters.c4_retraining_adapter import (
    C4RetrainingAdapter,
    C4ReviewPayload,
)

__all__ = [
    "C1DriftAdapter",
    "C1DriftPayload",
    "C2DiagnosisAdapter",
    "C2DiagnosisPayload",
    "C4RetrainingAdapter",
    "C4ReviewPayload",
    "normalize_c1_drift_type",
    "normalize_c1_severity",
    "normalize_c2_provenance",
]
