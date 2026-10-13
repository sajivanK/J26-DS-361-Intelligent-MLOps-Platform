"""Validate C1/C2 evidence and keep source declarations outside contracts."""
import math

import numpy as np
import pandas as pd

from c4.contracts.models import C1Evidence, C2Diagnosis
from c4.data import FEATURE_COLUMNS
from c4.scenarios.diagnosis import diagnosed_slice_mask


def _finite_payload(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_payload(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Upstream evidence must contain finite numeric values.')


def require_no_mock_sources(envelope):
    """A necessary research safeguard, not proof of real producer authenticity."""
    source = envelope['source_declarations']
    if source.get('c1') != 'real' or source.get('c2') != 'real':
        raise ValueError('Mock upstream evidence is restricted to engineering tests.')
    if envelope['c1_evidence']['engine_version'].lower().startswith('fake'):
        raise ValueError('Fake C1 evidence cannot enter a real-evidence research path.')


def adapt_evidence_pair(c1_payload, c2_payload, clean_current, *,
                        c1_origin, c2_origin, role='release_validation'):
    """Return a provenance envelope; this does not authorize model release.

    These adapters validate declared origins; they do not authenticate APIs.
    Diagnosed membership is computed on clean processed features only.
    """
    if role not in {'repair', 'release_validation'}:
        raise ValueError('Sealed-future evidence is forbidden here.')
    if c1_origin not in {'real', 'mock'} or c2_origin not in {'real', 'mock'}:
        raise ValueError('Origin must explicitly be real or mock.')
    if (clean_current.empty or not clean_current.index.is_unique
            or tuple(clean_current.columns) != (*FEATURE_COLUMNS, 'target')):
        raise ValueError('Clean current data has invalid schema or indices.')
    if (not all(pd.api.types.is_numeric_dtype(clean_current[c]) for c in clean_current)
            or not np.isfinite(clean_current.to_numpy(dtype=float)).all()
            or not clean_current['target'].isin([0, 1]).all()):
        raise ValueError('Clean current data must be finite numeric data with binary labels.')
    _finite_payload(c1_payload)
    _finite_payload(c2_payload)
    c1 = C1Evidence.model_validate(c1_payload)
    c2 = C2Diagnosis.model_validate(c2_payload)
    # Check parsed numbers too: strings may have been coerced by Pydantic.
    _finite_payload(c1.model_dump())
    _finite_payload(c2.model_dump())
    if c1_origin == 'real' and c1.engine_version.lower().startswith('fake'):
        raise ValueError('Fake C1 builder output cannot be labelled real.')
    if c1.batch_size != len(clean_current):
        raise ValueError('C1 batch_size differs from current-role row count.')
    membership = diagnosed_slice_mask(clean_current, c2, rule_feature_space='processed')
    if not membership.any() or not (~membership).any():
        raise ValueError('Both diagnosed and non-affected populations are required.')
    measured_support = float(membership.mean())
    return {
        'envelope_schema_version': '1.0',
        'c1_evidence': c1.model_dump(mode='json'),
        'c2_diagnosis': c2.model_dump(mode='json'),
        'source_declarations': {'c1': c1_origin, 'c2': c2_origin},
        'contains_mock_evidence': c1_origin == 'mock' or c2_origin == 'mock',
        'producer_authenticity_verified': False,
        'data_role': role, 'rule_feature_space': 'processed',
        'membership_basis': 'clean_before_injection',
        'source_row_ids': clean_current.index.tolist(),
        'diagnosed_membership': membership.tolist(),
        'diagnosed_rows': int(membership.sum()),
        'non_affected_rows': int((~membership).sum()),
        'computed_slice_support': measured_support,
        'declared_slice_support': c2.slice_support,
        'slice_support_difference': c2.slice_support - measured_support,
        'candidate_training_authorized': False,
        'release_decision': None,
    }
