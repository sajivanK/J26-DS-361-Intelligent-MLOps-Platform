"""Read-only integrity and in-memory reload check after native path failure."""
import inspect
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from _common import parser, resolve_data_root, checked_json, checked_bytes, require
from c4.data import FEATURE_COLUMNS


def main():
    p = parser(__doc__)
    p.add_argument('--scenario-id', default=
        'adult-ns2026-r004-marginal_multi-high-broadened_slice-label_corrupted_repair')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / args.scenario_id
    directory = run / 'scoped_candidate_validation-v1'
    metadata, metadata_hash = checked_json(directory)
    require(metadata['scenario_id'] == args.scenario_id, 'Candidate belongs to another scenario.')
    contents = checked_bytes(directory / 'candidate.ubj', metadata['model_sha256'])
    print('SAVED MODEL FILE EXISTS: True')
    print('MODEL BYTES:', len(contents))
    print('MODEL / METADATA CHECKSUMS: PASS')
    print('MODEL SHA256:', metadata['model_sha256'])
    print('METADATA SHA256:', metadata_hash)
    print('LOAD API:', inspect.signature(XGBClassifier.load_model))
    restored = XGBClassifier()
    # Bytes are read and verified by Python; the native loader receives no pathname.
    restored.load_model(bytearray(contents))
    require(restored.get_booster().feature_names == list(FEATURE_COLUMNS), 'Model feature schema differs.')
    probe = pd.DataFrame(np.zeros((4, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    predictions = restored.predict(probe)
    probabilities = restored.predict_proba(probe)
    require(probabilities.shape == (4, 2) and np.isfinite(probabilities).all() and
            ((probabilities >= 0) & (probabilities <= 1)).all() and
            np.allclose(probabilities.sum(axis=1), 1), 'Reloaded model probability check failed.')
    require(set(predictions.tolist()) <= {0, 1}, 'Reloaded model predictions are not binary.')
    print('IN-MEMORY LOAD / FEATURE SCHEMA / SYNTHETIC PROBE: PASS')
    print('TRAINING ROWS:', metadata['training_rows'])
    print('NO RETRAINING, REAL-DATA EVALUATION, OR ARTIFACT WRITES EXECUTED.')


if __name__ == '__main__':
    main()
