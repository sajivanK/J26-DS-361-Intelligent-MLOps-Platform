"""Build one verified forecast-training row; never train N1 from this snapshot."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import parser, resolve_data_root, checked_json, checked_bytes, require, sha, save_json, utc_now, git_context
from _future_research import verify_freeze_dependencies
from _scenario_meta import feature_row, validated_targets, TARGET_COLUMNS
from run_scoped_candidate_validation import SELECTED
from c4.contracts.models import UpstreamInput


def main():
    p = parser(__doc__)
    p.add_argument('--save', action='store_true', help='Save the immutable first-scenario engineering snapshot.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    benchmark = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    run = benchmark / SELECTED
    snapshot, snapshot_hash = checked_json(run / 'pre_release_freeze-v1', 'freeze')
    verify_freeze_dependencies(snapshot)
    outcomes, outcomes_hash = checked_json(run / 'future_research-v1', 'outcomes')
    require(outcomes['decision_freeze_sha256'] == snapshot_hash and outcomes['scenario'] == snapshot['scenario'],
            'Future outcomes link to different decision evidence.')
    require(outcomes['future_evaluation_executed'] is True, 'Future labels are not available.')
    require(outcomes['scenario']['split_role'] == 'forecast_train', 'First row must stay in its training fold.')
    future_dir = run / 'future_research-v1'
    for filename, checksum in outcomes['files'].items(): checked_bytes(future_dir / filename, checksum)
    # Independently recompute all primary margins from saved paired predictions.
    predictions = pd.read_csv(future_dir / 'future_predictions.csv', float_precision='round_trip')
    membership = pd.read_csv(future_dir / 'future_membership.csv')
    ids = outcomes['source_row_ids']
    require(predictions['source_row_id'].tolist() == membership['source_row_id'].tolist() == ids and
            len(ids) == len(set(ids)), 'Future prediction/membership IDs differ.')
    require(predictions['target'].isin([0, 1]).all() and
            predictions['champion_prediction'].isin([0, 1]).all() and
            predictions['candidate_prediction'].isin([0, 1]).all(), 'Invalid paired predictions.')
    require(pd.api.types.is_bool_dtype(membership['true_slice_member'].dtype), 'Invalid true membership.')
    targets = validated_targets(outcomes)
    masks = [np.ones(len(predictions), dtype=bool), membership['true_slice_member'].to_numpy(),
             ~membership['true_slice_member'].to_numpy()]
    for key, mask in zip(TARGET_COLUMNS, masks):
        require(mask.any(), 'Target population has no observations.')
        y = predictions.loc[mask, 'target'].to_numpy()
        before = np.mean(predictions.loc[mask, 'champion_prediction'].to_numpy() == y)
        after = np.mean(predictions.loc[mask, 'candidate_prediction'].to_numpy() == y)
        require(abs(float(after-before)-targets[key]) <= 1e-15, 'Target differs from paired predictions.')
    envelope, envelope_hash = checked_json(run / 'pre_release_freeze-v1', 'upstream_simulation')
    require(envelope_hash == snapshot['upstream_simulation_sha256'] and
            envelope['scope'] == 'controlled_offline_simulation', 'Upstream source differs.')
    upstream = UpstreamInput.model_validate(envelope['upstream']).model_dump(mode='json')
    evidence_dir = root / 'c4_artifacts/pre_release/adult' / SELECTED
    evidence, evidence_hash = checked_json(evidence_dir, 'evidence')
    require(evidence_hash == snapshot['pre_release_evidence_sha256'], 'Pre-release metrics differ.')
    features = feature_row(upstream, evidence)
    # Build features without passing outcomes, scenario factors, seeds, true masks or targets.
    feature_frame = pd.DataFrame([features], index=pd.Index([SELECTED], name='scenario_id'))
    target_frame = pd.DataFrame([targets], index=pd.Index([SELECTED], name='scenario_id'))
    panel, panel_hash = checked_json(benchmark / 'admission_panels/middle-strength-targeted-v1', 'panel')
    admissions = []
    for record in panel['records']:
        admission, checksum = checked_json(benchmark / record['scenario_id'], 'admission')
        require(checksum == record['admission_sha256'], 'Panel admission link differs.')
        admissions.append({'scenario_id': record['scenario_id'], 'group_id': admission['scenario']['group_id'],
            'split_role': admission['scenario']['split_role'], 'screen_status': record['screen_status'],
            'simulation_admission': record['simulation_admission'],
            'verified_target_row_available': record['scenario_id'] == SELECTED,
            'admission_sha256': checksum})
    index = pd.DataFrame([{'scenario_id': SELECTED, 'group_id': outcomes['scenario']['group_id'],
        'split_role': outcomes['scenario']['split_role'], 'source_scope': 'controlled_simulated_upstream',
        'decision_freeze_sha256': snapshot_hash, 'outcomes_sha256': outcomes_hash,
        'pre_release_evidence_sha256': evidence_hash, 'upstream_simulation_sha256': envelope_hash}])
    print('VERIFIED META ROWS:', len(feature_frame))
    print('DECISION-TIME FEATURE COUNT:', len(features))
    print('FOLD: forecast_train; GROUP:', outcomes['scenario']['group_id'])
    print('FEATURES / TARGETS SEPARATE; PAIRED TARGET RECOMPUTATION: PASS')
    print('ADMISSION RECORDS RETAINED:', len(admissions))
    print('N1 RESEARCH FIT READY: False — one row cannot support forecasting/calibration evaluation.')
    if not args.save:
        print('PLAN ONLY. Use --save to persist the first-scenario meta-dataset snapshot.')
        return
    output = root / 'c4_artifacts/meta_datasets/adult/adult-simulation-pp1-v2/first-scenario-v1'
    require(not output.exists(), 'Meta-dataset snapshot already exists; preserve it.')
    output.mkdir(parents=True, exist_ok=False)
    frames = {'features.csv': (feature_frame, True), 'targets.csv': (target_frame, True),
              'scenario_index.csv': (index, False), 'admission_table.csv': (pd.DataFrame(admissions), False)}
    hashes = {}
    for filename, (frame, include_index) in frames.items():
        contents = frame.to_csv(index=include_index, lineterminator='\n').encode()
        (output / filename).write_bytes(contents); hashes[filename] = sha(contents)
    metadata = {'purpose': 'first_verified_scenario_meta_dataset', 'created_at_utc': utc_now(), **git_context(),
        'rows': 1, 'group_count': 1, 'fold': 'forecast_train', 'feature_columns': list(features),
        'target_columns': list(TARGET_COLUMNS), 'files': hashes, 'panel_sha256': panel_hash,
        'decision_freeze_sha256': snapshot_hash, 'outcomes_sha256': outcomes_hash,
        'source_implementation_sha256': {Path(__file__).name: sha(Path(__file__).read_bytes()),
            '_scenario_meta.py': sha((Path(__file__).parent / '_scenario_meta.py').read_bytes())},
        'feature_encoding': 'numeric evidence; undefined AUC encoded as zero with a separate missing indicator',
        'excluded_predictors': ['future outcomes', 'true slice membership', 'scenario ID', 'group ID', 'fold',
                                'seeds', 'injection mechanism/strength', 'diagnosis/repair condition labels', 'checksums'],
        'negative_targets_preserved': True, 'failed_admission_targets_imputed': False,
        'N1_research_fit_ready': False, 'release_authorized': False,
        'limitation': 'One completed scenario and seven related admission cases from one training group; no novelty/generalization validation.'}
    verify_freeze_dependencies(snapshot)
    checked_bytes(future_dir / 'outcomes.json', outcomes_hash)
    checksum = save_json(output / 'metadata.json', metadata)
    for filename, expected in hashes.items(): checked_bytes(output / filename, expected)
    print('META-DATASET:', output)
    print('METADATA SHA256:', checksum)
    print('SAVED TABLE CHECKSUMS: PASS')


if __name__ == '__main__':
    main()
