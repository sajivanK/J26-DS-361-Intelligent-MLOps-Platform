"""Plan or evaluate sealed-future research labels for one frozen training case."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (parser, resolve_data_root, checked_json, checked_bytes, require,
                     sha, save_json, utc_now, git_context, C4_ROOT)
from _future_research import verify_freeze_dependencies, future_comparisons
from _all_mechanism_transforms import transform_future_role
from _general_training import group_context
from _portable_model_io import load_champion, load_candidate
from c4.contracts.models import C2Diagnosis
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule
from c4.scenarios.diagnosis import diagnosed_slice_mask
from c4.scenarios.adapter import _load_shared_functions
from c4.scenarios.feature_metadata import load_adult_feature_metadata


def main():
    p = parser(__doc__)
    p.add_argument('--execute', action='store_true', help='Open this scenario future role and save research labels once.')
    p.add_argument('--scenario-id', required=True)
    args = p.parse_args()
    SELECTED = args.scenario_id
    root = resolve_data_root(args.data_root)
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / SELECTED
    group_context(root, SELECTED)
    snapshot, snapshot_hash = checked_json(run / 'pre_release_freeze-v1', 'freeze')
    verify_freeze_dependencies(snapshot)
    require(snapshot['scenario']['scenario_id'] == SELECTED, 'Wrong frozen scenario.')
    row = snapshot['scenario']
    protocol_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, frozen_hash = checked_json(protocol_dir, 'freeze')
    require(frozen_hash == snapshot['protocol_freeze_sha256'], 'Protocol freeze differs.')
    protocol = json.loads(checked_bytes(protocol_dir / 'protocol.json', frozen['protocol_sha256']))
    md = root / 'c4_artifacts/splits/adult' / protocol['source_versions']['split_manifest']
    source, source_hash = checked_json(md, 'manifest')
    require(source_hash == frozen['split_manifest_sha256'], 'Split manifest differs.')
    admission, _ = checked_json(run, 'admission')
    preview = root / 'c4_artifacts/scenario_previews/adult' / admission['preview_id'] / 'release_validation'
    preview_md, _ = checked_json(preview)
    shared, drift_hash = _load_shared_functions()
    require(preview_md['transformation']['shared_drift_types_sha256'] ==
            (None if row['mechanism'] == 'none' else drift_hash), 'Shared drift implementation differs.')
    feature_metadata = load_adult_feature_metadata()
    require(feature_metadata.source_sha256 == preview_md['transformation']['feature_metadata_sha256'], 'Feature metadata differs.')
    config = ScenarioConfig(mechanism=row['mechanism'], magnitude=row['magnitude'],
        features=tuple(row['features']), interaction_pairs=tuple(tuple(pair) for pair in row['interaction_pairs']),
        seed=row['drift_seeds']['sealed_future'])
    print('FROZEN DECISION EVIDENCE / DEPENDENCIES: PASS')
    print('SHARED DRIFT CODE / FEATURE METADATA: PASS')
    print('SCENARIO:', SELECTED, 'FOLD:', row['split_role'])
    print('FUTURE SEED:', config.seed)
    if not args.execute:
        print('PLAN ONLY: future observations and outcomes have not been parsed or evaluated.')
        return
    output = run / 'future_research-v1'
    require(not output.exists(), 'Future research version already claimed; preserve it.')
    output.mkdir(parents=True, exist_ok=False)
    code_hashes = {path.name: sha(path.read_bytes()) for path in
                  [Path(__file__), C4_ROOT / 'scripts/_future_research.py', C4_ROOT / 'scripts/_general_training.py', C4_ROOT / 'scripts/_all_mechanism_transforms.py', C4_ROOT / 'scripts/_portable_model_io.py']}
    save_json(output / 'claim.json', {'scenario_id': SELECTED, 'created_at_utc': utc_now(), **git_context(),
        'decision_freeze_sha256': snapshot_hash, 'future_config': config.model_dump(mode='json'),
        'evaluator_source_sha256': code_hashes, 'shared_drift_types_sha256': drift_hash,
        'scope': 'sealed_future_research_targets_only_not_decision_time_evidence'})
    try:
        test_bytes = checked_bytes(root / 'uci_adult_data/test.csv', source['source_checksums']['test_csv_sha256'])
        ids = source['partitions']['sealed_future']['source_row_ids']
        require(len(ids) == len(set(ids)), 'Future IDs are repeated.')
        for role in ('repair', 'release_validation'):
            require(not set(ids) & set(source['partitions'][role]['source_row_ids']), 'Future roles overlap.')
        selected = set(ids)
        clean = pd.read_csv(io.BytesIO(test_bytes), skiprows=lambda line: line > 0 and line-1 not in selected)
        require(len(clean) == len(ids), 'Future source record layout differs.')
        clean.index = sorted(selected)
        clean = clean.loc[ids].copy()
        true = protocol['true_slice']
        true_rule = SliceRule(name=true['name'], feature=true['feature'], operator=true['operator'], value=true['resolved_threshold'])
        transformed, true_mask = transform_future_role(clean, config=config, true_rule=true_rule, shared=shared,
                                                       feature_types=feature_metadata.feature_types)
        require(_load_shared_functions()[1] == drift_hash, 'Shared drift code changed during execution.')
        champion_dir = root / 'c4_artifacts/champion/adult' / protocol['source_versions']['champion']
        champion, champion_md = load_champion(champion_dir)
        candidate, candidate_md = load_candidate(run / 'scoped_candidate_validation-v1')
        require(champion_md['model_sha256'] == candidate_md['champion_model_sha256'] == frozen['champion_model_sha256'], 'Models differ.')
        upstream, _ = checked_json(run / 'pre_release_freeze-v1', 'upstream_simulation')
        diagnosed = diagnosed_slice_mask(clean, C2Diagnosis.model_validate(upstream['upstream']['c2_diagnosis']),
                                         rule_feature_space='processed')
        comparisons, targets, predictions = future_comparisons(champion, candidate, transformed, true_mask, diagnosed)
        membership = pd.DataFrame({'source_row_id': clean.index,
            'true_slice_member': true_mask.to_numpy(), 'diagnosed_slice_member': diagnosed.to_numpy()})
        files = {'clean_future.csv': clean.to_csv(index=True, index_label='source_row_id', lineterminator='\n').encode(),
                 'transformed_future.csv': transformed.to_csv(index=True, index_label='source_row_id', lineterminator='\n').encode(),
                 'future_membership.csv': membership.to_csv(index=False, lineterminator='\n').encode(),
                 'future_predictions.csv': predictions.to_csv(index=False, lineterminator='\n').encode()}
        for filename, contents in files.items():
            (output / filename).write_bytes(contents)
        verify_freeze_dependencies(snapshot)
        checked_bytes(run / 'pre_release_freeze-v1/freeze.json', snapshot_hash)
        report = {'purpose': 'sealed_future_research_targets',
            'model_io_mode': 'verified_bytearray_candidate_and_champion_load', 'created_at_utc': utc_now(), **git_context(),
            'scenario': row, 'decision_freeze_sha256': snapshot_hash,
            'evaluation_role': 'sealed_future', 'source_row_ids': ids,
            'split_manifest_sha256': source_hash, 'source_test_csv_sha256': sha(test_bytes),
            'configuration': config.model_dump(mode='json'), 'configuration_sha256': config.configuration_sha256(),
            'shared_drift_types_sha256': drift_hash, 'feature_metadata_sha256': feature_metadata.source_sha256,
            'champion_model_sha256': champion_md['model_sha256'], 'candidate_model_sha256': candidate_md['model_sha256'],
            'membership_basis': 'clean_before_future_injection', 'primary_slice_basis': 'true_injected_slice',
            'target_units': 'accuracy_margin_fraction', 'target_sign': 'candidate_minus_champion',
            'primary_targets': targets, 'comparisons': comparisons,
            'rows': len(clean), 'true_affected_rows': int(true_mask.sum()), 'diagnosed_rows': int(diagnosed.sum()),
            'files': {filename: sha(contents) for filename, contents in files.items()},
            'future_evaluation_executed': True, 'training_executed': False, 'controller_received_future_targets': False,
            'production_release_authorized': False, 'N1_N4_validation_complete': False,
            'source_scope': 'one_training_group_controlled_simulated_upstream_not_real_producer_validation'}
        checksum = save_json(output / 'outcomes.json', report)
        for filename, expected in report['files'].items():
            checked_bytes(output / filename, expected)
        print('FUTURE ROWS:', len(clean), 'TRUE AFFECTED:', int(true_mask.sum()), 'DIAGNOSED:', int(diagnosed.sum()))
        print('PRIMARY FUTURE TARGETS:', json.dumps(targets, indent=2))
        print('COMPARISONS:', json.dumps(comparisons, indent=2))
        print('PRESERVATION / OUTPUT CHECKSUMS / FROZEN DEPENDENCIES: PASS')
        print('REPORT:', output / 'outcomes.json')
        print('REPORT SHA256:', checksum)
        print('RESEARCH LABELS ONLY; RELEASE AUTHORIZED: False')
    except Exception as error:
        save_json(output / 'failure.json', {'scenario_id': SELECTED, 'exception_type': type(error).__name__,
            'message': str(error), 'release_authorized': False})
        raise


if __name__ == '__main__':
    main()
