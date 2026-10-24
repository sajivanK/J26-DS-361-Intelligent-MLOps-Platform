"""Save all completed admitted rows and all 37 admission records separately."""
from pathlib import Path
import io
import numpy as np
import pandas as pd

from _common import parser, resolve_data_root, checked_json, checked_bytes, require, sha, save_json, utc_now, git_context, name
from _general_training import select_admitted_cases, group_context
from _future_research import verify_freeze_dependencies
from _scenario_meta import feature_row, validated_targets, TARGET_COLUMNS
from run_portable_training_pipeline import completed_case
from c4.contracts.models import UpstreamInput


def main():
    p = parser(__doc__)
    p.add_argument('--group-id', type=name, required=True)
    p.add_argument('--save', action='store_true')
    p.add_argument('--pipeline-version', type=name, default='portable-pipeline-v1')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    GROUP = args.group_id
    benchmark = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    group_dir = benchmark / 'group_admissions' / GROUP
    report, report_hash = checked_json(group_dir / 'v1', 'group_admissions')
    pipeline, pipeline_hash = checked_json(group_dir / args.pipeline_version, 'group_pipeline')
    require(pipeline['group_admissions_sha256'] == report_hash, 'Group pipeline admission link differs.')
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, frozen_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'] and
            report['freeze_sha256'] == frozen_hash, 'Group/manifest protocol differs.')
    rows = select_admitted_cases(manifest, report, GROUP)
    require(rows, 'No admitted cases; retain admission-only evidence without inventing targets.')
    require(pipeline['group_id'] == GROUP and pipeline['completed_scenarios'] == len(rows),
            'Completed pipeline group/count differs.')
    features, targets, index, links = [], [], [], []
    for row in rows:
        group_context(root, row['scenario_id'])
        record = completed_case(root, row)
        require(record is not None and record in pipeline['records'], 'A complete admitted scenario is missing.')
        run = benchmark / row['scenario_id']
        snapshot, snapshot_hash = checked_json(run / 'pre_release_freeze-v1', 'freeze')
        outcomes, outcomes_hash = checked_json(run / 'future_research-v1', 'outcomes')
        envelope, envelope_hash = checked_json(run / 'pre_release_freeze-v1', 'upstream_simulation')
        evidence_dir = root / 'c4_artifacts/pre_release/adult' / row['scenario_id']
        evidence, evidence_hash = checked_json(evidence_dir, 'evidence')
        require(evidence_hash == snapshot['pre_release_evidence_sha256'] and
                envelope_hash == snapshot['upstream_simulation_sha256'], 'Decision-time evidence links differ.')
        require(envelope['scope'] == 'controlled_offline_simulation', 'Wrong source scope.')
        feature = feature_row(UpstreamInput.model_validate(envelope['upstream']).model_dump(mode='json'), evidence)
        target = validated_targets(outcomes)
        directory = run / 'future_research-v1'
        predicted = pd.read_csv(io.BytesIO(checked_bytes(directory / 'future_predictions.csv', outcomes['files']['future_predictions.csv'])),
                                float_precision='round_trip')
        membership = pd.read_csv(io.BytesIO(checked_bytes(directory / 'future_membership.csv', outcomes['files']['future_membership.csv'])))
        require(predicted['source_row_id'].tolist() == membership['source_row_id'].tolist() == outcomes['source_row_ids'],
                'Future membership IDs differ.')
        require(pd.api.types.is_bool_dtype(membership['true_slice_member'].dtype), 'True future membership must be boolean.')
        masks = [np.ones(len(predicted), dtype=bool), membership['true_slice_member'].to_numpy(),
                 ~membership['true_slice_member'].to_numpy()]
        for key, mask in zip(TARGET_COLUMNS, masks):
            require(mask.any(), 'Empty primary target population.')
            y = predicted.loc[mask, 'target'].to_numpy()
            margin = float(np.mean(predicted.loc[mask, 'candidate_prediction'].to_numpy() == y) -
                           np.mean(predicted.loc[mask, 'champion_prediction'].to_numpy() == y))
            require(abs(margin-target[key]) <= 1e-15, 'Target differs from saved paired predictions.')
        features.append({'scenario_id': row['scenario_id'], **feature})
        targets.append({'scenario_id': row['scenario_id'], **target})
        index.append({'scenario_id': row['scenario_id'], 'group_id': GROUP, 'split_role': 'forecast_train',
            'source_scope': 'controlled_simulated_upstream', 'decision_freeze_sha256': snapshot_hash,
            'outcomes_sha256': outcomes_hash, 'upstream_simulation_sha256': envelope_hash,
            'pre_release_evidence_sha256': evidence_hash, 'admission_sha256': row['admission_sha256']})
        links.append((snapshot, run / 'pre_release_freeze-v1/freeze.json', snapshot_hash,
                      directory / 'outcomes.json', outcomes_hash))
    feature_frame = pd.DataFrame(features).set_index('scenario_id')
    target_frame = pd.DataFrame(targets).set_index('scenario_id')
    require(feature_frame.index.equals(target_frame.index) and feature_frame.notna().all().all(), 'Feature/target rows differ.')
    require(all(len(row)==len(features[0]) for row in features), 'Feature schema differs between cases.')
    admission_table = []
    for row in report['records']:
        admission, checksum = checked_json(benchmark / row['scenario_id'], 'admission')
        require(checksum == row['admission_sha256'], 'Admission record differs.')
        admission_table.append({'scenario_id': row['scenario_id'], 'group_id': GROUP, 'split_role': 'forecast_train',
            'screen_status': row['screen_status'], 'simulation_admission': row['simulation_admission'],
            'verified_target_row_available': row['scenario_id'] in feature_frame.index, 'admission_sha256': checksum})
    print('VERIFIED META ROWS:', len(feature_frame))
    print('DECISION-TIME FEATURES:', len(feature_frame.columns))
    print('GROUP COUNT: 1; FOLD: forecast_train')
    print('ADMISSION RECORDS RETAINED:', len(admission_table))
    print('FEATURES / TARGETS SEPARATE; PAIRED TARGET RECOMPUTATION: PASS')
    print('N1 RESEARCH FIT READY: False')
    if not args.save:
        print('PLAN ONLY. Use --save to create a new group snapshot.')
        return
    output = root / 'c4_artifacts/meta_datasets/adult/adult-simulation-pp1-v2' / (GROUP + '-portable-v1')
    require(not output.exists(), 'Group meta-dataset already exists; preserve it.')
    output.mkdir(parents=True, exist_ok=False)
    frames = {'features.csv': (feature_frame, True), 'targets.csv': (target_frame, True),
              'scenario_index.csv': (pd.DataFrame(index), False), 'admission_table.csv': (pd.DataFrame(admission_table), False)}
    files = {}
    for filename, (frame, include_index) in frames.items():
        contents = frame.to_csv(index=include_index, lineterminator='\n').encode()
        (output / filename).write_bytes(contents); files[filename] = sha(contents)
    for snapshot, freeze_path, freeze_hash, outcome_path, outcome_hash in links:
        verify_freeze_dependencies(snapshot)
        checked_bytes(freeze_path, freeze_hash); checked_bytes(outcome_path, outcome_hash)
    metadata = {'purpose': 'verified_training_group_meta_dataset', 'created_at_utc': utc_now(), **git_context(),
        'rows': len(features), 'group_count': 1, 'group_id': GROUP, 'fold': 'forecast_train',
        'feature_columns': feature_frame.columns.tolist(), 'target_columns': list(TARGET_COLUMNS),
        'files': files, 'group_admissions_sha256': report_hash, 'group_pipeline_sha256': pipeline_hash,
        'feature_builder_sha256': sha((Path(__file__).parent / '_scenario_meta.py').read_bytes()),
        'source_code_sha256': sha(Path(__file__).read_bytes()), 'negative_targets_retained': True,
        'failed_admission_targets_imputed': False, 'N1_research_fit_ready': False, 'release_authorized': False,
        'feature_encoding': '75 numeric decision-time evidence fields; undefined AUC uses zero plus a missing flag',
        'excluded_predictors': ['future outcomes', 'true membership', 'scenario/group IDs', 'folds', 'seeds',
                                'experimental condition labels', 'checksums'],
        'limitation': 'One related forecasting-training group; no conformal, policy or outer-test evaluation.'}
    checksum = save_json(output / 'metadata.json', metadata)
    for filename, expected in files.items(): checked_bytes(output / filename, expected)
    print('META-DATASET:', output)
    print('METADATA SHA256:', checksum)
    print('SAVED TABLE CHECKSUMS: PASS')


if __name__ == '__main__':
    main()
