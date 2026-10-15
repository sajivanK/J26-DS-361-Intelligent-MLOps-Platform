"""Verify all 32 frozen forecasting-training groups; never train or predict."""
import io
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (parser, name, resolve_data_root, checked_json, checked_bytes,
                     require, save_json, sha, utc_now, git_context)
from _general_training import select_admitted_cases
from _future_research import verify_freeze_dependencies
from _scenario_meta import feature_row, validated_targets, TARGET_COLUMNS
from c4.contracts.models import UpstreamInput

VERSION = 'adult-simulation-pp1-v2'
DEFAULT_GROUPS = (
    *(f'adult-ns123-{r}' for r in ('r003','r006','r008','r011','r013','r014','r018','r019')),
    *(f'adult-ns2026-{r}' for r in ('r001','r003','r004','r012','r013','r016','r017','r018'))
)


def combine_tables(features, targets, indices):
    require(bool(features) and len(features) == len(targets) == len(indices), 'Missing group tables.')
    columns = list(features[0].columns)
    for f, t, i in zip(features, targets, indices):
        require(list(f.columns) == columns and len(columns) == 75, 'Decision-time feature schema differs.')
        require(not set(TARGET_COLUMNS) & set(columns), 'Future targets cannot be predictors.')
        require(list(t.columns) == list(TARGET_COLUMNS), 'Target schema differs.')
        require(f.index.equals(t.index) and f.index.tolist() == i['scenario_id'].tolist(), 'Scenario order differs.')
        require(i['split_role'].eq('forecast_train').all(), 'Nontraining rows are forbidden.')
        require(np.isfinite(f.to_numpy(dtype=float)).all() and
                np.isfinite(t.to_numpy(dtype=float)).all(), 'Nonfinite features or targets.')
    f = pd.concat(features)
    t = pd.concat(targets)
    i = pd.concat(indices, ignore_index=True)
    require(f.index.is_unique and t.index.is_unique and i['scenario_id'].is_unique,
            'Duplicate scenario rows cannot be combined.')
    return f, t, i


def read_table(directory, filename, metadata):
    data = checked_bytes(directory / filename, metadata['files'][filename])
    return pd.read_csv(io.BytesIO(data), float_precision='round_trip')


def main():
    p = parser(__doc__)
    p.add_argument('--groups', nargs='+', type=name, default=None)
    p.add_argument('--snapshot-version', type=name, default='forecast-training-all32-audit-v1')
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    frozen, freeze_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'], 'Manifest link differs.')
    training_groups = sorted(g['group_id'] for g in manifest['groups'] if g['split_role'] == 'forecast_train')
    require(len(training_groups) == len(set(training_groups)) == 32, 'Frozen training group inventory differs.')
    args.groups = training_groups if args.groups is None else args.groups
    require(set(args.groups) == set(training_groups), 'This final audit requires all frozen training groups.')
    require(len(args.groups) == len(set(args.groups)) and args.groups, 'Requested groups must be unique and nonempty.')
    require(frozen['full_execution_enabled'] is False, 'Unexpected full execution scope.')
    runs = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    meta_root = root / 'c4_artifacts/meta_datasets/adult' / VERSION
    fs, ts, ix, admission_rows, group_rows = [], [], [], [], []
    source_links = {}
    frozen_snapshots = []
    mechanisms = []
    for group in args.groups:
        group_dir = runs / 'group_admissions' / group
        admission, admission_hash = checked_json(group_dir / 'v1', 'group_admissions')
        require(admission['freeze_sha256'] == freeze_hash, 'Admission protocol differs.')
        selected = select_admitted_cases(manifest, admission, group)
        by_id = {r['scenario_id']: r for r in selected}
        for record in admission['records']:
            raw, raw_hash = checked_json(runs / record['scenario_id'], 'admission')
            require(raw_hash == record['admission_sha256'] and raw['scenario']['group_id'] == group,
                    'Original admission record differs.')
            admission_rows.append({**record, 'verified_target_row_available': record['scenario_id'] in by_id})
            source_links[str(runs / record['scenario_id'] / 'admission.json')] = raw_hash
        source_links[str(group_dir / 'v1/group_admissions.json')] = admission_hash
        if not selected:
            group_rows.append({'group_id': group, 'rows': 0, 'blocked': 37, 'status': 'ALL_BLOCKED'})
            continue
        candidates = [meta_root / (group + suffix) for suffix in ('-v1', '-general-v1', '-all-mechanism-v1', '-portable-v1')]
        candidates = [d for d in candidates if (d / 'metadata.json').is_file()]
        require(len(candidates) == 1, 'Exactly one authoritative group snapshot is required: ' + group)
        directory = candidates[0]
        metadata, metadata_hash = checked_json(directory)
        require(metadata['group_id'] == group and metadata['fold'] == 'forecast_train' and
                metadata['group_admissions_sha256'] == admission_hash, 'Group snapshot identity differs.')
        pipelines = [group_dir / v for v in ('pipeline-v3', 'general-pipeline-v1', 'all-mechanism-pipeline-v1', 'portable-pipeline-v1')]
        verified = []
        for d in pipelines:
            if (d / 'group_pipeline.json').is_file():
                payload, checksum = checked_json(d, 'group_pipeline')
                if checksum == metadata['group_pipeline_sha256']:
                    verified.append((d, payload, checksum))
        require(len(verified) == 1, 'Snapshot pipeline dependency is missing or ambiguous.')
        pipeline_dir, pipeline, pipeline_hash = verified[0]
        require(pipeline['group_id'] == group and pipeline['group_admissions_sha256'] == admission_hash and
                pipeline['completed_scenarios'] == len(selected), 'Completed pipeline differs.')
        source_links[str(pipeline_dir / 'group_pipeline.json')] = pipeline_hash
        source_links[str(directory / 'metadata.json')] = metadata_hash
        for filename, expected in metadata['files'].items():
            checked_bytes(directory / filename, expected)
            source_links[str(directory / filename)] = expected
        f = read_table(directory, 'features.csv', metadata).set_index('scenario_id')
        t = read_table(directory, 'targets.csv', metadata).set_index('scenario_id')
        i = read_table(directory, 'scenario_index.csv', metadata)
        require(set(f.index) == set(by_id) and metadata['rows'] == len(selected), 'Admitted scenario inventory differs.')
        require(i['group_id'].eq(group).all(), 'Scenario group differs.')
        for sid in f.index:
            row = i.loc[i.scenario_id == sid]
            require(len(row) == 1, 'Scenario index is missing or duplicated.')
            row = row.iloc[0]
            run = runs / sid
            snapshot, snapshot_hash = checked_json(run / 'pre_release_freeze-v1', 'freeze')
            verify_freeze_dependencies(snapshot)
            outcomes, outcome_hash = checked_json(run / 'future_research-v1', 'outcomes')
            envelope, envelope_hash = checked_json(run / 'pre_release_freeze-v1', 'upstream_simulation')
            evidence_dir = root / 'c4_artifacts/pre_release/adult' / sid
            evidence, evidence_hash = checked_json(evidence_dir, 'evidence')
            require(snapshot_hash == row.decision_freeze_sha256 == outcomes['decision_freeze_sha256'] and
                    outcome_hash == row.outcomes_sha256 and envelope_hash == row.upstream_simulation_sha256 ==
                    snapshot['upstream_simulation_sha256'] and evidence_hash == row.pre_release_evidence_sha256 ==
                    snapshot['pre_release_evidence_sha256'], 'Scenario evidence links differ.')
            require(row.admission_sha256 == by_id[sid]['admission_sha256'], 'Index admission link differs.')
            require(outcomes['scenario']['scenario_id'] == sid and outcomes['scenario']['group_id'] == group and
                    outcomes['scenario']['split_role'] == 'forecast_train', 'Outcome scope differs.')
            require(snapshot['scenario'] == outcomes['scenario'] and
                    snapshot['protocol_freeze_sha256'] == freeze_hash, 'Scenario protocol freeze differs.')
            expected_f = feature_row(UpstreamInput.model_validate(envelope['upstream']).model_dump(mode='json'), evidence)
            require(list(f.columns) == list(expected_f), 'Feature columns differ from decision-time reconstruction.')
            require(np.allclose(f.loc[sid].to_numpy(dtype=float), list(expected_f.values()), rtol=0, atol=1e-15),
                    'Stored features differ from decision-time evidence.')
            expected_t = validated_targets(outcomes)
            future = run / 'future_research-v1'
            predicted = pd.read_csv(io.BytesIO(checked_bytes(future / 'future_predictions.csv',
                outcomes['files']['future_predictions.csv'])), float_precision='round_trip')
            membership = pd.read_csv(io.BytesIO(checked_bytes(future / 'future_membership.csv',
                outcomes['files']['future_membership.csv'])))
            require(predicted.source_row_id.is_unique and predicted.source_row_id.tolist() ==
                    membership.source_row_id.tolist() == outcomes['source_row_ids'], 'Future IDs differ.')
            require(pd.api.types.is_bool_dtype(membership.true_slice_member.dtype), 'Invalid true membership type.')
            masks = [np.ones(len(predicted), dtype=bool), membership.true_slice_member.to_numpy(),
                     ~membership.true_slice_member.to_numpy()]
            for key, mask in zip(TARGET_COLUMNS, masks):
                require(mask.any(), 'Empty target population.')
                y = predicted.loc[mask, 'target'].to_numpy()
                margin = float(np.mean(predicted.loc[mask, 'candidate_prediction'].to_numpy() == y) -
                               np.mean(predicted.loc[mask, 'champion_prediction'].to_numpy() == y))
                require(abs(margin - expected_t[key]) <= 1e-15 and abs(margin - t.loc[sid, key]) <= 1e-15,
                        'Future target differs from saved paired predictions.')
            mechanisms.append(by_id[sid]['mechanism'])
            frozen_snapshots.append(snapshot)
            for path, checksum in [(run / 'pre_release_freeze-v1/freeze.json', snapshot_hash),
                                   (future / 'outcomes.json', outcome_hash)]:
                source_links[str(path)] = checksum
        fs.append(f); ts.append(t); ix.append(i)
        group_rows.append({'group_id': group, 'rows': len(f), 'blocked': 37-len(f), 'status': 'COMPLETE'})
        print('VERIFIED GROUP:', group, 'META ROWS:', len(f), 'BLOCKED:', 37-len(f))
    features, targets, index = combine_tables(fs, ts, ix)
    admitted_mechanisms = pd.Series(mechanisms).value_counts().to_dict()
    variable_count = int(features.nunique(dropna=False).gt(1).sum())
    print('COMBINED META ROWS:', len(features), 'GROUPS AUDITED:', len(group_rows))
    usable_groups = int(index['group_id'].nunique())
    print('GROUPS WITH TARGET ROWS:', usable_groups, 'ALL-BLOCKED GROUPS:', len(group_rows)-usable_groups)
    print('DECISION-TIME FEATURES:', len(features.columns), 'VARYING FEATURES:', variable_count)
    print('ADMISSION RECORDS RETAINED:', len(admission_rows), 'BLOCKED:', sum(g['blocked'] for g in group_rows))
    print('ADMITTED MECHANISM COUNTS:', admitted_mechanisms)
    print('FEATURE RECONSTRUCTION / PAIRED TARGET RECOMPUTATION / FOLD CHECKS: PASS')
    print('N1 RESEARCH FIT READY: False; final feature and training-specification review remains pending.')
    if not args.save:
        print('PLAN ONLY. No models fitted, predictions regenerated, or snapshot written.')
        return
    output = meta_root / args.snapshot_version
    require(not output.exists(), 'Combined snapshot exists; never overwrite it.')
    for snapshot in frozen_snapshots:
        verify_freeze_dependencies(snapshot)
    for path, expected in source_links.items():
        checked_bytes(path, expected)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, frame, include_index in [('features.csv', features, True), ('targets.csv', targets, True),
            ('scenario_index.csv', index, False), ('admission_table.csv', pd.DataFrame(admission_rows), False),
            ('group_summary.csv', pd.DataFrame(group_rows), False)]:
        contents = frame.to_csv(index=include_index, lineterminator='\n').encode()
        (output / filename).write_bytes(contents); files[filename] = sha(contents)
    note = '\n'.join([
        '# C4 combined training evidence audit', '',
        f'- Rows: {len(features)}', f'- Groups audited: {len(group_rows)}',
        f'- Decision-time features: {len(features.columns)}; varying features: {variable_count}',
        f'- Admission records: {len(admission_rows)}; blocked: {sum(g["blocked"] for g in group_rows)}',
        f'- Admitted mechanisms: {admitted_mechanisms}',
        '- Feature reconstruction and paired future-target recomputation: PASS.',
        '- No model fitting or future prediction regeneration was performed.',
        '- Simulated upstream evidence; production release remains unauthorized.',
        '- Group count is not proof of statistical independence; source pools are shared.',
        '- N1 fitting and calibration readiness remain pending.', '',
        '| Group | Meta rows | Blocked | Status |', '|---|---:|---:|---|',
        *[f'| {g["group_id"]} | {g["rows"]} | {g["blocked"]} | {g["status"]} |' for g in group_rows],
        '', f'Protocol freeze SHA256: `{freeze_hash}`', ''])
    (output / 'C4_TRAINING_AUDIT.md').write_bytes(note.encode())
    files['C4_TRAINING_AUDIT.md'] = sha(note.encode())
    summary = {'purpose': 'verified_combined_forecasting_training_snapshot', 'created_at_utc': utc_now(),
        **git_context(), 'rows': len(features), 'groups': args.groups, 'fold': 'forecast_train',
        'feature_columns': list(features.columns), 'target_columns': list(TARGET_COLUMNS),
        'varying_features': variable_count, 'admitted_mechanism_counts': admitted_mechanisms,
        'groups_with_target_rows': usable_groups, 'all_blocked_groups': len(group_rows)-usable_groups,
        'admission_cases': len(admission_rows), 'blocked_cases': sum(g['blocked'] for g in group_rows),
        'source_links': source_links, 'protocol_freeze_sha256': freeze_hash, 'files': files,
        'source_script_sha256': sha(Path(__file__).read_bytes()), 'N1_research_fit_ready': False,
        'target_based_scenario_filtering': False, 'model_fitting_executed': False,
        'future_predictions_regenerated': False, 'release_authorized': False}
    digest = save_json(output / 'metadata.json', summary)
    print('SAVED COMBINED SNAPSHOT:', output)
    print('METADATA SHA256:', digest)


if __name__ == '__main__':
    main()
