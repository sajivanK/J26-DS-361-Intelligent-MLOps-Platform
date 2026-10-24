"""Audit all frozen outer decisions and paired future targets; retain blocked groups.

This does not refit, reselect a policy, or reevaluate candidate predictions on data.
Targets are recomputed from the saved paired prediction tables.
"""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from _common import (parser, resolve_data_root, checked_json, checked_bytes, require,
                     save_json, sha, utc_now, git_context)
from _outer_pipeline import (VERSION, inventory, runs, verified_phase,
                             verify_pre_release_registry, verify_freeze_dependencies)
from run_outer_candidate_batch import validated_case, frozen_case, completed_case
from _scenario_meta import feature_row, validated_targets, TARGET_COLUMNS
from c4.contracts.models import UpstreamInput
from c4.evaluation.comparison_decisions import METHOD_IDS


def recompute_targets(predicted, membership, source_ids, target):
    require(predicted['source_row_id'].tolist() == membership['source_row_id'].tolist() == source_ids,
            'Saved future paired prediction/membership identity differs.')
    require(len(source_ids) == len(set(source_ids)) and len(source_ids) > 0,
            'Repeated/empty future source IDs.')
    require(pd.api.types.is_bool_dtype(membership['true_slice_member'].dtype), 'True membership must be boolean.')
    for column in ('target', 'candidate_prediction', 'champion_prediction'):
        require(set(predicted[column].unique()) <= {0,1}, 'Future labels/predictions must be binary.')
    affected = membership['true_slice_member'].to_numpy()
    masks = [np.ones(len(predicted), dtype=bool), affected, ~affected]
    for key, mask in zip(TARGET_COLUMNS, masks):
        require(mask.any(), 'Empty primary future population.')
        y = predicted.loc[mask, 'target'].to_numpy()
        actual = float(np.mean(predicted.loc[mask, 'candidate_prediction'].to_numpy() == y) -
                       np.mean(predicted.loc[mask, 'champion_prediction'].to_numpy() == y))
        require(abs(actual-target[key]) <= 1e-15, 'Primary target differs from saved paired predictions.')


def collect(root):
    manifest, groups, reports, selected = inventory(root)
    registry, registry_hash = verify_pre_release_registry(root)
    features, targets, index, decisions = [], [], [], []
    forecasts = {key:[] for key in ('ORIGINAL','A1','A2','A3','linear')}
    links = {}
    for number, row in enumerate(selected, 1):
        sid = row['scenario_id']
        print(f'AUDITING {number}/56: {sid}', flush=True)
        require(validated_case(root, row) is not None and frozen_case(root,row) is not None and
                completed_case(root,row) is not None, 'All admitted outer stages must be complete.')
        run = runs(root)/sid
        snapshot, freeze_hash = checked_json(run/'pre_release_freeze-v1', 'freeze')
        verify_freeze_dependencies(snapshot)
        payload, payload_hash = checked_json(run/'pre_release_freeze-v1', 'comparison_forecasts')
        original, original_hash = checked_json(run/'pre_release_freeze-v1', 'n1_forecasts')
        envelope, envelope_hash = checked_json(run/'pre_release_freeze-v1', 'upstream_simulation')
        evidence, evidence_hash = checked_json(root/'c4_artifacts/pre_release/adult'/sid, 'evidence')
        outcome, outcome_hash = checked_json(run/'future_research-v1', 'outcomes')
        require(freeze_hash == registry['scenario_freeze_sha256'][sid] and
                payload_hash == registry['comparison_forecast_sha256'][sid] == snapshot['comparison_forecasts_sha256'] and
                original_hash == snapshot['n1_forecasts_sha256'] and
                envelope_hash == snapshot['upstream_simulation_sha256'] and
                evidence_hash == snapshot['pre_release_evidence_sha256'], 'Outer dependency link differs.')
        feature = feature_row(UpstreamInput.model_validate(envelope['upstream']).model_dump(mode='json'), evidence)
        require(feature == original['decision_time_features'] == payload['decision_time_features'] and len(feature) == 75,
                'Decision-time feature reconstruction differs.')
        require(payload['scenario_id'] == sid and payload['future_targets_included'] is False and
                set(payload['decisions']) == set(METHOD_IDS) and
                payload['forecasts']['ORIGINAL'] == original['forecasts']['lightgbm'], 'Frozen method inventory differs.')
        target = validated_targets(outcome)
        directory = run/'future_research-v1'
        predicted = pd.read_csv(io.BytesIO(checked_bytes(directory/'future_predictions.csv', outcome['files']['future_predictions.csv'])),
                                float_precision='round_trip')
        membership = pd.read_csv(io.BytesIO(checked_bytes(directory/'future_membership.csv', outcome['files']['future_membership.csv'])))
        recompute_targets(predicted, membership, outcome['source_row_ids'], target)
        features.append({'scenario_id':sid, **feature})
        targets.append({'scenario_id':sid, **target})
        index.append({'scenario_id':sid, 'group_id':row['group_id'], 'split_role':'outer_test',
            'source_scope':'controlled_simulated_upstream', 'decision_freeze_sha256':freeze_hash,
            'outcomes_sha256':outcome_hash, 'admission_sha256':row['admission_sha256'],
            'explicit_hard_gates_json':json.dumps(payload['explicit_hard_gates'],sort_keys=True)})
        for method in METHOD_IDS:
            decision = payload['decisions'][method]
            require(decision['decision'] in {'AUTO_DEPLOY','HUMAN_REVIEW','ESCALATE'} and
                    decision['comparison'] == method and decision.get('release_authorized') is False,
                    'Invalid or production-authorized research decision.')
            decisions.append({'scenario_id':sid, 'group_id':row['group_id'], 'comparison':method,
                              'decision':decision['decision'], 'reason':decision['reason']})
        for key in forecasts:
            values = original['forecasts']['linear'] if key == 'linear' else payload['forecasts'][key]
            forecasts[key].append({'scenario_id':sid, **values})
        for path, digest in ((run/'pre_release_freeze-v1/freeze.json', freeze_hash),
                             (run/'pre_release_freeze-v1/comparison_forecasts.json',payload_hash),
                             (run/'future_research-v1/outcomes.json',outcome_hash)):
            links[str(path)] = digest
    selected_ids = {r['scenario_id'] for r in selected}
    admissions, group_summary = [], []
    for gid in groups:
        report = reports[gid]['report']
        count = sum(r['group_id'] == gid for r in selected)
        group_summary.append({'group_id':gid,'split_role':'outer_test','admission_records':37,
                              'target_rows':count,'blocked':37-count,'all_blocked':count==0})
        for row in report['records']:
            _, admission_hash = checked_json(runs(root)/row['scenario_id'], 'admission')
            require(admission_hash == row['admission_sha256'], 'Admission checksum differs.')
            admissions.append({'scenario_id':row['scenario_id'], 'group_id':gid, 'split_role':'outer_test',
                'screen_status':row['screen_status'], 'simulation_admission':row['simulation_admission'],
                'verified_target_row_available':row['scenario_id'] in selected_ids, 'admission_sha256':admission_hash})
    frames = {'features.csv':pd.DataFrame(features).set_index('scenario_id'),
              'targets.csv':pd.DataFrame(targets).set_index('scenario_id'),
              'scenario_index.csv':pd.DataFrame(index).set_index('scenario_id'),
              'decisions.csv':pd.DataFrame(decisions), 'admissions.csv':pd.DataFrame(admissions),
              'group_summary.csv':pd.DataFrame(group_summary)}
    frames.update({'forecasts_'+key+'.csv':pd.DataFrame(values).set_index('scenario_id') for key,values in forecasts.items()})
    require(len(frames['features.csv']) == len(frames['targets.csv']) == 56 and
            frames['features.csv'].index.equals(frames['targets.csv'].index) and
            np.isfinite(frames['features.csv'].to_numpy(dtype=float)).all() and
            np.isfinite(frames['targets.csv'].to_numpy(dtype=float)).all() and
            len(frames['decisions.csv']) == 56*len(METHOD_IDS) and len(admissions)==592,
            'Outer snapshot schema/count differs.')
    return frames, links, registry_hash


def main():
    p = parser(__doc__)
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    with verified_phase(root, 'audit') as phase:
        frames, links, registry_hash = collect(root)
        print('OUTER SNAPSHOT: 56 TARGET ROWS / 16 ASSIGNED / 14 NONEMPTY / 2 ALL-BLOCKED GROUPS')
        print('ALL 592 ADMISSIONS / 536 BLOCKED RETAINED; NO BLOCKED TARGETS IMPUTED')
        print('FEATURE RECONSTRUCTION / ALL-METHOD DECISION FREEZES / PAIRED TARGETS: PASS')
        if not args.save:
            print('PLAN ONLY. Use --save; no fitting, policy selection or future reevaluation.')
            return
        output = root/'c4_artifacts/meta_datasets/adult'/VERSION/'outer-all16-outcome-audit-v1'
        require(not output.exists(), 'Outer snapshot exists; preserve it.')
        output.mkdir(parents=True, exist_ok=False)
        files = {}
        for filename, frame in frames.items():
            indexed = frame.index.name == 'scenario_id'
            contents = frame.to_csv(index=indexed, lineterminator='\n').encode()
            (output/filename).write_bytes(contents)
            files[filename] = sha(contents)
            restored = pd.read_csv(io.BytesIO(contents),index_col=0 if indexed else None,float_precision='round_trip')
            pd.testing.assert_frame_equal(frame, restored, check_dtype=False, check_exact=True)
        note = '''# C4 outer snapshot evidence

All 16 predefined outer groups retained: 592 admissions, 56 admitted outcome rows,
536 blocked cases, 14 nonempty groups and 2 all-blocked groups.

All 17 method decisions were frozen for all 56 admitted cases before any outer
future evaluator ran. Targets were recomputed from saved paired predictions.
The N1 models, N2 calibration and selected release policies were not refitted.
No future targets entered controller inference. Blocked-case targets were not imputed.

Scope: Adult accuracy-margin pilot, simulated upstream producers, no production
release. The primary measured policy is review-only. Synthetic controller fixtures
remain technical tests and are not empirical deployment evidence.

This snapshot is input to final held-out comparisons and statistical reporting;
it is not itself the final comparison report or a formal risk guarantee.
'''
        (output/'C4_OUTER_SNAPSHOT_EVIDENCE.md').write_text(note,encoding='utf-8')
        files['C4_OUTER_SNAPSHOT_EVIDENCE.md'] = sha(note.encode())
        for path, digest in links.items():
            checked_bytes(path,digest)
        digest = save_json(output/'metadata.json', {'purpose':'verified_outer_snapshot',
            'created_at_utc':utc_now(), **git_context(), 'outer_execution_freeze_sha256':phase['sha256'],
            'outer_pre_release_registry_sha256':registry_hash, 'rows':56,'assigned_groups':16,
            'nonempty_groups':14,'all_blocked_groups':2,'admission_records':592,'blocked':536,
            'fold':'outer_test','feature_columns':list(frames['features.csv'].columns),
            'target_columns':list(TARGET_COLUMNS),'method_ids':list(METHOD_IDS),
            'files':files,'source_links':links,'N1_refitted':False,'N2_refitted':False,
            'policy_reselected':False,'blocked_targets_imputed':False,'production_release_authorized':False})
        for filename, expected in files.items():
            checked_bytes(output/filename,expected)
        print('SAVED OUTER SNAPSHOT:',output)
        print('METADATA SHA256:',digest)
        print('NEXT: final outer comparisons, paired statistics, tables and plots.')


if __name__ == '__main__':
    main()
