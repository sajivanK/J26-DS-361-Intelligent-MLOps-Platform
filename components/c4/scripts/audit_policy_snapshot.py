"""Audit all saved policy results and freeze combined tables without classifier reevaluation."""
import io
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _policy_pipeline import VERSION, POLICY_SPEC_HASH, policy_inventory, verify_context_support, verify_freeze_dependencies
from _scenario_meta import TARGET_COLUMNS
from c4.conformal.benchmark_n2 import JointCalibration, apply_joint_calibration
from c4.risk.benchmark_policy import candidate_policies, evidence_context, policy_decision
from c4.risk.benchmark_policy_selection import validate_index

BATCH_HASH = '6e92dec3bbedb1850a511ac8e2d808efb298aa849c4dbf06af8473c05526dcdc'


def combine_tables(features, targets, indices):
    require(features and len(features) == len(targets) == len(indices), 'Missing policy tables.')
    columns = features[0].columns.tolist()
    require(len(columns) == 75 and not set(columns) & set(TARGET_COLUMNS), 'Invalid feature schema.')
    for f, t, i in zip(features, targets, indices):
        require(f.columns.tolist() == columns and t.columns.tolist() == list(TARGET_COLUMNS), 'Schema differs.')
        require(f.index.equals(t.index) and f.index.tolist() == i.scenario_id.tolist(), 'Scenario order differs.')
        require(np.isfinite(f.to_numpy(dtype=float)).all() and np.isfinite(t.to_numpy(dtype=float)).all(), 'Nonfinite values.')
    f, t, i = pd.concat(features), pd.concat(targets), pd.concat(indices, ignore_index=True)
    require(f.index.is_unique and t.index.is_unique, 'Duplicate scenario IDs.')
    validate_index(i.to_dict('records'))
    return f, t, i


def main():
    p = parser(__doc__)
    p.add_argument('--save', action='store_true')
    p.add_argument('--snapshot-version', type=name, default='policy-all12-audit-v1')
    args = p.parse_args(); root = resolve_data_root(args.data_root)
    manifest, groups, reports, selected = policy_inventory(root)
    support, support_hash = verify_context_support(root)
    runs = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    meta_root = root / 'c4_artifacts/meta_datasets/adult' / VERSION
    links = {}
    def document(directory, stem='metadata'):
        value, digest = checked_json(directory, stem)
        links[str(directory / (stem + '.json'))] = digest
        return value, digest
    def table(directory, filename, md):
        data = checked_bytes(directory / filename, md['files'][filename])
        links[str(directory / filename)] = sha(data)
        return pd.read_csv(io.BytesIO(data), float_precision='round_trip')
    batchdir = runs / 'policy_candidate_batches/policy-candidate-all12-v2'
    batch, digest = document(batchdir, 'batch_results')
    require(digest == BATCH_HASH and batch['group_ids_in_fixed_order'] == groups and batch['group_count'] == 12
            and batch['meta_rows'] == 44 and batch['blocked_cases'] == 400
            and batch['policy_specification_sha256'] == POLICY_SPEC_HASH
            and batch['policy_context_support_sha256'] == support_hash, 'Completed policy checkpoint differs.')
    claim, _ = document(batchdir, 'claim')
    for path, digest in claim['source_sha256'].items():
        checked_bytes(path, digest); links[path] = digest
    for filename, digest in batch['case_log_sha256'].items():
        checked_bytes(batchdir / filename, digest); links[str(batchdir / filename)] = digest
    records = {r['group_id']: r for r in batch['records']}
    require(len(records) == len(batch['records']) == 12, 'Batch groups missing or duplicated.')
    all_manifest_groups = [g['group_id'] for g in manifest['groups']]
    require(len(all_manifest_groups) == len(set(all_manifest_groups)), 'Manifest group IDs duplicated.')
    all_scenarios = manifest['scenarios']
    require(len({r['scenario_id'] for r in all_scenarios}) == len(all_scenarios), 'Manifest scenario IDs duplicated.')
    roles = {g['group_id']: g['split_role'] for g in manifest['groups']}
    require(all(roles.get(r['group_id']) == r['split_role'] for r in all_scenarios), 'Group belongs to conflicting folds.')
    features, targets, indices, admissions, summary = [], [], [], [], []
    by_group = {g: [r for r in selected if r['group_id'] == g] for g in groups}
    for gid in groups:
        report = reports[gid]['report']; record = records[gid]; rows = by_group[gid]
        require(record['admission_sha256'] == reports[gid]['sha256'] and record['admitted'] == len(rows)
                and record['blocked'] == 37 - len(rows), 'Group admission accounting differs.')
        document(runs / 'group_admissions' / gid / 'v1', 'group_admissions')
        for raw in report['records']:
            value, digest = document(runs / raw['scenario_id'], 'admission')
            require(digest == raw['admission_sha256'] and value['scenario']['split_role'] == 'policy_calibration'
                    and value['scenario']['group_id'] == gid, 'Individual admission differs.')
            admissions.append({**raw, 'group_id': gid, 'split_role': 'policy_calibration',
                'verified_target_row_available': raw['scenario_id'] in {r['scenario_id'] for r in rows}})
        summary.append({'group_id': gid, 'admitted': len(rows), 'blocked': 37-len(rows),
                        'status': 'COMPLETE' if rows else 'ALL_BLOCKED'})
        if not rows:
            require(record['status'] == 'ADMISSION_ONLY_ALL_BLOCKED' and record['meta_rows'] == 0, 'Empty group differs.')
            print('VERIFIED GROUP:', gid, 'META ROWS: 0 BLOCKED: 37; NO TARGETS IMPUTED'); continue
        # Existing verifier reconstructs all 75 features and recomputes paired targets from saved predictions.
        result = subprocess.run([sys.executable, str(C4_ROOT / 'scripts/build_policy_group_meta_dataset.py'),
            '--data-root', str(root), '--group-id', gid], capture_output=True, text=True)
        require(result.returncode == 0, 'Read-only group verification failed:\n' + result.stdout + result.stderr)
        directory = meta_root / (gid + '-policy-v1')
        md, md_hash = document(directory)
        pipeline, pipeline_hash = document(runs / 'group_admissions' / gid / 'policy-pipeline-v1', 'group_pipeline')
        require(md_hash == record['meta_metadata_sha256'] and pipeline_hash == record['pipeline_sha256']
                and md['group_pipeline_sha256'] == pipeline_hash and md['group_admissions_sha256'] == reports[gid]['sha256']
                and md['policy_context_support_sha256'] == support_hash and md['fold'] == 'policy_calibration', 'Group snapshot links differ.')
        for filename, digest in md['files'].items():
            checked_bytes(directory / filename, digest); links[str(directory / filename)] = digest
        f = table(directory, 'features.csv', md).set_index('scenario_id')
        t = table(directory, 'targets.csv', md).set_index('scenario_id')
        i = table(directory, 'scenario_index.csv', md)
        require(set(f.index) == {r['scenario_id'] for r in rows} and len(f) == len(rows)
                and i.group_id.eq(gid).all(), 'Target rows must be admitted scenarios only.')
        features.append(f); targets.append(t); indices.append(i)
        print('VERIFIED GROUP:', gid, 'META ROWS:', len(f), 'BLOCKED:', 37-len(f))
    f, t, index = combine_tables(features, targets, indices)
    require(f.shape == (44, 75) and len(admissions) == 444 and index.group_id.nunique() == 11, 'Combined inventory differs.')
    n1base = root / 'c4_artifacts/forecasting/adult' / VERSION / 'adult-n1-training-v1'
    spec, _ = document(n1base, 'specification')
    train_dir = meta_root / spec['binding']['configuration']['training_snapshot_version']
    train_md, train_hash = document(train_dir)
    require(train_hash == spec['binding']['training_metadata_sha256'] and f.columns.tolist() == spec['binding']['raw_columns'], 'N1 training binding differs.')
    caldir = meta_root / 'conformal-plus-extension-audit-v2'
    cal_md, cal_hash = document(caldir)
    require(cal_hash == '963e374608ac0b4090a2fe24b83d8f271055208a0a343878b2dcb107e4dc7613', 'Pooled calibration checkpoint differs.')
    for directory, md in ((train_dir, train_md), (caldir, cal_md)):
        other = table(directory, 'scenario_index.csv', md)
        require(not set(index.scenario_id) & set(other.scenario_id) and not set(groups) & set(other.group_id), 'Policy/training/conformal overlap.')
    n2dir = root / 'c4_artifacts/conformal/adult' / VERSION / 'n2-joint-v2'
    n2md, n2hash = document(n2dir)
    calibration, chash = document(n2dir, 'calibration')
    require(chash == n2md['files']['calibration.json'], 'N2 calibration checksum differs.')
    policies = candidate_policies(); signal_rows = []; forecast_rows = {m: [] for m in ('lightgbm', 'linear')}
    for sid in f.index:
        run = runs / sid / 'pre_release_freeze-v1'
        snapshot, snap_hash = document(run, 'freeze'); verify_freeze_dependencies(snapshot)
        links.update(snapshot['dependencies'])
        fixed, fixed_hash = document(run, 'n1_forecasts')
        signals, signal_hash = document(run, 'policy_forecasts')
        ix = index.loc[index.scenario_id.eq(sid)].iloc[0]
        require(snap_hash == ix.decision_freeze_sha256 and fixed_hash == snapshot['n1_forecasts_sha256']
                and signal_hash == snapshot['policy_forecasts_sha256'] and signals['n2_metadata_sha256'] == n2hash
                and fixed['future_targets_included'] is False and signals['future_targets_included'] is False
                and signals['policy_specification_sha256'] == POLICY_SPEC_HASH, 'Pre-future forecast links differ.')
        require(fixed['feature_columns'] == f.columns.tolist() and
                np.allclose(f.loc[sid].to_numpy(dtype=float), [fixed['decision_time_features'][c] for c in f.columns], rtol=0, atol=1e-15), 'Frozen N1 feature reconstruction differs.')
        evidence = fixed['decision_time_features']
        require(evidence_context(evidence) == signals['evidence_context'], 'Frozen evidence context differs.')
        outcomes, ohash = document(runs / sid / 'future_research-v1', 'outcomes')
        require(ohash == ix.outcomes_sha256 and outcomes['policy_context_support_sha256'] == support_hash, 'Outcome provenance differs.')
        for filename, expected in outcomes['files'].items():
            checked_bytes(runs / sid / 'future_research-v1' / filename, expected)
            links[str(runs / sid / 'future_research-v1' / filename)] = expected
        for method in forecast_rows:
            raw = fixed['forecasts'][method]
            forecast_rows[method].append({'scenario_id': sid, **raw})
            fit = JointCalibration.from_dict(calibration['methods'][method])
            adjusted = apply_joint_calibration(fit, pd.DataFrame([raw])).iloc[0]
            bounds = {target: [float(adjusted[target+'__lower']), float(adjusted[target+'__upper'])] for target in TARGET_COLUMNS}
            require(bounds == signals['intervals'][method], 'Frozen N2 interval reconstruction differs.')
            require(set(signals['candidate_signals'][method]) == {p['policy_id'] for ps in policies.values() for p in ps}, 'Frozen policy inventory differs.')
            for family, candidates in policies.items():
                for policy in candidates:
                    kwargs = dict(n2_supported=fit.auto_deploy_supported, policy=policy)
                    unscreened = policy_decision(bounds, evidence, signals['explicit_hard_gates'], **kwargs)
                    require(unscreened == signals['candidate_signals'][method][policy['policy_id']], 'Frozen pre-outcome policy signal differs.')
                    screened = policy_decision(bounds, evidence, signals['explicit_hard_gates'],
                        unsupported_contexts=support['unsupported_contexts'], **kwargs)
                    signal_rows.append({'scenario_id': sid, 'group_id': ix.group_id, 'split_role': 'policy_calibration',
                        'method': method, 'family': family, 'policy_id': policy['policy_id'],
                        'context': signals['evidence_context']['context'], 'raw_decision': unscreened['decision'],
                        'decision': screened['decision'], 'reason': screened['reason']})
    print('POLICY SNAPSHOT: 44 ROWS; 12 ASSIGNED / 11 NONEMPTY GROUPS; 444 ADMISSIONS; 400 BLOCKED')
    print('FEATURE / PAIRED TARGET / FORECAST / INTERVAL / PRE-FUTURE SIGNAL / FOLD CHECKS: PASS')
    print('GLOBAL/N4 SELECTION: PENDING; NO TRAINING OR OUTER-TEST ACCESS')
    if not args.save:
        print('PLAN ONLY. Use --save to persist the audited policy snapshot.'); return
    output = meta_root / args.snapshot_version
    require(not output.exists(), 'Policy audit snapshot exists; preserve it.')
    verify_context_support(root)
    for path, digest in links.items(): checked_bytes(path, digest)
    output.mkdir(parents=True, exist_ok=False); files = {}
    tables = {'features.csv': (f, True), 'targets.csv': (t, True), 'scenario_index.csv': (index, False),
        'admission_table.csv': (pd.DataFrame(admissions), False), 'group_summary.csv': (pd.DataFrame(summary), False),
        'candidate_signals.csv': (pd.DataFrame(signal_rows), False),
        **{m+'_frozen_forecasts.csv': (pd.DataFrame(rows).set_index('scenario_id'), True) for m, rows in forecast_rows.items()}}
    for filename, (frame, include_index) in tables.items():
        data = frame.to_csv(index=include_index, lineterminator='\n').encode(); (output / filename).write_bytes(data); files[filename] = sha(data)
    digest = save_json(output / 'metadata.json', {'purpose': 'verified_combined_policy_snapshot',
        'created_at_utc': utc_now(), **git_context(), 'rows': 44, 'groups': groups, 'nonempty_groups': 11,
        'all_blocked_groups': 1, 'admission_cases': 444, 'blocked_cases': 400, 'fold': 'policy_calibration',
        'feature_columns': f.columns.tolist(), 'target_columns': list(TARGET_COLUMNS), 'files': files,
        'policy_specification_sha256': POLICY_SPEC_HASH, 'policy_context_support_sha256': support_hash,
        'batch_results_sha256': BATCH_HASH, 'source_links': links,
        'source_code_sha256': sha(Path(__file__).read_bytes()), 'N1_refitted': False, 'N2_refitted': False,
        'policy_selected': False, 'release_authorized': False})
    for filename, expected in files.items(): checked_bytes(output / filename, expected)
    print('SAVED POLICY SNAPSHOT:', output); print('METADATA SHA256:', digest)


if __name__ == '__main__': main()
