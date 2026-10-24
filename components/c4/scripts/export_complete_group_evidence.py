"""Export exact test/scenario evidence for the completed group; never retrain.

All outputs remain in external C4_DATA_ROOT. Optional pytest run is recorded
with its own timestamp; it does not reconstruct a historical test duration.
"""
import io
import json
import os
import platform
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _common import parser, resolve_data_root, checked_json, checked_bytes, require, sha, utc_now, git_context, REPO_ROOT, save_json
from _future_research import verify_freeze_dependencies
from _admitted_training import GROUP, group_context


def command(arguments):
    result = subprocess.run(arguments, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    return {'command': arguments, 'returncode': result.returncode, 'output': result.stdout}


def main():
    p = parser(__doc__)
    p.add_argument('--run-tests', action='store_true', help='Run and capture every current pytest item before exporting.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    if os.name == 'nt' and not str(root).startswith(chr(92)*2+'?'+chr(92)):
        root = Path(chr(92)*2+'?'+chr(92)+str(root))
    snapshot_dir = root / 'c4_artifacts/meta_datasets/adult/adult-simulation-pp1-v2' / (GROUP + '-v1')
    metadata, metadata_hash = checked_json(snapshot_dir)
    require(metadata['rows'] == 6 and metadata['group_count'] == 1, 'Expected the completed six-row group snapshot.')
    tables = {filename: checked_bytes(snapshot_dir / filename, checksum) for filename, checksum in metadata['files'].items()}
    index = pd.read_csv(io.BytesIO(tables['scenario_index.csv']))
    require(len(index) == 6 and index['scenario_id'].is_unique and set(index['group_id']) == {GROUP}, 'Snapshot scenario index differs.')
    benchmark = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    reports = []
    for record in index.to_dict('records'):
        scenario = record['scenario_id']; run = benchmark / scenario
        group_context(root, scenario)
        freeze, freeze_hash = checked_json(run / 'pre_release_freeze-v1', 'freeze')
        verify_freeze_dependencies(freeze)
        outcomes, outcome_hash = checked_json(run / 'future_research-v1', 'outcomes')
        pre_dir = root / 'c4_artifacts/pre_release/adult' / scenario
        pre_release, pre_hash = checked_json(pre_dir, 'evidence')
        require(freeze_hash == record['decision_freeze_sha256'] == outcomes['decision_freeze_sha256'] and
                outcome_hash == record['outcomes_sha256'] and pre_hash == record['pre_release_evidence_sha256'],
                'Scenario report hashes differ from snapshot index.')
        for filename, checksum in outcomes['files'].items(): checked_bytes(run / 'future_research-v1' / filename, checksum)
        envelope, envelope_hash = checked_json(run / 'pre_release_freeze-v1', 'upstream_simulation')
        require(envelope_hash == record['upstream_simulation_sha256'], 'Upstream envelope differs.')
        reports.append((scenario, pre_release, outcomes, envelope, run, pre_dir))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'c4_artifacts/evidence_exports/adult' / ('group-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {'created_at_utc': utc_now(), 'purpose': 'exact_six_scenario_and_test_evidence_export',
                  **git_context(), 'group_id': GROUP, 'snapshot_metadata_sha256': metadata_hash,
                  'python_executable': sys.executable, 'python_version': platform.python_version(),
                  'platform': platform.platform(), 'source_export_script_sha256': sha(Path(__file__).read_bytes())}
    checks = {label: command(['git', *arguments]) for label, arguments in {
        'branch': ['branch', '--show-current'], 'head': ['rev-parse', 'HEAD'],
        'remote_tracking_head': ['rev-parse', 'origin/c4-kesigan'],
        'last_commit': ['log', '-1', '--format=fuller'], 'working_tree': ['status', '--porcelain'],
        'diff_check': ['diff', '--check'], 'cached_diff_check': ['diff', '--cached', '--check']}.items()}
    provenance['git_checks'] = checks
    provenance['local_head_matches_remote_tracking_ref'] = (
        checks['head']['returncode'] == checks['remote_tracking_head']['returncode'] == 0 and
        checks['head']['output'].strip() == checks['remote_tracking_head']['output'].strip())
    provenance['remote_tracking_note'] = 'Local remote-tracking reference only; no fetch/push or server query performed. User previously reported push success.'
    installed = command([sys.executable, '-m', 'pip', 'freeze'])
    (output / 'pip_freeze.txt').write_text(installed['output'], encoding='utf-8')
    provenance['pip_freeze_returncode'] = installed['returncode']
    test_rows = []
    if args.run_tests:
        xml_path = output / 'pytest_results.xml'
        result = command([sys.executable, '-m', 'pytest', 'components/c4', '-v', '--color=no', '--junitxml=' + str(xml_path)])
        (output / 'pytest_full.log').write_text(result['output'], encoding='utf-8')
        provenance['pytest'] = {'returncode': result['returncode'], 'recorded_at_utc': utc_now(),
                               'command': result['command'], 'historical_15_10_seconds_recreated': False}
        if xml_path.exists():
            xml = ET.parse(xml_path).getroot()
            for case in xml.iter('testcase'):
                status = 'ERROR' if case.find('error') is not None else 'FAILED' if case.find('failure') is not None else 'SKIPPED' if case.find('skipped') is not None else 'PASSED'
                test_rows.append({'class_or_module': case.get('classname',''), 'test_name': case.get('name',''),
                                  'result': status, 'duration_seconds': case.get('time','')})
            provenance['pytest']['testcase_count'] = len(test_rows)
            provenance['pytest']['complete_expected_244_test_inventory'] = len(test_rows) == 244
            provenance['pytest']['counts'] = {status: sum(r['result'] == status for r in test_rows)
                                             for status in ('PASSED','FAILED','ERROR','SKIPPED')}
        print('PYTEST CAPTURE RETURN CODE:', result['returncode'], 'TESTCASE RECORDS:', len(test_rows))
    else:
        provenance['pytest'] = {'executed': False, 'note': 'No complete current test log exported; rerun with --run-tests.'}
    # Copy original report bytes, failure logs and derived evidence, excluding raw feature datasets/models.
    roots = [snapshot_dir, benchmark / 'group_admissions' / GROUP,
             root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2',
             root / 'c4_artifacts/protocols/offline_pilot/adult/adult-prospective-pilot-v1',
             root / 'c4_artifacts/splits/adult/v1', root / 'c4_artifacts/champion/adult/v1']
    for scenario, pre, future, envelope, run, pre_dir in reports: roots.extend([run, pre_dir])
    admission_report, _ = checked_json(benchmark / 'group_admissions' / GROUP / 'v1', 'group_admissions')
    for item in admission_report['records']:
        run = benchmark / item['scenario_id']; roots.append(run)
        admission, checksum = checked_json(run, 'admission')
        require(checksum == item['admission_sha256'], 'Admission table source differs.')
        roots.append(root / 'c4_artifacts/offline_repair/adult' / admission['artifact_version'])
        roots.append(root / 'c4_artifacts/repair_preparation/adult' / admission['preview_id'] / admission['preparation_version'])
    for version in ('development-tvae-exact-v1','development-tvae-target-strat-v1','pilot-independent-tvae-v1'):
        directory = root / 'c4_artifacts/offline_repair/adult' / version
        if directory.exists(): roots.append(directory)
    copied = {}
    allowed_csv = {'features.csv','targets.csv','scenario_index.csv','admission_table.csv',
                   'future_predictions.csv','future_membership.csv','release_membership.csv',
                   'diagnosed_membership.csv','corruption_mask.csv','privacy_distances.csv',
                   'sampling_selection.csv','training_loss.csv'}
    for directory in roots:
        if not directory.exists(): continue
        for path in directory.rglob('*'):
            if not path.is_file() or not (path.suffix in {'.json','.sha256','.md','.log'} or path.name in allowed_csv): continue
            relative = path.relative_to(root).as_posix()
            if relative in copied: continue
            contents = path.read_bytes()
            destination = output / 'reports' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            copied[relative] = sha(contents)
    config_paths = [REPO_ROOT / 'requirements.txt', REPO_ROOT / 'components/c4/requirements.txt',
                    *(REPO_ROOT / 'components/c4/configs' / name for name in
                      ('benchmark_protocol.json','offline_pilot_protocol.json','repair_validation_protocol.json','scenario_config.yaml'))]
    for path in config_paths:
        if not path.is_file(): continue
        relative = 'repository_config/' + path.relative_to(REPO_ROOT).as_posix()
        contents = path.read_bytes(); destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(contents)
        copied[relative] = sha(contents)
    save_json(output / 'export_metadata.json', {**provenance, 'copied_file_sha256': copied,
        'raw_datasets_or_models_copied': False, 'experiments_reexecuted': False,
        'historical_test_logs_not_reconstructed': True})
    lines = ['# C4 complete group evidence supplement', '',
        'This supplement exports actual current artifacts and one newly timestamped test run. It does not recreate historical test timings or fill future research results.', '',
        '## Checkpoint and Git provenance', '', '```json', json.dumps(provenance, indent=2), '```', '',
        '## Every captured test case', '', '| Module/class | Test (including parameters) | Result | Duration, seconds |',
        '|---|---|---|---|']
    for row in test_rows:
        lines.append('| ' + ' | '.join(str(row[k]).replace('|','\\|').replace('\n',' ') for k in
            ('class_or_module','test_name','result','duration_seconds')) + ' |')
    if not test_rows: lines.append('\nNo per-test inventory was captured. Do not claim all 244 test cases are documented.\n')
    lines += ['', '## Exact meta-dataset schema and values', '']
    for filename in ('features.csv','targets.csv','scenario_index.csv','admission_table.csv'):
        lines += ['### ' + filename, '', '```csv', tables[filename].decode('utf-8').rstrip('\n'), '```', '']
    for scenario, pre, future, envelope, run, pre_dir in reports:
        lines += ['## Scenario: ' + scenario, '', '### Pre-release and future metrics — all reported dimensions', '',
            '| Role/population | Rows | Class 0 | Class 1 | Champion accuracy | Candidate accuracy | Accuracy margin | Champion macro-F1 | Candidate macro-F1 | F1 margin | Champion AUC | Candidate AUC | AUC margin |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for role, comparisons in [('release_validation', pre['report']['comparisons']), ('sealed_future', future['comparisons'])]:
            for population, comparison in comparisons.items():
                before, after, margin = comparison['champion'], comparison['candidate'], comparison['candidate_minus_champion']
                values = [role+'/'+population, comparison['rows'], comparison['target_counts']['0'], comparison['target_counts']['1'],
                    before['accuracy'],after['accuracy'],margin['accuracy'],before['macro_f1'],after['macro_f1'],margin['macro_f1'],
                    before['auc'],after['auc'],margin['auc']]
                lines.append('| ' + ' | '.join('undefined' if v is None else str(v) for v in values) + ' |')
        lines += ['', '### Primary targets', '', '```json', json.dumps(future['primary_targets'], indent=2), '```', '',
                  '### Complete labelled simulated upstream evidence', '', '```json', json.dumps(envelope, indent=2), '```', '']
    lines += ['## Environment', '', '```text', installed['output'].rstrip(), '```', '',
              '## Completeness limits', '',
              '- Original JSON/report/log bytes and their hashes are in the accompanying reports directory/archive.',
              '- This export retains failures and undefined AUC values; it does not invent labels for blocked cases.',
              '- A newly executed pytest duration is separate from the previously reported 244 passed in 15.10 seconds.',
              '- Local HEAD equality with origin/c4-kesigan is not a fresh server-side push verification.',
              '- Missing historical full logs cannot be recovered from milestone counts alone.',
              '- N1–N4, calibrated policies, baseline/ablation and statistical validation remain pending.', '']
    md_path = output / 'C4_EVIDENCE_COMPLETENESS_SUPPLEMENT.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    archive = output / 'C4_COMPLETE_GROUP_EVIDENCE.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob('*'):
            if path.is_file() and path != archive: z.write(path, path.relative_to(output).as_posix())
    print('SUPPLEMENT:', md_path)
    print('SUPPLEMENT SHA256:', sha(md_path.read_bytes()))
    print('EVIDENCE ARCHIVE:', archive)
    print('EXPORTED SCENARIOS:', len(reports), 'RETAINED ADMISSION CASES:', len(admission_report['records']))
    print('EXPECTED 244-TEST INVENTORY COMPLETE:', len(test_rows) == 244)
    print('RETRAINING / FUTURE REEVALUATION: False; OUTPUTS OUTSIDE REPOSITORY: True')


if __name__ == '__main__': main()
