"""Record-local checks plus explicit full development verification at group boundaries.

This additive implementation changes verification scheduling, not admission logic.
No stat-only cache or unverified persistent cache is used. Every local artifact
checksum and the recomputed admission decision is still checked for each record.
"""
from pathlib import Path
from _common import C4_ROOT, checked_bytes, checked_json, require, save_json, sha, git_context, utc_now
from _outer_group import (VERSION, verify_outer_protocol, outer_freeze_hash,
                          load_manifest, source_hashes as original_sources)

ORIGINAL_OUTER_FREEZE = 'd8815e1cc433daa45963ede0f9075e7d2ec4038dd2b9f6785cf38cd7ffa859b9'
REVISION = 'outer-admission-verification-v2'
SOURCE_FILES = ('scripts/_outer_admission_verification.py',
                'scripts/run_outer_group_admissions_portable.py',
                'scripts/run_outer_admission_batch_portable.py',
                'scripts/audit_outer_admissions_portable.py')


def source_hashes():
    return {p: sha((C4_ROOT / p).read_bytes()) for p in SOURCE_FILES}


def revision_directory(root):
    return Path(root) / 'c4_artifacts/outer_protocols/adult' / VERSION / REVISION


def full_boundary(root):
    """Read and hash the complete shared checkpoints at an explicit boundary."""
    print('VERIFYING SHARED DEVELOPMENT CHECKPOINTS (group boundary)...', flush=True)
    original, original_hash = verify_outer_protocol(root)
    require(original_hash == ORIGINAL_OUTER_FREEZE, 'Original outer admission freeze differs.')
    print('SHARED DEVELOPMENT CHECKPOINTS: PASS', flush=True)
    return original, original_hash


def verify_revision(root, *, full=False):
    if full:
        original, original_hash = full_boundary(root)
    else:
        from _outer_group import protocol_directory, assert_admission_only
        original, original_hash = checked_json(protocol_directory(root), 'freeze')
        require(original_hash == ORIGINAL_OUTER_FREEZE, 'Original admission freeze differs.')
        require(original['source_sha256'] == original_sources(), 'Original source changed after freeze.')
        manifest, _, _ = load_manifest(root)
        assert_admission_only(root, manifest)
    saved, digest = checked_json(revision_directory(root), 'freeze')
    require(saved['purpose'] == 'frozen_outer_admission_verification_scheduling' and
            saved['original_outer_admission_freeze_sha256'] == original_hash and
            saved['original_binding'] == original['binding'] and
            saved['source_sha256'] == source_hashes() and
            saved['shared_verification_schedule'] == 'full_before_and_after_each_group' and
            saved['record_checksums_and_screen_recomputation_retained'] is True and
            saved['outer_future_execution_authorized'] is False,
            'Verification scheduling revision differs.')
    return original, digest


def freeze_revision(root):
    original, original_hash = full_boundary(root)
    git = git_context()
    require(not git['working_tree_dirty'], 'Commit portable verification code before --freeze-spec.')
    output = revision_directory(root)
    require(not output.exists(), 'Verification revision exists; preserve it. Use normal plan/execute next.')
    digest = save_json(output / 'freeze.json', {
        'purpose': 'frozen_outer_admission_verification_scheduling', 'created_at_utc': utc_now(), **git,
        'original_outer_admission_freeze_sha256': original_hash, 'original_binding': original['binding'],
        'source_sha256': source_hashes(),
        'shared_verification_schedule': 'full_before_and_after_each_group',
        'record_checksums_and_screen_recomputation_retained': True,
        'persisted_verification_cache_used': False, 'stat_only_cache_used': False,
        'admission_rule_changed': False, 'scenario_or_seed_changed': False,
        'old_attempts_modified': False, 'outer_future_execution_authorized': False,
        'production_release_authorized': False,
    })
    print('VERIFICATION SCHEDULING FREEZE:', output)
    print('FREEZE SHA256:', digest)
    return digest


def existing_record(root, row, freeze_hash, n1_binding):
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / row['scenario_id']
    if not run.exists(): return None
    require((run / 'admission.json').is_file(),
            'Incomplete scenario claim found; preserve it and resolve the recorded failure before continuing: ' + str(run))
    admission, admission_hash = checked_json(run, 'admission')
    require(admission['scenario'] == row and admission['freeze_sha256'] == freeze_hash, 'Existing scenario/protocol differs.')
    require(admission.get('n1_binding') == n1_binding and admission.get('outer_admission_freeze_sha256') == outer_freeze_hash(root), 'Existing outer case links to different N1 evidence.')
    package = root / 'c4_artifacts/offline_repair/adult' / admission['artifact_version']
    generation, generation_hash = checked_json(package)
    privacy, privacy_hash = checked_json(package, 'privacy_diagnostics')
    require(generation_hash == admission['generation_metadata_sha256'] and
            privacy_hash == admission['privacy_report_sha256'] and
            privacy['generation_metadata_sha256'] == generation_hash, 'Existing admission artifact links differ.')
    import json
    from _scoped_benchmark import frozen_tail_screen
    protocol_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    protocol_freeze, _ = checked_json(protocol_dir, 'freeze')
    protocol = json.loads(checked_bytes(protocol_dir / 'protocol.json', protocol_freeze['protocol_sha256']))
    screen = frozen_tail_screen(protocol, privacy['summaries']['synthetic_to_generator_fit'],
                                privacy['summaries']['diagnosed_internal_validation_to_generator_fit'],
                                protocol_kind='benchmark')
    require(all(admission.get(key) == value for key, value in screen.items()),
            'Saved admission screen differs from recomputation.')
    accepted = screen['screen_status'] == 'PASS' and admission['simulated_c2_diagnosis']['diagnosis_stability'] >= .50
    expected_status = 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if accepted else 'BLOCKED_ESCALATE'
    require(admission['simulation_admission'] == expected_status and
            admission['candidate_training_executed'] is False and admission['future_evaluation_executed'] is False,
            'Saved admission status/execution flags differ.')
    checked_bytes(package / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])
    checked_bytes(package / 'generator.pkl', generation['generator_sha256'])
    if generation.get('corruption_variant'):
        checked_bytes(package / 'corruption_mask.csv', generation['corruption_variant']['mask_sha256'])
    return {'scenario_id': row['scenario_id'], 'group_id': row['group_id'], 'split_role': row['split_role'],
            'mechanism': row['mechanism'], 'magnitude': row['magnitude'], 'diagnosis_condition': row['diagnosis_condition'],
            'repair_condition': row['repair_condition'], 'is_no_drift_control': row['is_no_drift_control'],
            'screen_status': admission['screen_status'], 'tail_distance_ratio': admission['tail_distance_ratio'],
            'simulation_admission': admission['simulation_admission'], 'artifact_version': admission['artifact_version'],
            'admission_sha256': admission_hash}



def verify_report(root, group, rows, freeze_hash, n1_binding):
    directory = root / 'c4_artifacts/benchmark_runs/adult' / VERSION / 'group_admissions' / group / 'v1'
    report, digest = checked_json(directory, 'group_admissions')
    require(report['outer_admission_freeze_sha256'] == ORIGINAL_OUTER_FREEZE and
            report['group_id'] == group and report['freeze_sha256'] == freeze_hash and
            report['n1_binding'] == n1_binding, 'Outer group report binding differs.')
    actual = []
    for number, row in enumerate(rows, 1):
        print(f"VERIFYING RECORD {number}/{len(rows)}: {row['scenario_id']}", flush=True)
        record = existing_record(root, row, freeze_hash, n1_binding)
        require(record is not None, 'Missing completed admission record.')
        actual.append(record)
    require(report['records'] == actual, 'Group report differs from actual records.')
    admitted = sum(r['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' for r in actual)
    require(report['cases'] == 37 and report['scoped_admissible_cases'] == admitted and
            report['blocked_cases'] == 37-admitted, 'Outer group counts differ.')
    for filename, expected in report.get('case_log_sha256', {}).items():
        checked_bytes(directory / filename, expected)
    return {'group_id': group, 'admitted': admitted, 'blocked': 37-admitted,
            'admission_sha256': digest, 'status': 'ADMISSION_COMPLETE',
            'candidate_pipeline_executed_by_batch': False, 'future_evaluation_executed_by_batch': False}
