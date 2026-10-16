"""Outer admission scope and frozen development checkpoint verification."""
from pathlib import Path

from _common import C4_ROOT, checked_bytes, checked_json, require, sha
from _training_group import corrupt_labels

VERSION = 'adult-simulation-pp1-v2'
FOLD = 'outer_test'
N1_SPEC_HASH = '437ae92c313020697a35e8b1372442d327df22c9189aa6392afaeb2e354d98c3'
N1_RUN_HASH = 'c9bcedd6f74c47d856eead5a3dcfc5edb37605312318cebc695cf12fe0d05545'


def verify_n1(root):
    """Verify the new admission freeze, then return the unchanged original N1 binding."""
    frozen, _ = verify_outer_protocol(root)
    return frozen['binding']['n1_binding']


def outer_groups(manifest):
    groups = manifest['groups']
    mapping = {g['group_id']: g for g in groups}
    require(len(mapping) == len(groups), 'Duplicate group IDs in frozen manifest.')
    selected = [g['group_id'] for g in groups if g['split_role'] == FOLD]
    require(len(selected) == 16, 'Exactly 16 frozen outer groups are required.')
    return selected  # Preserve frozen manifest order; never sort by results.


def outer_group_cases(manifest, group_id):
    require(group_id in outer_groups(manifest), 'Only a frozen outer-test group is allowed.')
    rows = [r for r in manifest['scenarios'] if r['group_id'] == group_id]
    require(len(rows) == 37 and all(r['split_role'] == FOLD for r in rows),
            'Exactly 37 outer cases in the requested group are required.')
    require(len({r['scenario_id'] for r in rows}) == 37, 'Duplicate scenario IDs.')
    require(sum(bool(r['is_no_drift_control']) for r in rows) == 1, 'One no-drift control is required.')
    drifted = [r for r in rows if not r['is_no_drift_control']]
    expected = {(m, s, d, q) for m in ('marginal_multi','interaction','combined') for s in (.2,.5,.8)
                for d in ('exact_slice','broadened_slice') for q in ('targeted_repair','label_corrupted_repair')}
    actual = {(r['mechanism'],r['magnitude'],r['diagnosis_condition'],r['repair_condition']) for r in drifted}
    require(actual == expected, 'Outer group differs from the frozen factorial design.')
    control = next(r for r in rows if r['is_no_drift_control'])
    require(control['mechanism'] == 'none' and control['magnitude'] == 0 and
            control['diagnosis_condition'] == 'exact_slice' and control['repair_condition'] == 'targeted_repair',
            'No-drift control differs from frozen design.')
    return sorted(rows, key=lambda r:(r['shared_base_package_id'], r['repair_condition'] != 'targeted_repair', r['scenario_id']))


def batch_groups(manifest, number):
    require(type(number) is int and 1 <= number <= 4, 'Batch number must be 1 through 4.')
    return outer_groups(manifest)[(number-1)*4:number*4]


READINESS_HASH = '840f5723b7ef94853d3359982ba362d39692c5ab15b80ebd329b61989101ff59'
ENGINE_CHECK_HASH = 'bdf2a8a913e3df147bdc4d238cb9d655dbd949e8786c7e7b7c2e8b9a5e9cafeb'
COMPARISON_HASH = '3b51935a3f5dbe3a359edabe113c872d433699e11bfa590f4ee747da3ff917eb'
COMPARISON_SPEC_HASH = '5c16362b6abffac01e6409c5e90e68bd5267a332289730b564402beb0234ae55'
EVALUATION_HASH = 'e899365a05130a45a2f5c3ce513e99e1064605c5c98ecc985140fdc64eaab06e'
CONTROLLER_HASH = 'de09b0220f25b3c82f8849dd2becef4f3301806dfcb1e701d5108b0d252fb4a7'
PROTOCOL_VERSION = 'outer-admission-spec-v1'
SOURCE_FILES = (
    'scripts/_outer_group.py', 'scripts/freeze_outer_admission_protocol.py',
    'scripts/run_outer_case_admission.py', 'scripts/run_outer_group_admissions.py',
    'scripts/run_outer_admission_batch.py', 'scripts/audit_outer_admissions.py',
    'scripts/_policy_group.py', 'scripts/_training_group.py',
    'scripts/_scoped_benchmark.py', 'scripts/_common.py', 'scripts/_privacy_distance.py',
    'scripts/audit_development_repair.py', 'scripts/check_repair_distribution.py',
    'scripts/check_repair_privacy.py',
)


def protocol_directory(root):
    return Path(root) / 'c4_artifacts/outer_protocols/adult' / VERSION / PROTOCOL_VERSION


def runs_directory(root):
    return Path(root) / 'c4_artifacts/benchmark_runs/adult' / VERSION


def source_hashes():
    return {p: sha((C4_ROOT / p).read_bytes()) for p in SOURCE_FILES}


def load_manifest(root):
    import json
    directory = Path(root) / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    frozen, digest = checked_json(directory, 'freeze')
    manifest, manifest_hash = checked_json(directory, 'scenario_manifest')
    protocol = json.loads(checked_bytes(directory / 'protocol.json', frozen['protocol_sha256']))
    require(manifest_hash == frozen['scenario_manifest_sha256'] and
            frozen['full_execution_enabled'] is False and protocol['full_execution_enabled'] is False,
            'Original manifest/protocol scope differs.')
    groups = outer_groups(manifest)
    all_rows = manifest['scenarios']
    require(len({r['scenario_id'] for r in all_rows}) == len(all_rows), 'Scenario IDs must be globally unique.')
    mapping = {g['group_id']: g['split_role'] for g in manifest['groups']}
    require(all(r['group_id'] in mapping and r['split_role'] == mapping[r['group_id']] for r in all_rows),
            'Each scenario/group must belong to exactly one fold.')
    for group in groups:
        outer_group_cases(manifest, group)
    return manifest, digest, manifest_hash


def current_bindings(root):
    """Verify saved development checkpoints without calling development-only inventory loaders.

    Those loaders intentionally reject outer artifacts and cannot be reused once
    outer admissions start. Only saved development metadata/source bindings are
    checked here; no outer observations, predictions or outcomes are read.
    """
    from _policy_group import verify_n1 as verify_original_n1
    from _controller_artifacts import bindings, controller_directory
    root = Path(root)
    n1 = verify_original_n1(root)
    controller_binding, _ = bindings(root)
    controller, controller_hash = checked_json(controller_directory(root), 'freeze')
    require(controller_hash == CONTROLLER_HASH and controller['binding'] == controller_binding,
            'Frozen controller differs.')
    for p, digest in controller['source_sha256'].items():
        checked_bytes(C4_ROOT / p, digest)
    ready_dir = root / 'c4_artifacts/evaluation_readiness/adult' / VERSION / 'comparison-evaluation-ready-v1'
    ready, ready_hash = checked_json(ready_dir, 'freeze')
    require(ready_hash == READINESS_HASH and ready['comparisons_ready'] is True and
            ready['evaluation_engine_ready'] is True and ready['outer_pipeline_ready'] is False and
            ready['outer_execution_authorized'] is False and ready['outer_outcomes_read'] is False,
            'Comparison/evaluation readiness differs.')
    b = ready['binding']
    require(b['controller_freeze_sha256'] == controller_hash and
            b['prepared_comparisons_metadata_sha256'] == COMPARISON_HASH and
            b['preparation_specification_sha256'] == COMPARISON_SPEC_HASH and
            b['evaluation_specification_sha256'] == EVALUATION_HASH and
            ready['development_evaluation_metadata_sha256'] == ENGINE_CHECK_HASH,
            'Readiness checkpoint links differ.')
    for p, digest in b['engine_source_sha256'].items():
        checked_bytes(C4_ROOT / p, digest)
    engine_dir = root / 'c4_artifacts/evaluations/adult' / VERSION / 'development-engine-check-v1'
    engine, engine_hash = checked_json(engine_dir)
    require(engine_hash == ENGINE_CHECK_HASH and engine['binding'] == b and engine['rows'] == 44 and
            engine['bootstrap_replicates'] == 2000 and engine['bootstrap_seed'] == 20261016 and
            engine['outer_test_evaluated'] is False, 'Development evaluation check differs.')
    for filename, digest in engine['files'].items():
        checked_bytes(engine_dir / filename, digest)
    comparison_dir = root / 'c4_artifacts/comparison_calibrations/adult' / VERSION / 'comparison-calibration-v1'
    comparison, comparison_hash = checked_json(comparison_dir / 'prepared-v1')
    spec, spec_hash = checked_json(comparison_dir, 'specification')
    require(comparison_hash == COMPARISON_HASH and spec_hash == COMPARISON_SPEC_HASH and
            comparison['specification_sha256'] == spec_hash and
            comparison['evaluation_specification_sha256'] == EVALUATION_HASH,
            'Prepared comparisons differ.')
    for p, digest in spec['binding']['source_sha256'].items():
        checked_bytes(C4_ROOT / p, digest)
    for p, digest in spec['binding']['source_links'].items():
        checked_bytes(p, digest)
    for filename, digest in comparison['files'].items():
        checked_bytes(comparison_dir / 'prepared-v1' / filename, digest)
    from _evaluation_protocol import PROTOCOL_VERSION as evaluation_version, validate_definition, verify_manifest
    evaluation_dir = root / 'c4_artifacts/evaluation_protocols/adult' / VERSION / evaluation_version
    evaluation, evaluation_hash = checked_json(evaluation_dir, 'freeze')
    definition, definition_hash = checked_json(evaluation_dir, 'definition')
    require(evaluation_hash == EVALUATION_HASH and definition_hash == evaluation['definition_sha256'],
            'Evaluation definition differs.')
    validate_definition(definition)
    for p, digest in evaluation['source_sha256'].items():
        checked_bytes(C4_ROOT / p, digest)
    manifest, original_hash, manifest_hash = load_manifest(root)
    verify_manifest(manifest)
    require(original_hash == n1['protocol_freeze_sha256'] == evaluation['original_protocol_freeze_sha256'],
            'Original experiment binding differs.')
    return {'n1_binding': n1, 'controller_freeze_sha256': controller_hash,
            'comparison_readiness_sha256': ready_hash, 'development_evaluation_metadata_sha256': engine_hash,
            'prepared_comparisons_metadata_sha256': comparison_hash,
            'evaluation_specification_sha256': evaluation_hash,
            'protocol_freeze_sha256': original_hash, 'scenario_manifest_sha256': manifest_hash,
            'outer_group_ids_in_fixed_order': outer_groups(manifest)}


def assert_no_outer_artifacts(root, manifest):
    """Initial freeze must precede any outer admission/preparation artifacts."""
    for row in manifest['scenarios']:
        if row['split_role'] != FOLD:
            continue
        identifier = 'bmk-' + sha(row['scenario_id'].encode())[:16]
        for path in (runs_directory(root) / row['scenario_id'],
                     runs_directory(root) / 'group_admissions' / row['group_id'],
                     Path(root) / 'c4_artifacts/scenario_previews/adult' / identifier,
                     Path(root) / 'c4_artifacts/repair_preparation/adult' / identifier,
                     Path(root) / 'c4_artifacts/offline_repair/adult' / (identifier + '-repair-v1'),
                     Path(root) / 'c4_artifacts/pre_release/adult' / row['scenario_id']):
            require(not path.exists(), 'Outer artifact predates admission freeze: ' + str(path))


def assert_admission_only(root, manifest):
    """Admission authorization ends at repair screening, even for direct case CLI calls."""
    for row in manifest['scenarios']:
        if row['split_role'] != FOLD:
            continue
        run = runs_directory(root) / row['scenario_id']
        for stage in ('scoped_candidate_validation-v1', 'pre_release_freeze-v1', 'future_research-v1'):
            require(not (run / stage).exists(), 'Admission-only scope violated: ' + str(run / stage))
        require(not (Path(root) / 'c4_artifacts/pre_release/adult' / row['scenario_id']).exists(),
                'Outer pre-release evidence exists; use a separately frozen downstream harness.')


def verify_outer_protocol(root):
    frozen, digest = checked_json(protocol_directory(root), 'freeze')
    require(frozen['purpose'] == 'frozen_outer_admission_only_protocol' and
            frozen['admission_execution_authorized'] is True and
            frozen['outer_candidate_execution_authorized'] is False and
            frozen['outer_future_execution_authorized'] is False and
            frozen['production_release_authorized'] is False, 'Wrong outer authorization scope.')
    require(frozen['source_sha256'] == source_hashes(), 'Outer admission implementation changed after freeze.')
    require(frozen['binding'] == current_bindings(root), 'Frozen development artifacts changed.')
    manifest, _, _ = load_manifest(root)
    assert_admission_only(root, manifest)
    return frozen, digest


def outer_freeze_hash(root):
    return checked_json(protocol_directory(root), 'freeze')[1]


def import_preflight():
    """Fail before any new claim if the TVAE runtime cannot load its DLLs."""
    import torch
    from ctgan import TVAE
    require(callable(TVAE) and bool(torch.__version__), 'Torch/TVAE import preflight failed.')
    print('TORCH / TVAE IMPORT PREFLIGHT: PASS', flush=True)
