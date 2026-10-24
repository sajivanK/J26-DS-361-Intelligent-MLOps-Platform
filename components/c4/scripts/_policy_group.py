"""Policy-fold selection and frozen-N1 verification; no model fitting."""
from pathlib import Path

from _common import C4_ROOT, checked_bytes, checked_json, require
from _training_group import corrupt_labels

VERSION = 'adult-simulation-pp1-v2'
FOLD = 'policy_calibration'
N1_SPEC_HASH = '437ae92c313020697a35e8b1372442d327df22c9189aa6392afaeb2e354d98c3'
N1_RUN_HASH = 'c9bcedd6f74c47d856eead5a3dcfc5edb37605312318cebc695cf12fe0d05545'


def verify_n1(root):
    directory = Path(root) / 'c4_artifacts/forecasting/adult' / VERSION / 'adult-n1-training-v1'
    spec, spec_hash = checked_json(directory, 'specification')
    run, run_hash = checked_json(directory / 'n1-fit-v1')
    require(spec_hash == N1_SPEC_HASH and run_hash == N1_RUN_HASH,
            'N1 specification or fitted-model version differs from the verified checkpoint.')
    require(run['specification_sha256'] == spec_hash and run['n2_calibrated'] is False and
            run['outer_test_evaluated'] is False and run['release_authorized'] is False,
            'N1 stage/provenance differs.')
    require(run['internal_grouped_validation_complete'] is True and
            run['model_reload_prediction_equality'] is True, 'N1 completion evidence differs.')
    binding = spec['binding']
    require(run['training_metadata_sha256'] == binding['training_metadata_sha256'], 'N1 training snapshot link differs.')
    for relative, expected in binding['code_sha256'].items():
        checked_bytes(C4_ROOT / relative, expected)
    for filename, expected in run['files'].items():
        checked_bytes(directory / 'n1-fit-v1' / filename, expected)
    verify_n2_n3(root)
    return {'specification_sha256': spec_hash, 'run_metadata_sha256': run_hash,
            'protocol_freeze_sha256': binding['protocol_freeze_sha256']}


def policy_groups(manifest):
    groups = manifest['groups']
    mapping = {g['group_id']: g for g in groups}
    require(len(mapping) == len(groups), 'Duplicate group IDs in frozen manifest.')
    selected = [g['group_id'] for g in groups if g['split_role'] == FOLD]
    require(len(selected) == 12, 'Exactly 12 frozen policy groups are required.')
    return selected  # Preserve frozen manifest order; never sort by results.


def policy_group_cases(manifest, group_id):
    require(group_id in policy_groups(manifest), 'Only a frozen policy-calibration group is allowed.')
    rows = [r for r in manifest['scenarios'] if r['group_id'] == group_id]
    require(len(rows) == 37 and all(r['split_role'] == FOLD for r in rows),
            'Exactly 37 policy cases in the requested group are required.')
    require(len({r['scenario_id'] for r in rows}) == 37, 'Duplicate scenario IDs.')
    require(sum(bool(r['is_no_drift_control']) for r in rows) == 1, 'One no-drift control is required.')
    drifted = [r for r in rows if not r['is_no_drift_control']]
    expected = {(m, s, d, q) for m in ('marginal_multi','interaction','combined') for s in (.2,.5,.8)
                for d in ('exact_slice','broadened_slice') for q in ('targeted_repair','label_corrupted_repair')}
    actual = {(r['mechanism'],r['magnitude'],r['diagnosis_condition'],r['repair_condition']) for r in drifted}
    require(actual == expected, 'Policy group differs from the frozen factorial design.')
    control = next(r for r in rows if r['is_no_drift_control'])
    require(control['mechanism'] == 'none' and control['magnitude'] == 0 and
            control['diagnosis_condition'] == 'exact_slice' and control['repair_condition'] == 'targeted_repair',
            'No-drift control differs from frozen design.')
    return sorted(rows, key=lambda r:(r['shared_base_package_id'], r['repair_condition'] != 'targeted_repair', r['scenario_id']))


def batch_groups(manifest, number):
    require(type(number) is int and 1 <= number <= 3, 'Batch number must be 1 through 3.')
    return policy_groups(manifest)[(number-1)*4:number*4]


def verify_n2_n3(root):
    """Pin completed safety stages; no score tables or observed outcomes are read."""
    root = Path(root)
    n2_dir = root / 'c4_artifacts/conformal/adult' / VERSION / 'n2-joint-v1'
    n2, n2_hash = checked_json(n2_dir)
    require(n2_hash == 'd40e55f7934934cf8708b4b77cb7b4a3205f66b1d30540daa31846e682cacc80',
            'N2 checkpoint differs.')
    require(n2['informative_95_percent_intervals'] is False and n2['release_authorized'] is False,
            'N2 fallback must remain enforced.')
    for relative, expected in n2['source_sha256'].items():
        checked_bytes(C4_ROOT / relative, expected)
    n3_dir = root / 'c4_artifacts/safety/adult' / VERSION / 'n3-safety-v1'
    n3, n3_hash = checked_json(n3_dir)
    require(n3_hash == 'fa9a8eb674c0fc387e854c71316b57863f81bd79ab6cec9f935b926aa8c60c3a' and
            n3['n2_metadata_sha256'] == n2_hash and n3['n1_binding'] == n2['n1_binding'],
            'N3 checkpoint or stage links differ.')
    require(n3['release_authorized'] is False and n3['policy_calibrated'] is False,
            'N3 stage declarations differ.')
    for relative, expected in n3['source_sha256'].items():
        checked_bytes(C4_ROOT / relative, expected)
    checked_bytes(n3_dir / 'specification.json', n3['specification_sha256'])
