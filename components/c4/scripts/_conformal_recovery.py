"""Narrow recovery checks for the observed pre-fit PyTorch import failure."""
from _common import checked_bytes, require, sha

SCENARIO = 'adult-ns123-r012-interaction-low-broadened_slice-targeted_repair'
GROUP = 'adult-ns123-r012'
SEED = 411113177


def validate_interrupted_case(claim, failure, row, n1_binding):
    require(row['scenario_id'] == SCENARIO and row['group_id'] == GROUP and
            row['split_role'] == 'conformal_calibration' and row['generator_seed'] == SEED,
            'Recovery is restricted to the originally interrupted conformal case and seed.')
    require(row['repair_condition'] == 'targeted_repair' and row['label_flip_fraction'] == 0,
            'Recovery requires the original uncorrupted base-package condition.')
    require(claim['scenario'] == row and claim['n1_binding'] == n1_binding,
            'Original claim/scenario/N1 binding differs.')
    require(claim['full_execution_enabled'] is False and claim['candidate_training_executed'] is False and
            claim['future_evaluation_executed'] is False, 'Original claim scope differs.')
    require(failure['scenario_id'] == SCENARIO and failure['exception_type'] == 'OSError' and
            'WinError 126' in failure['message'] and 'shm.dll' in failure['message'],
            'Recorded failure is not the observed PyTorch DLL import failure.')


def retain_files(paths):
    """Bind originals before recovery; later writes must not mutate them."""
    result = {}
    for path in paths:
        require(path.is_file(), 'Expected original evidence file is missing: ' + str(path))
        result[str(path)] = sha(path.read_bytes())
    return result


def verify_retained(hashes):
    for path, expected in hashes.items():
        checked_bytes(path, expected)
