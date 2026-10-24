"""Fixed training-group selection and paired repair corruption."""
import numpy as np
import pandas as pd
from _common import require
from _scoped_benchmark import select_training_scenario


def training_group_cases(manifest, group_id=None):
    group_id = group_id or select_training_scenario(manifest)['group_id']
    rows = [r for r in manifest['scenarios'] if r['group_id'] == group_id]
    require(len(rows) == 37 and all(r['split_role'] == 'forecast_train' for r in rows),
            'Exactly 37 planned cases in a forecasting-training group are required.')
    require(sum(r['is_no_drift_control'] for r in rows) == 1, 'One no-drift control is required.')
    require(len({r['scenario_id'] for r in rows}) == 37, 'Duplicate scenario IDs.')
    drifted = [r for r in rows if not r['is_no_drift_control']]
    expected = {(m, s, d, q) for m in ('marginal_multi','interaction','combined') for s in (.2,.5,.8)
                for d in ('exact_slice','broadened_slice') for q in ('targeted_repair','label_corrupted_repair')}
    actual = {(r['mechanism'],r['magnitude'],r['diagnosis_condition'],r['repair_condition']) for r in drifted}
    require(actual == expected, 'Training group does not contain the frozen factorial grid.')
    # Each targeted package must be produced before its paired corruption condition.
    return sorted(rows, key=lambda r:(r['shared_base_package_id'], r['repair_condition'] != 'targeted_repair', r['scenario_id']))


def corrupt_labels(base, *, fraction, seed):
    require(fraction == .2 and len(base) == 500, 'Frozen corruption requires 20% of 500 rows.')
    require(base['target'].isin([0,1]).all(), 'Corruption requires binary targets.')
    positions = np.sort(np.random.default_rng(seed).choice(len(base), size=100, replace=False))
    corrupted = base.copy(deep=True)
    corrupted.iloc[positions, corrupted.columns.get_loc('target')] = 1-base.iloc[positions]['target'].to_numpy()
    mask = pd.DataFrame({'synthetic_row_position': np.arange(len(base)),
                         'label_flipped': np.isin(np.arange(len(base)), positions)})
    pd.testing.assert_frame_equal(corrupted.drop(columns='target'), base.drop(columns='target'), check_exact=True)
    require(int((corrupted['target'] != base['target']).sum()) == 100, 'Unexpected corruption count.')
    return corrupted, mask
