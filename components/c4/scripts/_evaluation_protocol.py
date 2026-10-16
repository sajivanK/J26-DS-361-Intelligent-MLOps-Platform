"""Validate pre-outer-test PP1 comparison definitions; no experimental execution."""
from fractions import Fraction

VERSION = 'adult-simulation-pp1-v2'
CONTROLLER_HASH = 'de09b0220f25b3c82f8849dd2becef4f3301806dfcb1e701d5108b0d252fb4a7'
VERIFICATION_HASH = 'fe0a26d19aae0bc6319c817754089773caf3a802169231713df8380fb2bf742e'
PROTOCOL_VERSION = 'pp1-evaluation-spec-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def conformal_rank(groups, targets=1):
    require(type(groups) is int and groups >= 0 and type(targets) is int and targets > 0, 'Invalid group or target count.')
    rank = (groups+1) * (1-Fraction(1, 20*targets))
    return (rank.numerator + rank.denominator-1) // rank.denominator


def verify_manifest(manifest):
    groups = manifest['groups']; rows = manifest['scenarios']
    roles = {g['group_id']:g['split_role'] for g in groups}
    require(len(roles) == len(groups) == 80, 'Original 80-group manifest required; duplicate groups forbidden.')
    require(len({r['scenario_id'] for r in rows}) == len(rows) == 2960, 'Original unique scenario inventory differs.')
    require(all(roles.get(r['group_id']) == r['split_role'] for r in rows), 'Group-fold assignment differs.')
    outer = [g['group_id'] for g in groups if g['split_role'] == 'outer_test']
    require(len(outer) == 16 and all(sum(r['group_id']==g for r in rows) == 37 for g in outer), 'Expected sixteen 37-case outer groups.')
    return outer, [r['scenario_id'] for r in rows if r['split_role'] == 'outer_test']


def validate_definition(config):
    require(config['protocol_version'] == PROTOCOL_VERSION and config['dataset'] == 'adult'
            and config['target_units'] == 'accuracy_margin_fraction', 'Wrong PP1 evaluation scope.')
    require([r['id'] for r in config['baselines']] == [f'B{i}' for i in range(1,9)], 'B1-B8 required exactly once in order.')
    require([r['id'] for r in config['ablations']] == [f'A{i}' for i in range(1,9)], 'A1-A8 required exactly once in order.')
    require(config['safety_boundaries'] == {'future_global_margin':0., 'future_slice_margin':0., 'future_nonaffected_margin':-.02}, 'Frozen safety boundaries cannot change.')
    require(config['outer_assigned_groups'] == 16 and config['planned_outer_cases'] == 592, 'Outer inventory differs.')
    require(config['primary_forecaster'] == 'lightgbm' and config['primary_policy_family'] == 'adaptive', 'Primary method differs.')
    require(config['shared_controls']['hard_gates_retained'] is True and config['shared_controls']['blocked_target_imputation'] is False,
            'Keep hard gates and never impute blocked-case targets.')
    require(config['statistics']['bootstrap_replicates'] == 2000 and config['statistics']['bootstrap_seed'] == 20261016
            and config['statistics']['bootstrap_unit'] == 'assigned_replication_group_with_replacement'
            and config['statistics']['paired_draws_shared_across_methods'] is True,
            'Frozen bootstrap definition differs.')
    require(config['statistics']['zero_deployment_conditional_risk'] == 'null_not_zero'
            and config['statistics']['minimum_valid_bootstrap_fraction'] == .95, 'Undefined-statistic handling differs.')
    require(config['bonferroni']['targets'] == 3 and config['bonferroni']['joint_alpha'] == .05
            and config['bonferroni']['insufficient_support'] == 'full_margin_bounds_disable_AUTO_DEPLOY'
            and config['bonferroni']['rank_clipping_allowed'] is False, 'Bonferroni fallback cannot be bypassed.')
    require(config['ablations'][7]['reuse'] == 'B8' and config['ablations'][5]['reuse_interval_artifacts'] == 'B5', 'Declared equivalent components differ.')
    require(all(r['scope'] == 'N1_feature_ablation_with_other_controls_retained' for r in config['ablations'][:3]),
            'Feature ablations cannot be represented as whole-component removal.')
    require(config['outer_execution_authorized'] is False and config['comparison_implementations_verified'] is False,
            'Definitions alone cannot authorize outer execution.')
    return config
