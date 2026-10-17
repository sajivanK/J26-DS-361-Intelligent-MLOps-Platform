"""Freeze additive outer candidate/decision/future authorization before execution."""
from _common import parser, resolve_data_root, require, save_json, utc_now, git_context
from _outer_pipeline import (shared_binding, inventory, runs, specification_directory,
                             source_hashes, verify_execution_spec)


def main():
    p = parser(__doc__)
    p.add_argument('--freeze', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    output = specification_directory(root)
    if output.exists():
        _, digest = verify_execution_spec(root, full=True)
        print('REUSE VERIFIED EXECUTION FREEZE:', digest, flush=True)
        return
    binding = shared_binding(root)
    manifest, groups, reports, selected = inventory(root)
    for row in manifest['scenarios']:
        if row['split_role'] != 'outer_test':
            continue
        for stage in ('scoped_candidate_validation-v1', 'pre_release_freeze-v1', 'future_research-v1'):
            require(not (runs(root)/row['scenario_id']/stage).exists(),
                    'Outer downstream artifact predates execution freeze; preserve and investigate.')
    print('16 ASSIGNED GROUPS / 592 ADMISSIONS / 56 CANDIDATES / 536 BLOCKED', flush=True)
    print('ALL 17 METHOD DECISIONS FOR ALL 56 CASES MUST BE FROZEN BEFORE ANY FUTURE', flush=True)
    print('N1 / N2 / POLICIES FIXED; NO TVAE FITTING; NO PRODUCTION RELEASE', flush=True)
    if not args.freeze:
        print('PLAN ONLY. Commit/push these additive files, then use --freeze.')
        return
    git = git_context()
    require(not git['working_tree_dirty'], 'Commit code before freezing outer execution.')
    digest = save_json(output/'freeze.json', {
        'purpose':'frozen_outer_candidate_decision_future_execution', 'created_at_utc':utc_now(), **git,
        'binding':binding, 'source_sha256':source_hashes(), 'assigned_groups':16,
        'admission_records':592, 'admitted_scenarios':56, 'blocked':536,
        'candidate_training_allowed':True, 'outer_future_execution_allowed':True,
        'pre_release_registry_required_before_future':True,
        'new_forecaster_or_policy_fitting_allowed':False,
        'shared_verification_schedule':'full_before_and_after_each_phase_plus_case_dependencies',
        'production_release_authorized':False})
    print('OUTER EXECUTION FREEZE:', output)
    print('FREEZE SHA256:', digest)
    print('NEXT: run_outer_candidate_batch.py --all-groups (plan, then --execute).')


if __name__ == '__main__':
    main()
