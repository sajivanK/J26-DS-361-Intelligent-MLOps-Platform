"""Freeze admission-only authorization for all 16 original outer-test groups."""
from _common import parser, resolve_data_root, require, git_context, utc_now, save_json
from _outer_group import (VERSION, current_bindings, load_manifest, outer_groups,
                          outer_group_cases, source_hashes, assert_no_outer_artifacts,
                          protocol_directory)


def main():
    p = parser(__doc__)
    p.add_argument('--freeze', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    binding = current_bindings(root)
    manifest, _, _ = load_manifest(root)
    groups = outer_groups(manifest)
    assert_no_outer_artifacts(root, manifest)
    paths = source_hashes()
    print('CONTROLLER / COMPARISONS / EVALUATION READINESS / ORIGINAL N1: PASS')
    print('OUTER GROUPS: 16; PLANNED CASES: 592; ADMISSION BATCHES: 4 x 4 GROUPS')
    for group in groups:
        print('GROUP:', group, 'FOLD: outer_test CASES:', len(outer_group_cases(manifest, group)))
    print('SAME ADMISSION RULE / SCENARIOS / SEEDS; ALL BLOCKED CASES RETAINED')
    print('SCOPE: REPAIR GENERATION AND ADMISSION ONLY; NO OUTER FUTURE OUTCOMES')
    print('N1 / N2 / N3 / POLICIES / COMPARISON DEFINITIONS: UNCHANGED')
    if not args.freeze:
        print('PLAN ONLY. Commit/push these new files, then use --freeze.')
        return
    git = git_context()
    require(not git['working_tree_dirty'], 'Commit outer admission code before freezing.')
    output = protocol_directory(root)
    require(not output.exists(), 'Outer admission freeze exists; preserve it.')
    digest = save_json(output / 'freeze.json', {
        'purpose': 'frozen_outer_admission_only_protocol', 'created_at_utc': utc_now(), **git,
        'version': 'outer-admission-spec-v1', 'benchmark_version': VERSION,
        'binding': binding, 'source_sha256': paths, 'assigned_groups': 16,
        'planned_cases': 592, 'batches': 4, 'groups_per_batch': 4,
        'admission_execution_authorized': True,
        'outer_candidate_execution_authorized': False,
        'outer_future_execution_authorized': False,
        'admission_rule_changed': False, 'outcome_based_selection': False,
        'automatic_replacement_groups': False, 'early_stopping_for_results': False,
        'production_release_authorized': False,
        'next_gate': 'Separate downstream execution-code freeze required before outer candidate/future stages.',
    })
    print('OUTER ADMISSION SPECIFICATION:', output)
    print('FREEZE SHA256:', digest)
    print('NEXT: run_outer_admission_batch.py --batch-number 1 (plan, then --execute-admissions).')
    print('OUTER FUTURE EXECUTION: DISABLED; PRODUCTION RELEASE: False')


if __name__ == '__main__':
    main()
