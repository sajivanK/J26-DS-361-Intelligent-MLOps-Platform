"""Verify all 592 outer admission records, without reading outer future outcomes."""
import csv
import io
from _common import (parser, resolve_data_root, checked_json, checked_bytes, require,
                     save_json, utc_now, git_context, sha)
from _outer_group import (VERSION, load_manifest,
                          outer_groups, outer_group_cases, runs_directory)
from _outer_admission_verification import verify_report, verify_revision, ORIGINAL_OUTER_FREEZE
from _outer_admission_verification import existing_record


def collect(root):
    frozen, outer_hash = verify_revision(root, full=True)
    manifest, original_hash, _ = load_manifest(root)
    summaries, records, links = [], [], {}
    for group in outer_groups(manifest):
        planned = outer_group_cases(manifest, group)
        summary = verify_report(root, group, planned, original_hash, frozen['binding']['n1_binding'])
        summaries.append(summary)
        links[group] = summary['admission_sha256']
        actual = [existing_record(root, row, original_hash, frozen['binding']['n1_binding']) for row in planned]
        require(all(r is not None for r in actual), 'Missing outer admission record.')
        records.extend(actual)
    require(len(summaries) == 16 and len(records) == 592 and
            len({r['scenario_id'] for r in records}) == 592 and
            all(r['split_role'] == 'outer_test' for r in records), 'Outer admission inventory differs.')
    admitted = sum(s['admitted'] for s in summaries)
    require(sum(s['blocked'] for s in summaries) == 592 - admitted, 'Outer counts differ.')
    return frozen, outer_hash, summaries, records, links


def main():
    p = parser(__doc__)
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    frozen, outer_hash, summaries, records, links = collect(root)
    admitted = sum(r['admitted'] for r in summaries)
    nonempty = sum(r['admitted'] > 0 for r in summaries)
    for record in summaries:
        print('VERIFIED GROUP:', record['group_id'], 'ADMITTED:', record['admitted'], 'BLOCKED:', record['blocked'])
    print('OUTER ADMISSION AUDIT: 16 GROUPS / 592 CASES; ADMITTED:', admitted, 'BLOCKED:', 592-admitted)
    print('NONEMPTY GROUPS:', nonempty, 'ALL-BLOCKED GROUPS:', 16-nonempty)
    print('RULE / SCENARIO IDs / FOLDS / REPAIR ARTIFACTS / CHECKSUMS: PASS')
    print('ADMISSION ONLY; NO OUTER CANDIDATE OR FUTURE STAGES EXECUTED')
    if not args.save:
        print('PLAN ONLY. Use --save to persist admission tables and Markdown evidence.')
        return
    output = root / 'c4_artifacts/outer_admission_audits/adult' / VERSION / 'outer-all16-admission-audit-portable-v2'
    require(not output.exists(), 'Outer admission audit exists; preserve it.')
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, rows in [('admissions.csv', records), ('group_summary.csv', summaries)]:
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        contents = buffer.getvalue().encode('utf-8')
        (output / filename).write_bytes(contents)
        files[filename] = sha(contents)
    lines = ['# Outer-test admission evidence', '',
             'Scope: controlled Adult accuracy-margin pilot; simulated upstream evidence.', '',
             f'Assigned groups: 16. Admission records: 592. Admitted: {admitted}. Blocked: {592-admitted}.',
             f'Nonempty groups: {nonempty}. All-blocked groups: {16-nonempty}.', '',
             'All original assigned groups retained. No targets imputed for blocked cases.',
             'No outer candidate training, future outcome evaluation, model/policy tuning or live release.', '',
             f'Verification scheduling freeze SHA256: `{outer_hash}`.', '',
             '| Group | Admitted | Blocked | Group admission SHA256 |',
             '|---|---:|---:|---|']
    lines += [f"| {r['group_id']} | {r['admitted']} | {r['blocked']} | {r['admission_sha256']} |" for r in summaries]
    lines += ['', 'These are admission results, not independent predictive performance or deployment evidence.',
              'Next: implement/test and separately freeze the outer candidate/decision/future pipeline.']
    contents = ('\n'.join(lines) + '\n').encode()
    (output / 'C4_OUTER_ADMISSION_EVIDENCE.md').write_bytes(contents)
    files['C4_OUTER_ADMISSION_EVIDENCE.md'] = sha(contents)
    # Recheck original records before the final audit marker.
    _, check_hash, check_summaries, check_records, check_links = collect(root)
    require((outer_hash, summaries, records, links) ==
            (check_hash, check_summaries, check_records, check_links), 'Admission records changed during audit.')
    digest = save_json(output / 'metadata.json', {
        'purpose': 'verified_all_outer_admissions_only', 'created_at_utc': utc_now(), **git_context(),
        'verification_scheduling_freeze_sha256': outer_hash, 'outer_admission_freeze_sha256': ORIGINAL_OUTER_FREEZE, 'binding': frozen['binding'],
        'groups': [r['group_id'] for r in summaries], 'assigned_groups': 16,
        'admission_cases': 592, 'admitted': admitted, 'blocked': 592-admitted,
        'nonempty_groups': nonempty, 'all_blocked_groups': 16-nonempty,
        'group_admission_sha256': links, 'files': files,
        'outer_future_evaluated': False, 'N1_refitted': False, 'N2_refitted': False,
        'policy_reselected': False, 'production_release_authorized': False,
    })
    for filename, expected in files.items():
        checked_bytes(output / filename, expected)
    print('OUTPUT:', output)
    print('METADATA SHA256:', digest)
    print('NEXT: separately frozen outer candidate / decision / future harness.')


if __name__ == '__main__':
    main()
