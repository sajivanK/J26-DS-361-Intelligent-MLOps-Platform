"""Administrative partition audit; calculates no model outcomes."""
from _common import parser, resolve_data_root, manifest_root, name, utc_now, save_json, git_context
from c4.data import load_adult_dataset
from c4.data.splits import AdultSplitConfig, split_adult_roles
from c4.data.split_manifest import load_adult_split_manifest


def main():
    p = parser(__doc__)
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--report', type=__import__('pathlib').Path, default=None,
                   help='Optional new external JSON report path; refuses overwriting.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    manifest = load_adult_split_manifest(manifest_root(root, args.manifest_version), data_root=root)
    splits = split_adult_roles(load_adult_dataset(root), AdultSplitConfig(**manifest['split_config']))
    counts = {}
    for role in ('repair', 'release_validation', 'sealed_future'):
        actual = [int(value) for value in getattr(splits, role).index]
        if actual != manifest['partitions'][role]['source_row_ids']:
            raise ValueError(f'{role}: exact membership differs.')
        counts[role] = len(actual)
        print(f'{role}: EXACT MEMBERSHIP PASS ({len(actual)} rows)')
    print('SPLIT REPRODUCIBILITY: PASS')
    if args.report:
        report_path = args.report.resolve()
        from _common import REPO_ROOT
        if report_path.is_relative_to(REPO_ROOT):
            raise ValueError('Save verification reports outside the Git repository.')
        checksum = save_json(report_path, {
            'purpose': 'administrative_split_verification', 'created_at_utc': utc_now(),
            'manifest_version': args.manifest_version, 'exact_membership': 'PASS',
            'counts': counts, 'future_performance_evaluated': False, **git_context(),
        })
        print('REPORT:', report_path)
        print('REPORT SHA256:', checksum)


if __name__ == '__main__':
    main()
