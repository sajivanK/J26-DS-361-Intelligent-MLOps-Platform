"""Verify a non-executable benchmark draft and its reference-derived slices."""
from _common import parser, default_yaml
from pathlib import Path
from c4.scenarios.benchmark import load_draft_benchmark, verify_draft_sources


def main():
    p = parser(__doc__)
    p.add_argument('--config', type=Path, default=default_yaml())
    args = p.parse_args()
    draft = load_draft_benchmark(args.config)
    verified = verify_draft_sources(draft, data_root=args.data_root)
    print('YAML SHA256:', draft.yaml_sha256)
    print('INJECTION SPECIFICATIONS:', len(draft.injection_configs))
    print('PROPOSED COMBINATIONS:', draft.proposed_scenario_count)
    print('SOURCE VERIFICATION:', verified['source_verification'])
    print('EXECUTION ENABLED:', verified['execution_enabled'])
    for slice_name, rule in verified['verified_slice_rules'].items():
        print('VERIFIED SLICE:', slice_name, rule)


if __name__ == '__main__':
    main()
