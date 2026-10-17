"""Require every outer method decision to be frozen before opening any outer future."""
from _common import parser, resolve_data_root, save_json, utc_now, git_context
from _outer_pipeline import (execution_binding, registry_directory, registry_definition,
                             verify_pre_release_registry)


def main():
    p = parser(__doc__)
    p.add_argument('--freeze', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    execution_binding(root, phase='freeze')
    output = registry_directory(root)
    if output.exists():
        _, digest = verify_pre_release_registry(root)
        print('REUSE VERIFIED ALL-56 PRE-FUTURE REGISTRY:', digest, flush=True)
        return
    definition = registry_definition(root, forbid_future=True)
    print('ALL 56 OUTER CASES / ALL 17 METHOD DECISIONS FROZEN; NO OUTER FUTURE EXISTS', flush=True)
    if not args.freeze:
        print('PLAN ONLY. Use --freeze to save the whole-fold decision registry.')
        return
    digest = save_json(output/'registry.json', {'definition':definition,
                       'created_at_utc':utc_now(), **git_context()})
    verify_pre_release_registry(root)
    print('REGISTRY:', output/'registry.json')
    print('REGISTRY SHA256:', digest)


if __name__ == '__main__':
    main()
