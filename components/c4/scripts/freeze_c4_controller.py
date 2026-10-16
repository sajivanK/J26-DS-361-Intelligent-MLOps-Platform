"""Freeze the measured N1-N4 inference controller with its review-only policy fallback."""
from _common import parser, resolve_data_root, require, save_json, utc_now, git_context
from _controller_artifacts import bindings, controller_directory, source_hashes


def main():
    p = parser(__doc__); p.add_argument('--freeze', action='store_true'); args = p.parse_args()
    root = resolve_data_root(args.data_root)
    binding, _ = bindings(root); sources = source_hashes()
    print('N1 / N2-v2 / N3-v2 / SAVED GLOBAL-N4 POLICIES: PASS')
    print('MEASURED POLICY: review_only; AUTO_DEPLOY ENABLED: False')
    print('SCOPE: Adult accuracy-margin pilot; upstream producers simulated; no live deployment')
    print('ACCEPTANCE FIXTURES: separate synthetic technical checks, not empirical deployment evidence')
    if not args.freeze:
        print('PLAN ONLY. Commit/push the controller code, then use --freeze.'); return
    git = git_context(); require(not git['working_tree_dirty'], 'Commit controller implementation before freezing.')
    output = controller_directory(root); require(not output.exists(), 'Controller freeze exists; preserve it.')
    digest = save_json(output / 'freeze.json', {'purpose': 'frozen_adult_pilot_inference_controller',
        'binding': binding, 'source_sha256': sources, 'created_at_utc': utc_now(), **git,
        'outer_test_evaluated': False, 'production_release_authorized': False})
    print('CONTROLLER:', output); print('FREEZE SHA256:', digest)
    print('Next run verify_c4_controller.py; no outer-test execution authorized by this freeze.')


if __name__ == '__main__': main()
