"""Freeze whole-fold evidence-context support before any policy future evaluation."""
from _common import parser,resolve_data_root,require,save_json,git_context,utc_now
from _policy_pipeline import VERSION,build_context_registry,verify_context_support


def main():
    p=parser(__doc__);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    root=resolve_data_root(args.data_root)
    output=root/'c4_artifacts/policy_pre_release/adult'/VERSION/'policy-context-support-v1'
    if output.exists():
        _,digest=verify_context_support(root)
        print('REUSE VERIFIED WHOLE-FOLD POLICY CONTEXT SUPPORT:',digest)
        return
    definition=build_context_registry(root,forbid_future=True)
    print('ALL 44 POLICY EVIDENCE / FORECAST FREEZES COMPLETE; FUTURE STAGES PRESENT: 0')
    print('CONTEXT SUPPORT:',definition['distinct_admitted_groups_by_context'])
    print('UNSUPPORTED CONTEXTS:',definition['unsupported_contexts'])
    if not args.freeze:
        print('PLAN ONLY. Use --freeze; no future labels read.')
        return
    output.mkdir(parents=True,exist_ok=False)
    digest=save_json(output/'support.json',{'definition':definition,'created_at_utc':utc_now(),**git_context()})
    print('PRE-FUTURE POLICY CONTEXT SUPPORT FROZEN:',output/'support.json')
    print('SUPPORT SHA256:',digest)


if __name__=='__main__':main()
