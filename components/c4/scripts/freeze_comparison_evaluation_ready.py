"""Freeze comparison/evaluation readiness; outer harness execution remains separately gated."""
from _common import parser,resolve_data_root,require,checked_json,checked_bytes,save_json,utc_now,git_context
from _evaluation_engine_inputs import VERSION,load_inputs


def main():
    p=parser(__doc__);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    root=resolve_data_root(args.data_root);_,binding=load_inputs(root)
    directory=root/'c4_artifacts/evaluations/adult'/VERSION/'development-engine-check-v1'
    md,digest=checked_json(directory)
    require(md['purpose']=='verified_development_evaluation_engine' and md['binding']==binding and md['rows']==44
            and md['bootstrap_replicates']==2000 and md['bootstrap_seed']==20261016 and md['outer_test_evaluated'] is False,
            'Completed development engine check differs.')
    for filename,expected in md['files'].items():checked_bytes(directory/filename,expected)
    print('COMPARISON ARTIFACTS / EXECUTABLE DECISIONS / SAVED EVALUATION CHECK: PASS')
    print('EVALUATION DEFINITIONS UNCHANGED; ALL 16 OUTER GROUPS UNTOUCHED')
    print('READINESS SCOPE: comparisons and evaluation engine; outer pipeline code/freeze still required')
    if not args.freeze:
        print('PLAN ONLY. Use --freeze to save the readiness binding.');return
    git=git_context();require(not git['working_tree_dirty'],'Commit evaluation code before readiness freeze.')
    output=root/'c4_artifacts/evaluation_readiness/adult'/VERSION/'comparison-evaluation-ready-v1'
    require(not output.exists(),'Readiness freeze exists; preserve it.')
    checksum=save_json(output/'freeze.json',{'purpose':'frozen_comparison_and_evaluation_readiness',
        'created_at_utc':utc_now(),**git,'binding':binding,'development_evaluation_metadata_sha256':digest,
        'comparisons_ready':True,'evaluation_engine_ready':True,'outer_pipeline_ready':False,
        'outer_execution_authorized':False,'outer_outcomes_read':False,'production_release_authorized':False})
    print('READINESS:',output);print('FREEZE SHA256:',checksum)
    print('NEXT: implement and freeze the outer harness before executing its groups.')


if __name__=='__main__':main()
