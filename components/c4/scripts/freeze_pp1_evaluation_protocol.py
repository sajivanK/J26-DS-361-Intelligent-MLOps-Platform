"""Freeze PP1 baseline, ablation and statistical definitions before outer-test execution."""
import json
from pathlib import Path
from _common import C4_ROOT, parser, resolve_data_root, require, checked_json, checked_bytes, save_json, sha, utc_now, git_context
from _controller_artifacts import controller_directory, bindings
from _evaluation_protocol import VERSION, CONTROLLER_HASH, VERIFICATION_HASH, PROTOCOL_VERSION, conformal_rank, validate_definition, verify_manifest


def main():
    p=parser(__doc__);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    root=resolve_data_root(args.data_root)
    path=C4_ROOT/'configs/pp1_evaluation_protocol_v1.json'
    definition=validate_definition(json.loads(path.read_bytes()))
    current,_=bindings(root)
    frozen,controller_hash=checked_json(controller_directory(root),'freeze')
    require(controller_hash==CONTROLLER_HASH and frozen['binding']==current,'Measured controller checkpoint differs.')
    for relative,digest in frozen['source_sha256'].items():checked_bytes(C4_ROOT/relative,digest)
    verified,verified_hash=checked_json(controller_directory(root)/'verification-v1')
    require(verified_hash==VERIFICATION_HASH and verified['controller_freeze_sha256']==controller_hash
            and verified['replayed_decisions']==176 and verified['synthetic_fixtures']==7,'Controller verification checkpoint differs.')
    for filename,digest in verified['files'].items():checked_bytes(controller_directory(root)/'verification-v1'/filename,digest)
    base=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    protocol,base_hash=checked_json(base,'freeze')
    require(base_hash==current['n1_binding']['protocol_freeze_sha256'],'Original scenario protocol differs.')
    manifest=json.loads(checked_bytes(base/'scenario_manifest.json',protocol['scenario_manifest_sha256']))
    groups,scenarios=verify_manifest(manifest)
    runs=root/'c4_artifacts/benchmark_runs/adult'/VERSION
    require(not any((runs/sid).exists() for sid in scenarios),
            'Outer scenario artifacts already exist; do not claim a pre-outer-execution freeze.')
    print('CONTROLLER / VERIFICATION / ORIGINAL MANIFEST: PASS')
    print('B1-B8 DEFINITIONS: 8; A1-A8 DEFINITIONS: 8; A8 REUSES B8')
    print('OUTER GROUPS: 16; PLANNED CASES: 592; EXISTING OUTER SCENARIO ARTIFACTS: 0')
    print('A1-A3: N1 FEATURE ABLATIONS; OTHER CONTROLS RETAINED; SEPARATE REFITS REQUIRED')
    print('BONFERRONI: 37 USABLE GROUPS; REQUIRED RANK:',conformal_rank(37,3),'; FINITE ADJUSTMENT: False')
    print('B5/A6 MUST RETAIN FULL-BOUND FALLBACK; NO ADDITIONAL CALIBRATION GROUPS AUTHORIZED')
    print('BOOTSTRAP: 2000 PAIRED GROUP DRAWS; SEED: 20261016; UNDEFINED RISK REMAINS NULL')
    print('EXACT BASELINE RULES / COST WEIGHTS / STATISTICAL SETTINGS ARE DECLARED PP1 DESIGN CHOICES')
    print('DEFINITIONS ONLY; COMPARISON IMPLEMENTATION / ARTIFACT READINESS FREEZE STILL REQUIRED')
    if not args.freeze:
        print('PLAN ONLY. Review the configuration, commit/push, then use --freeze.');return
    git=git_context();require(not git['working_tree_dirty'],'Commit evaluation protocol code/configuration before freezing.')
    output=root/'c4_artifacts/evaluation_protocols/adult'/VERSION/PROTOCOL_VERSION
    require(not output.exists(),'Evaluation specification exists; preserve it.')
    sources=[path,Path(__file__),C4_ROOT/'scripts/_evaluation_protocol.py']
    hashes={str(p.relative_to(C4_ROOT)):sha(p.read_bytes()) for p in sources}
    definition_hash=save_json(output/'definition.json',definition)
    digest=save_json(output/'freeze.json',{'purpose':'frozen_pp1_comparison_and_statistical_definitions',
        'created_at_utc':utc_now(),**git,'definition_sha256':definition_hash,
        'controller_freeze_sha256':controller_hash,'controller_verification_metadata_sha256':verified_hash,
        'original_protocol_freeze_sha256':base_hash,'scenario_manifest_sha256':protocol['scenario_manifest_sha256'],
        'outer_group_ids_in_manifest_order':groups,'outer_scenario_ids':scenarios,'source_sha256':hashes,
        'comparison_implementations_verified':False,'outer_outcomes_read':False,'outer_execution_authorized':False,
        'production_release_authorized':False})
    print('EVALUATION SPECIFICATION:',output);print('FREEZE SHA256:',digest)
    print('NEXT: implement/verify comparison models and policies on development folds; keep outer groups untouched.')


if __name__=='__main__':main()
