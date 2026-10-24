"""Freeze operational policy candidates before any policy future outcomes are opened."""
import json
from pathlib import Path
from _common import C4_ROOT, checked_bytes, checked_json, git_context, parser, require, resolve_data_root, save_json, sha, utc_now
from _conformal_group import VERSION, verify_n1
from c4.risk.benchmark_n3 import BOUNDARIES
from c4.risk.benchmark_policy import BUFFER_GRID, candidate_policies

N2_HASH='b79af74ecd63029e34461e6a9632225fb50a033e405fc5ffe8683b9da0d32934'
N3_HASH='fbe087c0bd07e1fd45a424f87d8ad190c86cf552c4e219b076008009e0a93286'


def main():
    p=parser(__doc__)
    p.add_argument('--freeze',action='store_true')
    args=p.parse_args()
    root=resolve_data_root(args.data_root)
    n1=verify_n1(root)
    path=C4_ROOT/'configs/policy_benchmark_v1.json'
    data=path.read_bytes();config=json.loads(data)
    require(config['safety_boundaries']==BOUNDARIES and config['additional_buffer_grid_fraction']==list(BUFFER_GRID)
            and config['calibration_fold']=='policy_calibration' and config['assigned_groups']==12
            and config['planned_cases']==444 and config['primary_method']=='lightgbm'
            and config['production_release_authorized'] is False,'Policy definition differs from implementation.')
    require(config['selection']['minimum_auto_deploy_groups']==3 and
            config['selection']['empirical_conditional_unsafe_ceiling']==.05,'Policy empirical selection definition differs.')
    base=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    base_md,base_hash=checked_json(base,'freeze')
    require(base_hash==n1['protocol_freeze_sha256'],'Original protocol differs.')
    manifest=json.loads(checked_bytes(base/'scenario_manifest.json',base_md['scenario_manifest_sha256']))
    groups=[g['group_id'] for g in manifest['groups'] if g['split_role']=='policy_calibration']
    rows=[r for r in manifest['scenarios'] if r['split_role']=='policy_calibration']
    require(len(groups)==len(set(groups))==12 and len(rows)==444 and
            len({r['scenario_id'] for r in rows})==444,'Policy manifest inventory differs.')
    require(all(sum(r['group_id']==g for r in rows)==37 for g in groups),'Policy group sizes differ.')
    benchmark=root/'c4_artifacts/benchmark_runs/adult'/VERSION
    require(not any((benchmark/r['scenario_id']/'future_research-v1').exists() for r in rows),
            'Policy future-stage artifacts already exist; do not claim a pre-outcome freeze.')
    n2dir=root/'c4_artifacts/conformal/adult'/VERSION/'n2-joint-v2'
    n3dir=root/'c4_artifacts/safety/adult'/VERSION/'n3-safety-v2'
    source_links={}
    for directory,expected in ((n2dir,N2_HASH),(n3dir,N3_HASH)):
        md,digest=checked_json(directory)
        require(digest==expected and md['n1_binding']==n1 and md['release_authorized'] is False,'Frozen N2/N3 binding differs.')
        source_links[str(directory/'metadata.json')]=digest
        for relative,value in md['source_sha256'].items():checked_bytes(C4_ROOT/relative,value)
        spec,spec_hash=checked_json(directory,'specification')
        require(spec_hash==md['specification_sha256'],'N2/N3 specification differs.')
        source_links[str(directory/'specification.json')]=spec_hash
    candidates=candidate_policies()
    require(len(candidates['global'])==4 and len(candidates['adaptive'])==20,'Policy candidate inventory differs.')
    print('FROZEN N1 / N2-v2 / N3-v2 / POLICY MANIFEST: PASS')
    print('POLICY GROUPS: 12; CASES: 444; FUTURE STAGES PRESENT: 0')
    print('GLOBAL CANDIDATES: 4; ADAPTIVE MONOTONE CANDIDATES: 20')
    print('BUFFERS (ACCURACY-MARGIN FRACTION):',list(BUFFER_GRID))
    print('EVIDENCE QUALITY: guide heuristic; not a calibrated failure probability')
    print('SELECTION: empirical grouped risk/automation; NO 95% conditional-risk guarantee')
    print('HARD GATES / N2 FALLBACK / N3 UNCERTAINTY CANNOT BE OVERRIDDEN')
    print('NO POLICY OUTCOMES READ; NO POLICY FITTED; NO PRODUCTION RELEASE AUTHORIZED')
    if not args.freeze:
        print('PLAN ONLY. Review investigator-defined grid/support/risk criteria, commit/push, then use --freeze.')
        return
    git=git_context()
    require(not git['working_tree_dirty'],'Commit reviewed configuration and code before freezing.')
    output=root/'c4_artifacts/policy_protocols/adult'/VERSION/config['policy_specification_version']
    require(not output.exists(),'Policy specification already exists; preserve it.')
    paths=[path,Path(__file__),C4_ROOT/'src/c4/risk/benchmark_policy.py',C4_ROOT/'src/c4/risk/benchmark_n3.py']
    code_hashes={str(p.relative_to(C4_ROOT)):sha(p.read_bytes()) for p in paths}
    output.mkdir(parents=True,exist_ok=False)
    definition_hash=save_json(output/'definition.json',config)
    candidate_hash=save_json(output/'candidates.json',candidates)
    digest=save_json(output/'freeze.json',{'purpose':'frozen_operational_global_and_adaptive_policy_specification',
        'created_at_utc':utc_now(),**git,'definition_sha256':definition_hash,'candidates_sha256':candidate_hash,
        'n1_binding':n1,'n2_metadata_sha256':N2_HASH,'n3_metadata_sha256':N3_HASH,
        'protocol_freeze_sha256':base_hash,'scenario_manifest_sha256':base_md['scenario_manifest_sha256'],
        'policy_group_ids_in_manifest_order':groups,'planned_scenario_ids':[r['scenario_id'] for r in rows],
        'source_sha256':code_hashes,'source_links':source_links,'policy_future_stages_present':False,
        'policy_outcomes_read':False,'policy_calibrated':False,'outer_test_evaluated':False,
        'formal_conditional_risk_guarantee_established':False,'release_authorized':False})
    print('FROZEN POLICY SPECIFICATION:',output)
    print('FREEZE SHA256:',digest)
    print('SPECIFICATION ONLY; GLOBAL/N4 CALIBRATION AND FINAL CONTROLLER INTEGRATION PENDING')


if __name__=='__main__':main()
