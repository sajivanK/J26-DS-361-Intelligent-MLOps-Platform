"""Review/freeze a support extension; never generate repairs or open future outcomes."""
import json
from pathlib import Path
from _common import C4_ROOT, checked_bytes, checked_json, git_context, parser, require, resolve_data_root, save_json, sha, utc_now
from _conformal_group import VERSION, verify_n1
from _calibration_extension_manifest import build_extension


def main():
    p=parser(__doc__)
    p.add_argument('--freeze',action='store_true')
    args=p.parse_args()
    root=resolve_data_root(args.data_root)
    n1=verify_n1(root)
    base=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    frozen, frozen_hash=checked_json(base,'freeze')
    require(frozen_hash==n1['protocol_freeze_sha256'],'Base protocol and N1 differ.')
    original=json.loads(checked_bytes(base/'scenario_manifest.json',frozen['scenario_manifest_sha256']))
    protocol_bytes=checked_bytes(base/'protocol.json',frozen['protocol_sha256'])
    protocol=json.loads(protocol_bytes)
    n2dir=root/'c4_artifacts/conformal/adult'/VERSION/'n2-joint-v1'
    n2,n2hash=checked_json(n2dir)
    require(n2hash=='d40e55f7934934cf8708b4b77cb7b4a3205f66b1d30540daa31846e682cacc80'
            and n2['n1_binding']==n1 and n2['nonempty_groups']==17 and n2['quantile_rank']==18,
            'Original N2 checkpoint differs.')
    n3dir=root/'c4_artifacts/safety/adult'/VERSION/'n3-safety-v1'
    n3,n3hash=checked_json(n3dir)
    require(n3hash=='fa9a8eb674c0fc387e854c71316b57863f81bd79ab6cec9f935b926aa8c60c3a'
            and n3['n2_metadata_sha256']==n2hash,'Original N3 differs.')
    definition_path=C4_ROOT/'configs/calibration_extension_v2.json'
    definition_bytes=definition_path.read_bytes()
    definition=json.loads(definition_bytes)
    manifest=build_extension(original,definition)
    print('ORIGINAL PROTOCOL / N1 / N2 / N3 CHECKPOINTS: PASS')
    print('EXTENSION GROUPS: 20; PLANNED CASES: 740; ROLE: conformal_extension')
    for g in manifest['groups']: print('GROUP:',g['group_id'],'CASES: 37')
    print('COMBINATION: all original nonempty groups + all extension nonempty groups, one score per group')
    print('SAME ADMISSION / GENERATOR / TARGET / GROUP-MAX SCORE RULES; NO EARLY STOPPING')
    print('POLICY / OUTER-TEST GROUPS EXCLUDED; NO OUTCOMES READ OR EXPERIMENTS EXECUTED')
    if not args.freeze:
        print('PLAN ONLY. Commit/push code and configuration, then use --freeze.')
        return
    git=git_context()
    require(not git['working_tree_dirty'],'Commit the reviewed code/configuration before freezing.')
    output=root/'c4_artifacts/calibration_extensions/adult'/VERSION/definition['extension_version']
    require(not output.exists(),'Extension version already exists; preserve it.')
    paths=[Path(__file__),C4_ROOT/'scripts/_calibration_extension_manifest.py',C4_ROOT/'scripts/_benchmark_manifest.py',definition_path]
    source_hashes={str(path.relative_to(C4_ROOT)):sha(path.read_bytes()) for path in paths}
    output.mkdir(parents=True,exist_ok=False)
    definition_hash=save_json(output/'definition.json',definition)
    manifest_hash=save_json(output/'scenario_manifest.json',manifest)
    # Preserve byte-identical base protocol; separate extension definition never rewrites it.
    (output/'base_protocol.json').write_bytes(protocol_bytes)
    digest=save_json(output/'freeze.json',{'purpose':'frozen_calibration_support_extension',
        'created_at_utc':utc_now(),**git,'extension_version':definition['extension_version'],
        'base_protocol_freeze_sha256':frozen_hash,'base_protocol_sha256':frozen['protocol_sha256'],
        'base_scenario_manifest_sha256':frozen['scenario_manifest_sha256'],
        'definition_sha256':definition_hash,'scenario_manifest_sha256':manifest_hash,
        'n1_binding':n1,'original_n2_metadata_sha256':n2hash,'original_n3_metadata_sha256':n3hash,
        'unchanged_admission_definition':protocol['admission'],'unchanged_N2_definition':protocol['N2'],
        'source_sha256':source_hashes,'full_execution_enabled':False,'experiments_executed':0,
        'original_results_preserved':True,'policy_outcomes_read':False,'outer_test_outcomes_read':False,
        'requires_new_scoped_extension_runners':True,'production_release_authorized':False})
    print('FROZEN EXTENSION:',output)
    print('MANIFEST SHA256:',manifest_hash)
    print('FREEZE SHA256:',digest)
    print('SPECIFICATION ONLY; EXECUTION STILL DISABLED')


if __name__=='__main__': main()
