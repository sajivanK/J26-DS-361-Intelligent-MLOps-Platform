"""Manifest-bound selection for additional policy-calibration groups."""

from _common import checked_json, require
from _policy_group import policy_group_cases, verify_n1, VERSION, N1_SPEC_HASH, N1_RUN_HASH
from pathlib import Path
import json
from _common import checked_bytes


def select_admitted_cases(manifest, report, group_id):
    planned = policy_group_cases(manifest, group_id)
    require(report['group_id'] == group_id and report['cases'] == 37,
            'Wrong group or incomplete admission count.')
    records = report['records']
    by_id = {r['scenario_id']: r for r in records}
    require(len(records) == len(by_id) == 37, 'Duplicate or missing admission cases.')
    require(set(by_id) == {r['scenario_id'] for r in planned}, 'Admission IDs differ from manifest.')
    require(report['n1_binding']['specification_sha256'] == N1_SPEC_HASH and
            report['n1_binding']['run_metadata_sha256'] == N1_RUN_HASH, 'Admission links to different N1.')
    selected = []
    fields = ('group_id', 'split_role', 'mechanism', 'magnitude',
              'diagnosis_condition', 'repair_condition', 'is_no_drift_control')
    allowed = {'READY_FOR_SCOPED_CANDIDATE_VALIDATION', 'BLOCKED_ESCALATE'}
    for row in planned:
        record = by_id[row['scenario_id']]
        require(all(record[k] == row[k] for k in fields), 'Admission factor/fold differs from manifest.')
        require(record['simulation_admission'] in allowed, 'Unknown admission status.')
        if record['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION':
            require(record['screen_status'] == 'PASS', 'Admitted case has no passing screen.')
            selected.append(record)
    require(report['scoped_admissible_cases'] == len(selected) and
            report['blocked_cases'] == 37 - len(selected), 'Admission totals differ.')
    # No future targets, metrics or outcomes are used to decide membership.
    return sorted(selected, key=lambda r: r['scenario_id'])


def general_group_context(root, group_id, scenario_id):
    verify_policy_spec(root)
    n1_binding = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    freeze, freeze_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == freeze['scenario_manifest_sha256'], 'Frozen manifest link differs.')
    require(freeze['full_execution_enabled'] is False, 'Unexpected execution scope.')
    run_root = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    report, report_hash = checked_json(run_root / 'group_admissions' / group_id / 'v1', 'group_admissions')
    require(report['freeze_sha256'] == freeze_hash == n1_binding['protocol_freeze_sha256'] and
            report['n1_binding'] == n1_binding, 'Group protocol/N1 binding differs.')
    selected = select_admitted_cases(manifest, report, group_id)
    matches = [r for r in selected if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Scenario is not admitted in the requested policy group.')
    record = matches[0]
    admission, checksum = checked_json(run_root / scenario_id, 'admission')
    planned = next(r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id)
    require(checksum == record['admission_sha256'] and admission['scenario'] == planned and
            admission['freeze_sha256'] == freeze_hash and admission['simulation_admission'] ==
            'READY_FOR_SCOPED_CANDIDATE_VALIDATION', 'Scenario admission/provenance differs.')
    require(admission['n1_binding'] == n1_binding, 'Scenario N1 binding differs.')
    return report, report_hash, record


def group_context(root, scenario_id):
    """Resolve the group from the frozen manifest, never from an ID substring."""
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    manifest, _ = checked_json(frozen_dir, 'scenario_manifest')
    matches = [r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Scenario is missing or duplicated in the manifest.')
    return general_group_context(root, matches[0]['group_id'], scenario_id)


def verify_freeze_dependencies(snapshot):
    require(snapshot.get('complete_decision_evidence_frozen') is True and
            snapshot.get('future_evaluation_executed') is False, 'Complete original pre-future freeze is required.')
    require(snapshot.get('scope') == 'one_admitted_policy_scenario_under_simulated_upstream_evidence' and
            snapshot['scenario']['split_role'] == 'policy_calibration', 'Wrong policy freeze scope.')
    require(snapshot['n1_binding']['specification_sha256'] == N1_SPEC_HASH and
            snapshot['n1_binding']['run_metadata_sha256'] == N1_RUN_HASH, 'Wrong N1 model/specification binding.')
    require(snapshot.get('policy_specification_sha256') == POLICY_SPEC_HASH, 'Wrong operational policy specification.')
    dependencies = snapshot.get('dependencies', {})
    require(bool(dependencies), 'Missing frozen dependencies.')
    require(dependencies.get(snapshot['policy_forecasts_path']) == snapshot['policy_forecasts_sha256'], 'Missing pre-future policy forecasts.')
    require(any(Path(p).name == 'freeze.json' and h == POLICY_SPEC_HASH for p,h in dependencies.items()), 'Policy specification not pinned.')
    n1_dir = Path(snapshot['n1_run_directory'])
    required = {str(n1_dir / 'metadata.json'): N1_RUN_HASH,
                str(n1_dir.parent / 'specification.json'): N1_SPEC_HASH}
    model_metadata = json.loads(checked_bytes(n1_dir / 'metadata.json', N1_RUN_HASH))
    for filename, expected in model_metadata['files'].items():
        required[str(n1_dir / filename)] = expected
    require(all(dependencies.get(path) == expected for path, expected in required.items()),
            'Policy freeze must pin all original N1 artifacts.')
    require(snapshot.get('n1_forecasts_frozen') is True, 'Pre-future N1 forecasts are missing.')
    forecast_path = snapshot['n1_forecasts_path']
    require(dependencies.get(forecast_path) == snapshot['n1_forecasts_sha256'], 'N1 forecast dependency is missing.')
    for path, checksum in dependencies.items():
        checked_bytes(path, checksum)

POLICY_SPEC_HASH='113e2c5cd4007301ecddc2fd8bf3a226c1b2e3f1acd108c49a3b2dc389b2d505'
N2_V2_HASH='b79af74ecd63029e34461e6a9632225fb50a033e405fc5ffe8683b9da0d32934'
N3_V2_HASH='fbe087c0bd07e1fd45a424f87d8ad190c86cf552c4e219b076008009e0a93286'


def verify_policy_spec(root):
    from _common import C4_ROOT
    from c4.risk.benchmark_policy import candidate_policies
    directory=root/'c4_artifacts/policy_protocols/adult'/VERSION/'policy-spec-v1'
    frozen,digest=checked_json(directory,'freeze')
    require(digest==POLICY_SPEC_HASH and frozen['n1_binding']==verify_n1(root) and
            frozen['n2_metadata_sha256']==N2_V2_HASH and frozen['n3_metadata_sha256']==N3_V2_HASH,
            'Frozen operational policy/model bindings differ.')
    for relative,expected in frozen['source_sha256'].items():checked_bytes(C4_ROOT/relative,expected)
    for path,expected in frozen['source_links'].items():checked_bytes(path,expected)
    definition,definition_hash=checked_json(directory,'definition')
    candidates,candidate_hash=checked_json(directory,'candidates')
    require(definition_hash==frozen['definition_sha256'] and candidate_hash==frozen['candidates_sha256']
            and candidates==candidate_policies(),'Frozen policy definitions/candidates differ.')
    for path,expected in ((root/'c4_artifacts/conformal/adult'/VERSION/'n2-joint-v2',N2_V2_HASH),
                          (root/'c4_artifacts/safety/adult'/VERSION/'n3-safety-v2',N3_V2_HASH)):
        md,md_hash=checked_json(path)
        require(md_hash==expected,'N2/N3 checkpoint differs.')
        for relative,value in md['source_sha256'].items():checked_bytes(C4_ROOT/relative,value)
    return frozen,directory


def policy_inventory(root):
    frozen_spec,_=verify_policy_spec(root)
    base=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    freeze,base_hash=checked_json(base,'freeze')
    manifest=json.loads(checked_bytes(base/'scenario_manifest.json',freeze['scenario_manifest_sha256']))
    groups=frozen_spec['policy_group_ids_in_manifest_order']
    require(len(groups)==12 and len(set(groups))==12,'Policy group inventory differs.')
    reports={}; selected=[]; admitted_counts=[]
    for group in groups:
        report,digest=checked_json(root/'c4_artifacts/benchmark_runs/adult'/VERSION/'group_admissions'/group/'v1','group_admissions')
        require(report['freeze_sha256']==base_hash and report['n1_binding']==frozen_spec['n1_binding'],'Policy admission binding differs.')
        rows=select_admitted_cases(manifest,report,group)
        reports[group]={'report':report,'sha256':digest}; selected.extend(rows);admitted_counts.append(len(rows))
    require(admitted_counts==[8,4,2,4,2,4,6,0,4,6,2,2] and len(selected)==44,'Reported 44-admission policy inventory differs.')
    return manifest,groups,reports,selected


def context_support_from_rows(rows):
    """Count distinct admitted groups using pre-release contexts, never outcomes."""
    from c4.risk.benchmark_policy import CONTEXTS
    require(len({r['scenario_id'] for r in rows})==len(rows),'Duplicate context-support scenario.')
    require(all(r['context'] in CONTEXTS and r['split_role']=='policy_calibration' for r in rows),
            'Wrong context or fold in support calculation.')
    counts={c:len({r['group_id'] for r in rows if r['context']==c}) for c in CONTEXTS}
    return {'distinct_admitted_groups_by_context':counts,
            'unsupported_contexts':[c for c in CONTEXTS if counts[c]<3],
            'minimum_distinct_groups':3}


def build_context_registry(root, *, forbid_future):
    from _common import C4_ROOT, sha
    from c4.risk.benchmark_policy import evidence_context
    manifest,groups,reports,selected=policy_inventory(root)
    runs=root/'c4_artifacts/benchmark_runs/adult'/VERSION
    if forbid_future:
        planned=[r for r in manifest['scenarios'] if r['group_id'] in groups]
        require(not any((runs/r['scenario_id']/'future_research-v1').exists() for r in planned),
                'A policy future stage exists before complete context freeze.')
    contexts=[];freeze_links={};forecast_links={}
    for row in selected:
        run=runs/row['scenario_id']/'pre_release_freeze-v1'
        snapshot,digest=checked_json(run,'freeze')
        if forbid_future:verify_freeze_dependencies(snapshot)
        require(snapshot['scenario']['scenario_id']==row['scenario_id'] and
                snapshot['scenario']['group_id']==row['group_id'] and
                snapshot['scenario']['split_role']=='policy_calibration' and
                snapshot['policy_specification_sha256']==POLICY_SPEC_HASH,'Wrong pre-future policy snapshot.')
        forecasts,forecast_hash=checked_json(run,'policy_forecasts')
        n1,_=checked_json(run,'n1_forecasts')
        require(forecast_hash==snapshot['policy_forecasts_sha256'] and forecasts['future_targets_included'] is False,
                'Policy forecasts differ or contain targets.')
        context=evidence_context(n1['decision_time_features'])
        require(context==forecasts['evidence_context'],'Context differs from frozen pre-release features.')
        contexts.append({'scenario_id':row['scenario_id'],'group_id':row['group_id'],
                         'split_role':'policy_calibration',**context})
        freeze_links[row['scenario_id']]=digest;forecast_links[row['scenario_id']]=forecast_hash
    files=['_policy_pipeline.py','freeze_policy_context_support.py','freeze_policy_case.py','evaluate_policy_case_future.py']
    source_hashes={str(C4_ROOT/'scripts'/f):sha((C4_ROOT/'scripts'/f).read_bytes()) for f in files}
    return {'purpose':'all_policy_pre_release_context_support_before_future',
        'policy_specification_sha256':POLICY_SPEC_HASH,'group_ids_in_fixed_order':groups,
        'admitted_scenarios':44,'blocked_cases':400,'nonempty_groups':11,'all_blocked_groups':1,
        'group_admission_sha256':{g:r['sha256'] for g,r in reports.items()},
        'scenario_freeze_sha256':freeze_links,'policy_forecast_sha256':forecast_links,
        'contexts':contexts,**context_support_from_rows(contexts),'source_sha256':source_hashes,
        'future_labels_used':False,'production_release_authorized':False}


def verify_context_support(root):
    directory=root/'c4_artifacts/policy_pre_release/adult'/VERSION/'policy-context-support-v1'
    support,digest=checked_json(directory,'support')
    require(support['definition']==build_context_registry(root,forbid_future=False),
            'Whole-fold pre-future support registry differs.')
    return support['definition'],digest
