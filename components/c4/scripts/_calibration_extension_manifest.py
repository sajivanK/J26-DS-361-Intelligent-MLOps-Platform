"""Deterministic extension design; no observed outcomes or model fits."""
import copy
from _benchmark_manifest import seed


def seed_labels(row):
    suffix = row['scenario_id'][len(row['group_id'])+1:]
    condition, strength, diagnosis, repair = suffix.split('-')
    condition_key = condition+':'+strength
    base_key = condition_key+':'+diagnosis
    labels = {'generator_seed': 'generator:'+base_key, 'c1_mock_seed': 'c1:'+condition_key,
              'c2_bootstrap_seed': 'c2:'+diagnosis, 'corruption_seed': 'corruption:'+base_key}
    labels.update({'drift:'+role: 'drift:'+condition_key+':'+role
                   for role in ('repair','release_validation','sealed_future')})
    return labels


def build_extension(original, definition):
    if (definition['split_role'] != 'conformal_extension' or definition['alpha'] != .05
        or definition['expected_groups'] != 20 or definition['expected_cases_per_group'] != 37
        or definition['seed_namespaces'] != [42,123,2026,3407]
        or definition['replication_indices'] != [20,21,22,23,24]
        or definition['full_execution_enabled'] is not False):
        raise ValueError('Unexpected extension design.')
    old_groups = original['groups']
    if len(old_groups) != 80 or len({g['group_id'] for g in old_groups}) != 80:
        raise ValueError('Original frozen 80-group manifest required.')
    old_scenarios = original['scenarios']
    if len(old_scenarios) != 2960 or len({r['scenario_id'] for r in old_scenarios}) != 2960:
        raise ValueError('Original frozen 2960-scenario manifest required.')
    seed_registry = {}
    def register(value, identity):
        if value in seed_registry and seed_registry[value] != identity:
            raise ValueError('Derived seed collision; do not execute or silently change seeds.')
        seed_registry[value] = identity
    for row in old_scenarios:
        for field,label in seed_labels(row).items():
            value = row['drift_seeds'][field[6:]] if field.startswith('drift:') else row[field]
            register(value,(row['group_id'],label))
    template_id = old_groups[0]['group_id']
    template = [r for r in old_scenarios if r['group_id']==template_id]
    if len(template) != 37:
        raise ValueError('Original group template must contain 37 scenarios.')
    groups, rows = [], []
    for namespace in definition['seed_namespaces']:
        for index in definition['replication_indices']:
            identity = {'dataset':'uci_adult','namespace':namespace,'replication_index':index}
            if any(all(g.get(k)==v for k,v in identity.items()) for g in old_groups):
                raise ValueError('Extension RNG identity overlaps original group.')
            gid = f'adult-calext-v2-ns{namespace}-r{index:03d}'
            groups.append({'group_id':gid,**identity,'split_role':'conformal_extension'})
            for old in template:
                row=copy.deepcopy(old)
                suffix=old['scenario_id'][len(template_id):]
                row.update(group_id=gid,scenario_id=gid+suffix,split_role='conformal_extension')
                row['shared_base_package_id']=gid+old['shared_base_package_id'][len(template_id):]
                for field,label in seed_labels(row).items():
                    value=seed(identity,label)
                    register(value,(gid,label))
                    if field.startswith('drift:'): row['drift_seeds'][field[6:]]=value
                    else: row[field]=value
                rows.append(row)
    if len(groups)!=20 or len(rows)!=740 or len({r['scenario_id'] for r in rows})!=740:
        raise ValueError('Extension inventory differs.')
    if {r['scenario_id'] for r in rows} & {r['scenario_id'] for r in old_scenarios}:
        raise ValueError('Extension scenario ID overlaps original.')
    return {'manifest_schema_version':'1.0','extension_version':definition['extension_version'],
            'base_protocol_version':definition['base_protocol_version'],'groups':groups,'scenarios':rows,
            'experiments_executed':0,'full_execution_enabled':False,'production_release_authorized':False}
