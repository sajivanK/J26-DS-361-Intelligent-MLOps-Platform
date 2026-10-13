"""Deterministic scenario design; never inspect outcome labels."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seed(group, label):
    return int(digest({'group': group, 'component': label})[:8], 16)


def build_manifest(protocol):
    if protocol['schema_version'] != '1.0':
        raise ValueError('Unsupported protocol schema.')
    replication = protocol['replication']
    namespaces = replication['seed_namespaces']
    n = replication['replications_per_namespace']
    counts = protocol['scenario_split']['per_namespace']
    roles = ('forecast_train', 'conformal_calibration', 'policy_calibration', 'outer_test')
    if (type(n) is not int or n < 1 or len(set(namespaces)) != len(namespaces)
            or any(type(s) is not int or not 0 <= s <= 2**32-1 for s in namespaces)):
        raise ValueError('Invalid replication design.')
    if set(counts) != set(roles) or any(type(v) is not int or v < 1 for v in counts.values()) or sum(counts.values()) != n:
        raise ValueError('Four positive role counts must sum to replications per namespace.')
    groups, scenarios, all_seeds = [], [], {}
    for namespace in namespaces:
        ordering = sorted(range(n), key=lambda i: digest({'fold_seed': protocol['scenario_split']['fold_seed'],
                                                        'namespace': namespace, 'replication': i}))
        assignments = {}
        offset = 0
        for role in roles:
            for i in ordering[offset:offset+counts[role]]:
                assignments[i] = role
            offset += counts[role]
        for i in range(n):
            group_id = f'adult-ns{namespace}-r{i:03d}'
            group = {'dataset': protocol['dataset'], 'namespace': namespace, 'replication_index': i}
            groups.append({'group_id': group_id, **group, 'split_role': assignments[i]})
            seeds = {}
            def component_seed(label):
                if label not in seeds:
                    result = seed(group, label)
                    identity = (group_id, label)
                    if result in all_seeds and all_seeds[result] != identity:
                        raise ValueError('Derived seed collision; resolve the design before freezing.')
                    all_seeds[result] = identity
                    seeds[label] = result
                return seeds[label]
            def add(condition, strength_name, magnitude, diagnosis, repair, control=False):
                condition_key = f"{condition['name']}:{strength_name}"
                base_key = f"{condition_key}:{diagnosis['name']}"
                scenario_id = f"{group_id}-{condition['name']}-{strength_name}-{diagnosis['name']}-{repair['name']}"
                scenarios.append({
                    'scenario_id': scenario_id, 'group_id': group_id, 'split_role': assignments[i],
                    'mechanism': condition['mechanism'], 'magnitude': magnitude,
                    'features': condition['features'], 'interaction_pairs': condition['interaction_pairs'],
                    'diagnosis_condition': diagnosis['name'], 'diagnosed_rule': diagnosis,
                    'repair_condition': repair['name'], 'label_flip_fraction': repair['label_flip_fraction'],
                    'is_no_drift_control': control,
                    'shared_base_package_id': f'{group_id}:{base_key}',
                    'drift_seeds': {role: component_seed(f'drift:{condition_key}:{role}')
                                    for role in ('repair', 'release_validation', 'sealed_future')},
                    'generator_seed': component_seed(f'generator:{base_key}'),
                    'c1_mock_seed': component_seed(f'c1:{condition_key}'),
                    'c2_bootstrap_seed': component_seed(f'c2:{diagnosis["name"]}'),
                    'corruption_seed': component_seed(f'corruption:{base_key}'),
                })
            for condition in protocol['drift_conditions']:
                for strength_name, magnitude in protocol['injection_strengths'].items():
                    for diagnosis in protocol['diagnosis_conditions']:
                        for repair in protocol['repair_conditions']:
                            add(condition, strength_name, magnitude, diagnosis, repair)
            control = protocol['no_drift_controls']
            if control['per_replication_group'] != 1:
                raise ValueError('v1 manifest implements exactly one no-drift control per group.')
            diagnosis = next(d for d in protocol['diagnosis_conditions'] if d['name'] == control['diagnosis_condition'])
            repair = next(r for r in protocol['repair_conditions'] if r['name'] == control['repair_condition'])
            add({'name': 'none', 'mechanism': 'none', 'features': [], 'interaction_pairs': []},
                'zero', 0., diagnosis, repair, True)
    if len({s['scenario_id'] for s in scenarios}) != len(scenarios):
        raise ValueError('Scenario ID collision.')
    return {'manifest_schema_version': '1.0', 'protocol_version': protocol['protocol_version'],
            'groups': groups, 'scenarios': scenarios,
            'experiments_executed': 0, 'production_approval_export': False}
