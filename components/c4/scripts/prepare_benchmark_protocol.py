"""Review or freeze a full simulation protocol; this script never runs experiments."""
import json

from _common import parser, resolve_data_root, checked_json, checked_bytes, manifest_root, require, save_json, sha, git_context, utc_now, C4_ROOT, REPO_ROOT
from _benchmark_manifest import build_manifest
from c4.models import load_champion
from c4.data.split_manifest import load_adult_split_manifest
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import reference_quantile_slice
import io
import pandas as pd


def main():
    p = parser(__doc__)
    p.add_argument('--freeze', action='store_true', help='Persist immutable protocol and planned manifest; no execution.')
    args = p.parse_args()
    config_path = C4_ROOT / 'configs' / 'benchmark_protocol.json'
    contents = config_path.read_bytes()
    protocol = json.loads(contents)
    require(protocol['status'] == 'proposed_for_review', 'Use the reviewed source specification.')
    require(protocol['full_execution_enabled'] is False and protocol['production_approval_export'] is False,
            'This preparer never authorizes execution or production approval.')
    manifest = build_manifest(protocol)
    for scenario in manifest['scenarios']:
        ScenarioConfig(mechanism=scenario['mechanism'], magnitude=scenario['magnitude'],
                       features=tuple(scenario['features']),
                       interaction_pairs=tuple(tuple(p) for p in scenario['interaction_pairs']),
                       seed=scenario['generator_seed'])
    roles = ('forecast_train', 'conformal_calibration', 'policy_calibration', 'outer_test')
    counts = {role: sum(group['split_role'] == role for group in manifest['groups']) for role in roles}
    controls = sum(s['is_no_drift_control'] for s in manifest['scenarios'])
    print('PROTOCOL SHA256:', sha(contents))
    print('PLANNED GROUPS:', len(manifest['groups']))
    print('GROUP FOLDS:', counts)
    print('DRIFT SCENARIOS:', len(manifest['scenarios']) - controls)
    print('NO-DRIFT CONTROLS:', controls)
    print('TOTAL PLANNED SCENARIOS:', len(manifest['scenarios']))
    print('EXPERIMENTS EXECUTED: 0')
    print('STATUS:', 'FREEZE_REQUESTED' if args.freeze else 'REVIEW_ONLY')
    if not args.freeze:
        print('Review scope and assumptions before using --freeze. No artifacts were written.')
        return
    root = resolve_data_root(args.data_root)
    output = root / 'c4_artifacts' / 'benchmark_protocols' / 'adult' / protocol['protocol_version']
    if output.exists():
        raise FileExistsError(f'Frozen benchmark protocol exists: {output}')
    md = manifest_root(root, protocol['source_versions']['split_manifest'])
    source_manifest = load_adult_split_manifest(md, data_root=root)
    _, champion = load_champion(root / 'c4_artifacts' / 'champion' / 'adult' / protocol['source_versions']['champion'])
    reference_bytes = checked_bytes(root / 'uci_adult_data' / 'reference.csv', champion['reference_csv_sha256'])
    require(champion['reference_csv_sha256'] == source_manifest['source_checksums']['reference_csv_sha256'],
            'Champion and data manifest reference differ.')
    reference = pd.read_csv(io.BytesIO(reference_bytes))
    rules = [protocol['true_slice'], *protocol['diagnosis_conditions']]
    for rule in rules:
        derived = reference_quantile_slice(reference, name=rule['name'], feature=rule['feature'],
                                           quantile=rule['reference_quantile'], operator=rule['operator'])
        require(abs(derived.value - rule['resolved_threshold']) <= 1e-12, 'Reference-derived slice differs.')
    source_paths = {
        'scenario_adapter': C4_ROOT / 'src/c4/scenarios/adapter.py',
        'tvae_generation': C4_ROOT / 'src/c4/experiments/tvae_generation.py',
        'distance_metric': C4_ROOT / 'scripts/_privacy_distance.py',
        'shared_C1_mock': REPO_ROOT / 'shared/contracts/c1_fake_output.py',
        'C4_contracts': C4_ROOT / 'src/c4/contracts/models.py',
        'manifest_builder': C4_ROOT / 'scripts/_benchmark_manifest.py',
    }
    record = {
        'purpose': 'frozen_controlled_simulation_benchmark_protocol',
        'status': 'frozen_specification_execution_disabled', 'created_at_utc': utc_now(), **git_context(),
        'protocol_sha256': sha(contents), 'split_manifest_sha256': sha((md / 'manifest.json').read_bytes()),
        'champion_model_sha256': champion['model_sha256'],
        'source_implementation_sha256': {key: sha(path.read_bytes()) for key, path in source_paths.items()},
        'group_counts': counts, 'scenarios_planned': len(manifest['scenarios']),
        'experiments_executed': 0, 'full_execution_enabled': False,
        'production_approval_export': False,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'protocol.json').open('xb') as file:
        file.write(contents)
    manifest_hash = save_json(output / 'scenario_manifest.json', manifest)
    record['scenario_manifest_sha256'] = manifest_hash
    freeze_hash = save_json(output / 'freeze.json', record)
    print('FROZEN DIRECTORY:', output)
    print('SCENARIO MANIFEST SHA256:', manifest_hash)
    print('FREEZE SHA256:', freeze_hash)
    print('SPECIFICATION FROZEN; EXECUTION STILL DISABLED')


if __name__ == '__main__':
    main()
