"""Replay saved policy evidence and separately verify labelled synthetic controller branches."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from _common import C4_ROOT, checked_bytes, checked_json, parser, require, resolve_data_root, save_json, sha, utc_now, git_context
from _controller_artifacts import VERSION, SNAPSHOT_HASH, POLICY_HASH, controller_directory, load_controller, source_hashes
from c4.conformal.benchmark_n2 import JointCalibration
from c4.risk.benchmark_controller import FrozenPilotController
from c4.risk.benchmark_policy import GATES
from c4.risk.benchmark_n3 import TARGETS


class SyntheticAcceptanceForecaster:
    """Only for explicitly labelled acceptance fixtures, never research predictions."""
    def __init__(self, columns, quantiles):
        self.input_columns = list(columns); self.quantiles = quantiles; self.calls = 0
    def predict(self, frame):
        self.calls += 1
        return pd.DataFrame({t+'__'+label: [self.quantiles[t][n]]
            for t in TARGETS for n, label in enumerate(('lower','median','upper'))}, index=frame.index)


def acceptance_results(definition, measured_policy):
    require(definition['research_observations'] is False and definition['empirical_auto_deployment_evidence'] is False,
            'Acceptance fixtures must be labelled synthetic.')
    features = {**definition['evidence_features'], **{f'fixture_field_{n:02d}': 0. for n in range(71)}}
    columns = list(features); results = []
    for case in definition['fixtures']:
        policy = measured_policy if case['policy_source'] == 'measured_fallback' else case['policy']
        family = policy['kind'] if policy['kind'] != 'review_only' else 'adaptive'
        chosen = {'global': {'policy_id':'fixture-review-only','kind':'review_only'},
                  'adaptive': {'policy_id':'fixture-review-only','kind':'review_only'}}
        chosen[family] = policy
        n = case['calibration_groups']; fallback = n == 17
        fit = JointCalibration(.05, 'replication_group_max_over_admitted_variants_and_three_targets',
                               n, 18 if fallback else 37, case['adjustment'], fallback, not fallback, not fallback)
        models = {m: SyntheticAcceptanceForecaster(columns, case['quantiles']) for m in ('lightgbm','linear')}
        controller = FrozenPilotController(models, {m:fit for m in models}, {m:chosen for m in models},
                                           columns, case['unsupported_contexts'])
        gates = dict.fromkeys(GATES, True)
        if 'failed_gate' in case: gates[case['failed_gate']] = False
        actual = controller.decide(features, gates, family=family, evidence_id=case['fixture_id'])
        require(actual['decision'] == case['expected_decision'], 'Acceptance fixture failed: '+case['fixture_id'])
        require(actual['production_release_authorized'] is False, 'Fixture must not authorize production release.')
        results.append({'fixture_id':case['fixture_id'], 'expected_decision':case['expected_decision'],
            'actual_decision':actual['decision'], 'policy_source':case['policy_source'], 'passed':True,
            'empirical_deployment_evidence':False, 'synthetic_forecaster_calls':models['lightgbm'].calls, 'result':actual})
    return results


def main():
    p = parser(__doc__); p.add_argument('--save', action='store_true'); args = p.parse_args()
    root = resolve_data_root(args.data_root); controller, binding, freeze_hash = load_controller(root)
    snapshot = root / 'c4_artifacts/meta_datasets/adult' / VERSION / 'policy-all12-audit-v1'
    md, digest = checked_json(snapshot); require(digest == SNAPSHOT_HASH, 'Policy snapshot differs.')
    def table(filename):
        return pd.read_csv(io.BytesIO(checked_bytes(snapshot / filename, md['files'][filename])), float_precision='round_trip')
    features = table('features.csv').set_index('scenario_id')
    policy_dir = root / 'c4_artifacts/policies/adult' / VERSION / 'policy-fit-v1'
    pmd, phash = checked_json(policy_dir); require(phash == POLICY_HASH, 'Measured policy selection differs.')
    expected = pd.read_csv(io.BytesIO(checked_bytes(policy_dir / 'selected_decisions.csv', pmd['files']['selected_decisions.csv'])))
    require(len(expected) == 176 and not expected.duplicated(['method','family','scenario_id']).any(), 'Saved replay decision inventory differs.')
    replay = []; counts = {}
    for sid, row in features.iterrows():
        run = root / 'c4_artifacts/benchmark_runs/adult' / VERSION / sid / 'pre_release_freeze-v1'
        fixed, fixed_hash = checked_json(run, 'n1_forecasts')
        signals, signal_hash = checked_json(run, 'policy_forecasts')
        snap, _ = checked_json(run, 'freeze')
        require(fixed_hash == snap['n1_forecasts_sha256'] and signal_hash == snap['policy_forecasts_sha256'], 'Replay provenance differs.')
        for method in ('lightgbm','linear'):
            for family in ('global','adaptive'):
                result = controller.decide(row.to_dict(), signals['explicit_hard_gates'], method=method, family=family, evidence_id=sid)
                e = expected.loc[expected.scenario_id.eq(sid) & expected.method.eq(method) & expected.family.eq(family)]
                require(len(e) == 1 and result['decision'] == e.iloc[0].decision, 'Controller replay decision differs.')
                if result['N1_executed']:
                    require(set(result['n1_forecasts']) == set(fixed['forecasts'][method]) and
                            all(abs(float(v)-float(fixed['forecasts'][method][k])) <= 1e-12 for k,v in result['n1_forecasts'].items()),
                            'Reloaded N1 forecast differs from pre-future evidence.')
                    require(all(np.allclose(result['n2_intervals'][t], signals['intervals'][method][t], rtol=0, atol=1e-12) for t in TARGETS),
                            'Reloaded N2 interval differs.')
                key = method+'/'+family
                counts.setdefault(key, {'AUTO_DEPLOY':0, 'HUMAN_REVIEW':0, 'ESCALATE':0})[result['decision']] += 1
                replay.append({'scenario_id':sid,'method':method,'family':family,'decision':result['decision'],
                               'reason':result['reason'],'matched_saved_decision':True})
    require(len(replay) == 176, 'Controller replay incomplete.')
    fixture_path = C4_ROOT / 'configs/controller_acceptance_fixtures_v1.json'
    definition = json.loads(fixture_path.read_bytes())
    fixtures = acceptance_results(definition, binding['selected_policies']['lightgbm']['adaptive'])
    print('FROZEN CONTROLLER RELOAD / 176 SAVED POLICY DECISION REPLAYS: PASS')
    for key, value in counts.items(): print('MEASURED POLICY REPLAY:', key, value)
    print('SYNTHETIC ACCEPTANCE FIXTURES:', len(fixtures), 'PASS; NOT EMPIRICAL AUTO_DEPLOY EVIDENCE')
    for result in fixtures: print('FIXTURE:', result['fixture_id'], '->', result['actual_decision'])
    print('MEASURED AUTO_DEPLOY ENABLED: False; N1/N2 REFITTED: False; OUTER TEST ACCESSED: False')
    if not args.save:
        print('PLAN ONLY. Use --save to persist verification evidence.'); return
    output = controller_directory(root) / 'verification-v1'
    require(not output.exists(), 'Controller verification exists; preserve it.')
    # Revalidate pinned code/models before saving verification evidence.
    load_controller(root)
    output.mkdir(parents=True, exist_ok=False); files = {}
    data = pd.DataFrame(replay).to_csv(index=False, lineterminator='\n').encode()
    (output / 'policy_replay.csv').write_bytes(data); files['policy_replay.csv'] = sha(data)
    fixtures_hash = save_json(output / 'synthetic_acceptance_results.json', {'purpose':'synthetic_controller_acceptance_checks',
        'controller_freeze_sha256':freeze_hash, 'fixture_definition_sha256':sha(fixture_path.read_bytes()),
        'research_observations':False, 'empirical_auto_deployment_evidence':False, 'fixtures':fixtures})
    files['synthetic_acceptance_results.json'] = fixtures_hash
    note = '\n'.join(['# C4 pilot controller verification', '',
        '176 policy-evidence decision replays matched saved decisions; this is reproducibility checking, not independent evaluation.',
        *[f'{key}: {value}' for key,value in counts.items()], '',
        'Seven labelled synthetic acceptance fixtures verify the actual controller branches.',
        'The AUTO_DEPLOY positive control uses a synthetic forecaster and explicit test policy.',
        'The measured policy remains review-only; no empirical AUTO_DEPLOY or live production release is established.',
        'Raw C1/C2/C3 service integration and the untouched outer-test evaluation remain separate work.', ''])
    (output / 'C4_CONTROLLER_VERIFICATION.md').write_bytes(note.encode()); files['C4_CONTROLLER_VERIFICATION.md'] = sha(note.encode())
    digest = save_json(output / 'metadata.json', {'purpose':'saved_pilot_controller_verification',
        'created_at_utc':utc_now(), **git_context(), 'controller_freeze_sha256':freeze_hash,
        'policy_metadata_sha256':POLICY_HASH, 'replayed_decisions':176, 'synthetic_fixtures':len(fixtures),
        'measured_policy_replay_counts':counts, 'files':files, 'source_sha256':source_hashes(),
        'independent_evaluation':False, 'production_release_authorized':False})
    for filename, expected_hash in files.items(): checked_bytes(output / filename, expected_hash)
    print('OUTPUT:', output); print('METADATA SHA256:', digest)
    print('BASELINE / ABLATION / STATISTICAL PROTOCOL FREEZES REQUIRED BEFORE OUTER-TEST EXECUTION')


if __name__ == '__main__': main()
