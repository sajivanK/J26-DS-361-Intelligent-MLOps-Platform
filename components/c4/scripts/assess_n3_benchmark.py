"""Freeze guide-based N3 rules and assess stored intervals without reading future labels."""
import io
import json
from pathlib import Path

import pandas as pd

from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _conformal_group import VERSION, verify_n1
from c4.risk.benchmark_n3 import BOUNDARIES, POPULATIONS, TARGETS, assess_safety

N2_HASH = 'd40e55f7934934cf8708b4b77cb7b4a3205f66b1d30540daa31846e682cacc80'


def main():
    p = parser(__doc__)
    p.add_argument('--assessment-version', type=name, default='n3-safety-v1')
    p.add_argument('--freeze-spec', action='store_true')
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    require(not (args.freeze_spec and args.save), 'Freeze specification and save assessment in separate invocations.')
    root = resolve_data_root(args.data_root)
    n1 = verify_n1(root)
    source = root / 'c4_artifacts/conformal/adult' / VERSION / 'n2-joint-v1'
    metadata, digest = checked_json(source)
    require(digest == N2_HASH and metadata['n1_binding'] == n1 and
            metadata['status'] == 'INSUFFICIENT_NONEMPTY_GROUPS_FALLBACK', 'Wrong frozen N2 artifact.')
    for relative, expected in metadata['source_sha256'].items():
        checked_bytes(C4_ROOT / relative, expected)
    # Read only calibration parameters and interval tables, never targets or score tables.
    calibration = json.loads(checked_bytes(source / 'calibration.json', metadata['files']['calibration.json']))
    require(calibration['n1_binding'] == n1 and not calibration['informative_95_percent_intervals'] and
            not calibration['automatic_release_authorized'], 'N2 support declarations differ.')
    tables = {}
    input_hashes = {'calibration.json': metadata['files']['calibration.json']}
    for method in ('lightgbm', 'linear'):
        c = calibration['methods'][method]
        require(c['fallback'] is True and c['auto_deploy_supported'] is False and c['adjustment'] is None
                and c['nonempty_groups'] == 17 and c['rank'] == 18, 'Frozen fallback differs.')
        filename = method + '_intervals.csv'
        input_hashes[filename] = metadata['files'][filename]
        frame = pd.read_csv(io.BytesIO(checked_bytes(source / filename, input_hashes[filename])),
                            index_col=0, float_precision='round_trip')
        require(len(frame) == 64 and frame.index.is_unique and frame.index.name == 'scenario_id', 'Wrong interval inventory.')
        require(set(frame.columns) == {t+'__'+q for t in TARGETS for q in ('lower','median','upper')}, 'Wrong interval schema.')
        for _, row in frame.iterrows():
            assess_safety({t: [row[t+'__lower'], row[t+'__upper']] for t in TARGETS},
                          auto_deploy_supported=False)
        tables[method] = frame
    require(tables['lightgbm'].index.equals(tables['linear'].index), 'Method scenario IDs differ.')
    code_hashes = {str(path.relative_to(C4_ROOT)): sha(path.read_bytes()) for path in
                   (Path(__file__), C4_ROOT / 'src/c4/risk/benchmark_n3.py')}
    specification = {'purpose': 'benchmark_n3_safety_specification', 'schema_version': '1.0',
        'boundary_source': 'C4 Research Guide v1.1 N3 table and algorithm, pages 20-21; separately frozen here',
        'boundaries': BOUNDARIES, 'population_basis': POPULATIONS,
        'population_deviation_from_guide': 'Benchmark primary slice is true injected slice, not diagnosed slice; no diagnosed-slice safety guarantee claimed.',
        'units': 'candidate_minus_champion_accuracy_fraction',
        'safe_rule': 'lower >= boundary', 'unsafe_rule': 'upper < boundary',
        'otherwise': 'UNCERTAIN', 'aggregation': 'any UNSAFE -> ESCALATE; all SAFE and N2 support -> AUTO_DEPLOY signal; otherwise HUMAN_REVIEW',
        'fallback': 'all three [-1,1]; AUTO_DEPLOY unsupported',
        'n2_metadata_sha256': digest, 'n1_binding': n1, 'input_sha256': input_hashes,
        'source_sha256': code_hashes, 'observed_future_labels_as_inputs': False,
        'policy_calibrated': False, 'production_release_authorized': False}
    output = root / 'c4_artifacts/safety/adult' / VERSION / args.assessment_version
    print('N1 / N2 BINDINGS / INTERVAL CHECKSUMS: PASS')
    print('BOUNDARIES: global=0.0; true injected slice=0.0; true slice complement=-0.02')
    if args.freeze_spec:
        require(not output.exists(), 'N3 version already exists; preserve it.')
        output.mkdir(parents=True, exist_ok=False)
        frozen = {'definition': specification, 'created_at_utc': utc_now(), **git_context()}
        spec_hash = save_json(output / 'specification.json', frozen)
        print('SPECIFICATION:', output / 'specification.json')
        print('SPECIFICATION SHA256:', spec_hash)
        print('NO ASSESSMENT SAVED. Next use a separate invocation with --save.')
        return
    if not args.save:
        print('PLAN ONLY. Use --freeze-spec, then --save. No target reads, fitting or future reevaluation.')
        return
    frozen, spec_hash = checked_json(output, 'specification')
    require(frozen['definition'] == specification, 'N3 inputs or implementation changed after specification freeze.')
    require(not (output / 'metadata.json').exists(), 'Assessment already saved; never overwrite.')
    files = {}
    counts = {}
    for method, frame in tables.items():
        records = []
        for scenario_id, row in frame.iterrows():
            intervals = {t: [row[t+'__lower'], row[t+'__upper']] for t in TARGETS}
            result = assess_safety(intervals, auto_deploy_supported=False)
            records.append({'scenario_id': scenario_id, **result})
        require(all(r['release_signal'] == 'HUMAN_REVIEW' for r in records), 'Fallback unexpectedly produced another signal.')
        counts[method] = {'scenarios': len(records), 'HUMAN_REVIEW': len(records), 'AUTO_DEPLOY': 0, 'ESCALATE': 0}
        files[method+'_assessment.json'] = save_json(output / (method+'_assessment.json'),
            {'method': method, 'specification_sha256': spec_hash, 'records': records,
             'evaluation_scope': 'conformal interval assessment only; no independent policy performance evaluation'})
        print('METHOD:', method, 'SCENARIOS: 64; ALL THREE DIMENSIONS UNCERTAIN; HUMAN_REVIEW: 64')
    require(verify_n1(root) == n1, 'N1 changed during assessment.')
    checked_bytes(source / 'metadata.json', digest)
    for filename, expected in input_hashes.items():
        checked_bytes(source / filename, expected)
    for relative, expected in code_hashes.items():
        checked_bytes(C4_ROOT / relative, expected)
    saved_hash = save_json(output / 'metadata.json', {'purpose': 'verified_n3_benchmark_assessment',
        'created_at_utc': utc_now(), **git_context(), 'specification_sha256': spec_hash,
        'n2_metadata_sha256': digest, 'n1_binding': n1, 'files': files, 'counts': counts,
        'source_sha256': code_hashes, 'observed_future_labels_read': False,
        'policy_calibrated': False, 'outer_test_evaluated': False, 'release_authorized': False})
    print('OUTPUT:', output)
    print('METADATA SHA256:', saved_hash)
    print('N3 SIGNALS ONLY; GLOBAL / N4 POLICY PENDING; RELEASE AUTHORIZED: False')


if __name__ == '__main__':
    main()
