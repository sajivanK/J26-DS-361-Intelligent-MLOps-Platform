"""Select global/N4 policies from the audited policy fold under the frozen empirical rule."""
import io
from pathlib import Path
import pandas as pd
from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _policy_pipeline import VERSION, POLICY_SPEC_HASH, verify_policy_spec, verify_context_support
from c4.risk.benchmark_policy import candidate_policies
from c4.risk.benchmark_n3 import TARGETS
from c4.risk.benchmark_policy_selection import select_policy, policy_statistics


def main():
    p = parser(__doc__)
    p.add_argument('--save', action='store_true')
    p.add_argument('--snapshot-version', type=name, default='policy-all12-audit-v1')
    p.add_argument('--policy-version', type=name, default='policy-fit-v1')
    args = p.parse_args(); root = resolve_data_root(args.data_root)
    frozen, specdir = verify_policy_spec(root)
    definition, _ = checked_json(specdir, 'definition')
    candidates, _ = checked_json(specdir, 'candidates')
    require(candidates == candidate_policies(), 'Frozen candidate inventory differs.')
    support, support_hash = verify_context_support(root)
    directory = root / 'c4_artifacts/meta_datasets/adult' / VERSION / args.snapshot_version
    md, md_hash = checked_json(directory)
    require(md['purpose'] == 'verified_combined_policy_snapshot' and md['rows'] == 44
            and md['fold'] == 'policy_calibration' and md['nonempty_groups'] == 11
            and md['admission_cases'] == 444 and md['blocked_cases'] == 400
            and md['policy_specification_sha256'] == POLICY_SPEC_HASH
            and md['policy_context_support_sha256'] == support_hash, 'Wrong policy snapshot.')
    checked_bytes(C4_ROOT / 'scripts/audit_policy_snapshot.py', md['source_code_sha256'])
    for path, digest in md['source_links'].items(): checked_bytes(path, digest)
    for filename, digest in md['files'].items(): checked_bytes(directory / filename, digest)
    def table(filename):
        return pd.read_csv(io.BytesIO(checked_bytes(directory / filename, md['files'][filename])), float_precision='round_trip')
    targets = table('targets.csv').set_index('scenario_id')
    index = table('scenario_index.csv')
    require(targets.index.tolist() == index.scenario_id.tolist() and targets.columns.tolist() == list(TARGETS), 'Target order/schema differs.')
    signals = table('candidate_signals.csv')
    require(len(signals) == 44 * 24 * 2 and set(signals.method) == set(definition['methods'])
            and not signals.duplicated(['method', 'policy_id', 'scenario_id']).any(), 'Policy signal inventory differs.')
    membership = index.set_index('scenario_id').group_id.to_dict()
    require(signals.split_role.eq('policy_calibration').all() and
            all(membership.get(r.scenario_id) == r.group_id for r in signals.itertuples()), 'Signal group/fold differs.')
    target_rows = targets.to_dict('index'); ix = index.to_dict('records')
    selections, candidate_summary, per_group, selected_rows = {}, [], [], []
    for method in definition['methods']:
        selections[method] = {}
        method_signals = signals.loc[signals.method.eq(method)]
        require(set(method_signals.policy_id) == {p['policy_id'] for ps in candidates.values() for p in ps}, 'Method candidates differ.')
        reference = method_signals.loc[method_signals.policy_id.eq('global-00')].set_index('scenario_id').decision.to_dict()
        for family, policies in candidates.items():
            decisions = {}
            for policy in policies:
                subset = method_signals.loc[method_signals.policy_id.eq(policy['policy_id'])]
                require(subset.family.eq(family).all(), 'Policy family differs.')
                decisions[policy['policy_id']] = subset.set_index('scenario_id').decision.to_dict()
                require({sid for sid, d in decisions[policy['policy_id']].items() if d == 'ESCALATE'} ==
                        {sid for sid, d in reference.items() if d == 'ESCALATE'}, 'Mandatory escalation differs across policies.')
            result = select_policy(policies, decisions, target_rows, ix, definition['selection'])
            if result['selected_policy']['kind'] == 'review_only':
                final = {sid: 'ESCALATE' if d == 'ESCALATE' else 'HUMAN_REVIEW' for sid, d in reference.items()}
            else:
                final = decisions[result['selected_policy']['policy_id']]
            final_metrics = policy_statistics(final, target_rows, ix)
            result['selected_or_fallback_metrics'] = final_metrics
            selections[method][family] = result
            print('METHOD:', method, 'FAMILY:', family, 'STATUS:', result['status'])
            print('SELECTED:', result['selected_policy'], 'DECISIONS:', final_metrics['decision_counts'])
            print('GROUP AUTOMATION:', final_metrics['equal_group_automation'],
                  'CONDITIONAL UNSAFE RATE:', final_metrics['equal_group_conditional_unsafe_rate'])
            for sid, decision in final.items():
                selected_rows.append({'scenario_id': sid, 'group_id': membership[sid], 'split_role': 'policy_calibration',
                    'method': method, 'family': family, 'selected_policy_id': result['selected_policy']['policy_id'], 'decision': decision})
            for entry in result['candidates']:
                metrics = entry['metrics']
                candidate_summary.append({'method': method, 'family': family, 'policy_id': entry['policy']['policy_id'],
                    'eligible': entry['eligible'], 'equal_group_automation': metrics['equal_group_automation'],
                    'equal_group_conditional_unsafe_rate': metrics['equal_group_conditional_unsafe_rate'],
                    'auto_deploy_groups': metrics['auto_deploy_groups'], 'unsafe_auto_deploy_count': metrics['unsafe_auto_deploy_count'],
                    **metrics['decision_counts']})
                per_group.extend({'method': method, 'family': family, 'policy_id': entry['policy']['policy_id'], **r} for r in metrics['per_group'])
    print('EMPIRICAL POLICY SELECTION ONLY; NO FORMAL 95% CONDITIONAL-RISK GUARANTEE')
    print('N1/N2 REFITTING / OUTER-TEST / PRODUCTION RELEASE: False')
    if not args.save:
        print('PLAN ONLY. Use --save to freeze selections and the review-only fallback where required.'); return
    git = git_context(); require(not git['working_tree_dirty'], 'Commit the policy audit/selection implementation before saving.')
    output = root / 'c4_artifacts/policies/adult' / VERSION / args.policy_version
    require(not output.exists(), 'Policy selection output exists; preserve it.')
    source_paths = [Path(__file__), C4_ROOT / 'scripts/audit_policy_snapshot.py',
        C4_ROOT / 'src/c4/risk/benchmark_policy_selection.py']
    sources = {str(p.relative_to(C4_ROOT)): sha(p.read_bytes()) for p in source_paths}
    verify_policy_spec(root); verify_context_support(root)
    for path, digest in md['source_links'].items(): checked_bytes(path, digest)
    for filename, digest in md['files'].items(): checked_bytes(directory / filename, digest)
    output.mkdir(parents=True, exist_ok=False); files = {}
    for filename, rows in [('candidate_metrics.csv', candidate_summary), ('candidate_group_metrics.csv', per_group),
                           ('selected_decisions.csv', selected_rows)]:
        data = pd.DataFrame(rows).to_csv(index=False, lineterminator='\n').encode()
        (output / filename).write_bytes(data); files[filename] = sha(data)
    selected_hash = save_json(output / 'selection.json', {'purpose': 'frozen_empirical_global_and_adaptive_policy_selection',
        'policy_specification_sha256': POLICY_SPEC_HASH, 'policy_snapshot_metadata_sha256': md_hash,
        'policy_context_support_sha256': support_hash, 'unsupported_contexts': support['unsupported_contexts'],
        'n1_binding': frozen['n1_binding'], 'n2_metadata_sha256': frozen['n2_metadata_sha256'],
        'n3_metadata_sha256': frozen['n3_metadata_sha256'], 'primary_method': definition['primary_method'],
        'methods': selections, 'selection_rule': definition['selection'],
        'evaluation': 'policy_calibration_in_sample_selection_not_outer_test',
        'formal_conditional_risk_guarantee_established': False, 'release_authorized': False})
    files['selection.json'] = selected_hash
    note = '\n'.join(['# C4 global and adaptive policy selection', '',
        '44 admitted scenarios in 11 nonempty policy groups; 12 assigned groups, 400 blocked cases retained.',
        'Selection uses the previously frozen four global and twenty monotone adaptive candidates.',
        'Three deployment groups and an empirical group-averaged conditional unsafe rate <= 0.05 are required.',
        'Zero deployments produce an undefined conditional unsafe rate, never evidence of zero risk.',
        'No eligible candidate yields a review-only fallback that preserves mandatory escalation.',
        'These are policy-calibration selection results, not independent test performance or a formal risk guarantee.', '',
        *[f"{m} / {family}: {r['status']}; {r['selected_or_fallback_metrics']['decision_counts']}"
          for m, families in selections.items() for family, r in families.items()], ''])
    (output / 'C4_POLICY_SELECTION_EVIDENCE.md').write_bytes(note.encode()); files['C4_POLICY_SELECTION_EVIDENCE.md'] = sha(note.encode())
    digest = save_json(output / 'metadata.json', {'purpose': 'saved_frozen_policy_selection', 'created_at_utc': utc_now(), **git,
        'policy_specification_sha256': POLICY_SPEC_HASH, 'policy_snapshot_metadata_sha256': md_hash,
        'policy_snapshot_directory': str(directory), 'policy_context_support_sha256': support_hash,
        'n1_binding': frozen['n1_binding'], 'n2_metadata_sha256': frozen['n2_metadata_sha256'],
        'n3_metadata_sha256': frozen['n3_metadata_sha256'], 'files': files, 'source_sha256': sources,
        'N1_refitted': False, 'N2_refitted': False, 'outer_test_evaluated': False, 'release_authorized': False})
    for filename, expected in files.items(): checked_bytes(output / filename, expected)
    require(checked_json(output, 'selection')[0]['methods'] == selections, 'Selection JSON reload differs.')
    print('SELECTION / CHECKSUM / JSON RELOAD: PASS')
    print('OUTPUT:', output); print('METADATA SHA256:', digest)
    print('FINAL CONTROLLER / ACCEPTANCE FIXTURES / FROZEN EVALUATION PROTOCOL: PENDING')


if __name__ == '__main__': main()
