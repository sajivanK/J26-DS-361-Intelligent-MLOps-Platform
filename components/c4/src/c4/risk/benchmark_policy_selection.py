"""Apply the pre-outcome policy specification using empirical group statistics."""
import math
from collections import defaultdict
from fractions import Fraction
from c4.risk.benchmark_n3 import BOUNDARIES
from c4.risk.benchmark_policy import CONTEXTS

DECISIONS = ('AUTO_DEPLOY', 'HUMAN_REVIEW', 'ESCALATE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_index(rows):
    require(bool(rows), 'Nonempty policy scenario inventory required.')
    ids = [r['scenario_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate scenario IDs.')
    require(all(r['split_role'] == 'policy_calibration' for r in rows), 'Only policy-calibration rows allowed.')
    require(all(isinstance(r['group_id'], str) and r['group_id'] for r in rows), 'Valid group IDs required.')
    return {r['scenario_id']: r['group_id'] for r in rows}


def policy_statistics(decisions, targets, index):
    """Risk is undefined when nobody is deployed; empty groups have no invented labels."""
    membership = validate_index(index)
    require(set(decisions) == set(targets) == set(membership), 'Decision/target/scenario inventory differs.')
    grouped = defaultdict(list)
    counts = dict.fromkeys(DECISIONS, 0)
    unsafe_auto = 0
    for sid, gid in membership.items():
        decision = decisions[sid]
        require(decision in DECISIONS, 'Unknown decision.')
        values = targets[sid]
        require(set(values) == set(BOUNDARIES), 'Exactly three primary margins required.')
        require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                    and -1 <= v <= 1 for v in values.values()), 'Invalid future margins.')
        unsafe = any(values[t] < boundary for t, boundary in BOUNDARIES.items())
        grouped[gid].append((decision, unsafe))
        counts[decision] += 1
        unsafe_auto += int(decision == 'AUTO_DEPLOY' and unsafe)
    per_group = []
    for gid, entries in sorted(grouped.items()):
        deployed = [unsafe for decision, unsafe in entries if decision == 'AUTO_DEPLOY']
        per_group.append({'group_id': gid, 'admitted_scenarios': len(entries),
            'auto_deploy_count': len(deployed), 'unsafe_auto_deploy_count': sum(deployed),
            'automation': len(deployed) / len(entries),
            'conditional_unsafe_rate': sum(deployed) / len(deployed) if deployed else None})
    active = [g for g in per_group if g['auto_deploy_count']]
    return {'scenarios': len(membership), 'nonempty_groups': len(grouped), 'decision_counts': counts,
        'auto_deploy_groups': len(active), 'unsafe_auto_deploy_count': unsafe_auto,
        'equal_group_automation': float(sum(Fraction(g['auto_deploy_count'], g['admitted_scenarios']) for g in per_group) / len(per_group)),
        'equal_group_conditional_unsafe_rate':
            float(sum(Fraction(g['unsafe_auto_deploy_count'], g['auto_deploy_count']) for g in active) / len(active)) if active else None,
        'scenario_automation': counts['AUTO_DEPLOY'] / len(membership),
        'scenario_conditional_unsafe_rate': unsafe_auto / counts['AUTO_DEPLOY'] if counts['AUTO_DEPLOY'] else None,
        'scenario_review_burden': counts['HUMAN_REVIEW'] / len(membership),
        'per_group': per_group,
        'interpretation': 'policy-calibration empirical statistics; not independent validation or a formal risk guarantee'}


def buffer_tuple(policy):
    if policy['kind'] == 'global':
        return (policy['buffer'],) * 3
    require(policy['kind'] == 'adaptive', 'Unknown candidate policy kind.')
    return tuple(policy['buffers'][c] for c in CONTEXTS)


def select_policy(candidates, decisions_by_policy, targets, index, selection):
    """Select exactly by frozen support, risk, objective and tie-break rules."""
    require(selection['minimum_auto_deploy_groups'] == 3 and
            selection['empirical_conditional_unsafe_ceiling'] == .05, 'Frozen selection criteria differ.')
    ids = [p['policy_id'] for p in candidates]
    require(bool(ids) and len(ids) == len(set(ids)) and set(ids) == set(decisions_by_policy),
            'Candidate policy inventory differs.')
    results = []
    for policy in candidates:
        metrics = policy_statistics(decisions_by_policy[policy['policy_id']], targets, index)
        risk = metrics['equal_group_conditional_unsafe_rate']
        eligible = metrics['auto_deploy_groups'] >= 3 and risk is not None and risk <= .05
        results.append({'policy': policy, 'eligible': eligible, 'metrics': metrics})
    eligible = [r for r in results if r['eligible']]
    if not eligible:
        return {'status': 'UNSUPPORTED_SELECTION_REVIEW_ONLY',
            'selected_policy': {'policy_id': 'unsupported-review-only', 'kind': 'review_only'},
            'selected_candidate_metrics': None, 'candidates': results,
            'formal_conditional_risk_guarantee_established': False, 'release_authorized': False}
    best = max(eligible, key=lambda r: (r['metrics']['equal_group_automation'],
        -r['metrics']['equal_group_conditional_unsafe_rate'], buffer_tuple(r['policy'])))
    return {'status': 'EMPIRICALLY_SELECTED', 'selected_policy': best['policy'],
        'selected_candidate_metrics': best['metrics'], 'candidates': results,
        'formal_conditional_risk_guarantee_established': False, 'release_authorized': False}
