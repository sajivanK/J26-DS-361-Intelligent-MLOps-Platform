"""Frozen Adult pilot inference: hard gates -> N1 -> N2 -> N3 -> global/N4 policy.

Consumes a decision-time evidence vector already built by the candidate pipeline.
Returns a research signal; this module does not train or deploy a model.
"""
from copy import deepcopy
from numbers import Real
import math
import pandas as pd
from c4.conformal.benchmark_n2 import apply_joint_calibration
from c4.risk.benchmark_n3 import TARGETS
from c4.risk.benchmark_policy import GATES, evidence_context, policy_decision


def require(condition, message):
    if not condition:
        raise ValueError(message)


class FrozenPilotController:
    def __init__(self, models, calibrations, selected_policies, raw_columns, unsupported_contexts=()):
        require(set(models) == set(calibrations) == set(selected_policies) == {'lightgbm', 'linear'}, 'Two frozen methods required.')
        require(len(raw_columns) == len(set(raw_columns)) == 75, 'Exactly 75 named decision-time features required.')
        require(all(set(families) == {'global', 'adaptive'} for families in selected_policies.values()), 'Two policy families required.')
        self.models = dict(models)
        self.calibrations = dict(calibrations)
        self.policies = deepcopy(selected_policies)
        self.raw_columns = tuple(raw_columns)
        self.unsupported_contexts = tuple(unsupported_contexts)
        for method, model in self.models.items():
            require(model.input_columns == list(self.raw_columns), 'Frozen model feature schema differs.')
            for family, policy in self.policies[method].items():
                require(policy['kind'] in {family, 'review_only'}, 'Selected policy family differs.')

    def decide(self, features, gates, *, method='lightgbm', family='adaptive', evidence_id='decision'):
        require(method in self.models and family in {'global', 'adaptive'}, 'Unknown method or policy family.')
        require(isinstance(gates, dict), 'Explicit hard-gate results required.')
        policy = self.policies[method][family]
        base = {'scope': 'adult_accuracy_margin_pilot_simulated_upstream', 'method': method, 'family': family,
            'evidence_id': evidence_id, 'selected_policy_id': policy['policy_id'], 'selected_policy_kind': policy['kind'],
            'population_basis': 'global / true-injected-slice target / true-slice-complement target; diagnosed slice is evidence',
            'future_targets_used': False, 'production_release_authorized': False}
        failed = [g for g in GATES if gates.get(g) is not True]
        if failed:
            return {**base, 'decision': 'ESCALATE', 'reason': 'mandatory_gate_failed_or_unknown',
                'failed_gates': failed, 'N1_executed': False, 'N2_executed': False, 'release_authorized': False}
        require(isinstance(features, dict) and set(features) == set(self.raw_columns), 'Decision-time feature schema differs; no targets or extra fields allowed.')
        require(all(isinstance(v, Real) and math.isfinite(v) for v in features.values()), 'Decision-time features must be numeric and finite.')
        context = evidence_context(features)
        frame = pd.DataFrame([[float(features[c]) for c in self.raw_columns]], columns=self.raw_columns,
                             index=pd.Index([evidence_id], name='scenario_id'))
        forecasts = self.models[method].predict(frame)
        require(len(forecasts) == 1 and forecasts.index.equals(frame.index), 'N1 changed evidence-row identity.')
        calibration = self.calibrations[method]
        calibrated = apply_joint_calibration(calibration, forecasts).iloc[0]
        intervals = {t: [float(calibrated[t+'__lower']), float(calibrated[t+'__upper'])] for t in TARGETS}
        result = policy_decision(intervals, features, gates, n2_supported=calibration.auto_deploy_supported,
                                 policy=policy, unsupported_contexts=self.unsupported_contexts)
        raw = {k: bool(v) if k.endswith('__crossing_corrected') else float(v)
               for k, v in forecasts.iloc[0].to_dict().items()}
        return {**base, **result, 'N1_executed': True, 'N2_executed': True, 'n1_forecasts': raw,
            'n2_intervals': intervals, 'evidence_context': context,
            'auto_deployment_policy_enabled': policy['kind'] != 'review_only' and calibration.auto_deploy_supported}
