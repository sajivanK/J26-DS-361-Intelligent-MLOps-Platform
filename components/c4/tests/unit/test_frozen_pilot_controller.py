"""Technical acceptance checks are separate from empirical research decisions."""
import ast
import json
import math
from pathlib import Path
import pandas as pd
import pytest
from c4.conformal.benchmark_n2 import JointCalibration
from c4.risk.benchmark_controller import FrozenPilotController
from c4.risk.benchmark_policy import GATES
from c4.risk.benchmark_n3 import TARGETS

BASE = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition: raise ValueError(message)


def acceptance_helpers():
    path = BASE / 'scripts/verify_c4_controller.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in {'acceptance_results', 'SyntheticAcceptanceForecaster'}]
    namespace = {'pd':pd, 'TARGETS':TARGETS, 'GATES':GATES, 'require':require,
                 'JointCalibration':JointCalibration, 'FrozenPilotController':FrozenPilotController}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def fixture_definition():
    return json.loads((BASE / 'configs/controller_acceptance_fixtures_v1.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('number', range(7))
def test_predefined_acceptance_fixture_through_actual_controller(number):
    definition = fixture_definition(); definition['fixtures'] = [definition['fixtures'][number]]
    results = acceptance_helpers()['acceptance_results'](definition, {'policy_id':'measured-review-only','kind':'review_only'})
    assert results[0]['passed'] and results[0]['empirical_deployment_evidence'] is False
    assert results[0]['result']['production_release_authorized'] is False
    if results[0]['fixture_id'] == 'hard-gate-failure':
        assert results[0]['synthetic_forecaster_calls'] == 0


def controller_fixture(*, review_only=False):
    definition = fixture_definition(); case = definition['fixtures'][0]
    features = {**definition['evidence_features'], **{f'fixture_field_{n:02d}':0. for n in range(71)}}
    cls = acceptance_helpers()['SyntheticAcceptanceForecaster']
    models = {m:cls(list(features), case['quantiles']) for m in ('lightgbm','linear')}
    fit = JointCalibration(.05, 'replication_group_max_over_admitted_variants_and_three_targets',37,37,.001,False,True,True)
    policies = {m:{'global':{'policy_id':'review-only','kind':'review_only'} if review_only else case['policy'],
                   'adaptive':{'policy_id':'review-only','kind':'review_only'}} for m in models}
    controller = FrozenPilotController(models,{m:fit for m in models},policies,list(features))
    return controller, features, dict.fromkeys(GATES,True), models


@pytest.mark.parametrize('gate', GATES)
def test_each_failed_gate_escalates_without_model_inference(gate):
    controller, _, gates, models = controller_fixture(); gates[gate] = False
    result = controller.decide(None,gates)
    assert result['decision'] == 'ESCALATE' and result['N1_executed'] is False
    assert models['lightgbm'].calls == 0


@pytest.mark.parametrize('gate', GATES)
def test_each_missing_gate_escalates(gate):
    controller, features, gates, _ = controller_fixture(); gates.pop(gate)
    assert controller.decide(features,gates)['decision'] == 'ESCALATE'


def test_string_pass_is_not_explicit_boolean_gate():
    controller, features, gates, _ = controller_fixture(); gates['privacy_check'] = 'PASS'
    assert controller.decide(features,gates)['decision'] == 'ESCALATE'


def test_measured_review_only_policy_cannot_deploy_even_with_positive_forecasts():
    controller, features, gates, _ = controller_fixture(review_only=True)
    result = controller.decide(features,gates,family='global')
    assert result['decision'] == 'HUMAN_REVIEW'
    assert result['reason'] == 'policy_selection_unsupported'
    assert result['auto_deployment_policy_enabled'] is False


def test_caller_cannot_override_selected_policy_per_request():
    controller, features, gates, _ = controller_fixture(review_only=True)
    with pytest.raises(TypeError):
        controller.decide(features,gates,policy={'kind':'global','buffer':0})


@pytest.mark.parametrize('field', ['future_global_margin','target','true_slice_member','scenario_id'])
def test_future_targets_or_extra_fields_rejected(field):
    controller, features, gates, models = controller_fixture(); features[field] = 0.
    with pytest.raises(ValueError,match='schema differs'):
        controller.decide(features,gates)
    assert models['lightgbm'].calls == 0


@pytest.mark.parametrize('value', [math.nan, math.inf, '0.1', None])
def test_non_numeric_or_nonfinite_evidence_rejected(value):
    controller, features, gates, _ = controller_fixture(); features['c1_harm_width'] = value
    with pytest.raises(ValueError,match='numeric and finite'):
        controller.decide(features,gates)


def test_input_dictionary_order_does_not_change_model_schema():
    controller, features, gates, _ = controller_fixture()
    normal = controller.decide(features,gates,family='global')
    shuffled = controller.decide(dict(reversed(list(features.items()))),gates,family='global')
    assert normal == shuffled


def test_missing_feature_rejected():
    controller, features, gates, _ = controller_fixture(); features.pop('fixture_field_00')
    with pytest.raises(ValueError,match='schema differs'):
        controller.decide(features,gates)


def test_model_feature_schema_mismatch_rejected():
    controller, _, _, models = controller_fixture()
    models['lightgbm'].input_columns = models['lightgbm'].input_columns[:-1]
    with pytest.raises(ValueError,match='model feature schema'):
        FrozenPilotController(models,controller.calibrations,controller.policies,list(controller.raw_columns))


def test_fixture_definition_cannot_claim_empirical_observations():
    definition = fixture_definition(); definition['research_observations'] = True
    with pytest.raises(ValueError,match='labelled synthetic'):
        acceptance_helpers()['acceptance_results'](definition,{'policy_id':'review','kind':'review_only'})


def test_controller_exposes_no_fit_or_deploy_method():
    assert not hasattr(FrozenPilotController,'fit')
    assert not hasattr(FrozenPilotController,'deploy')
