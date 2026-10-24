"""Pre-outer definition scope, finite-rank fallback and immutable study controls."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

BASE=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('isolated_evaluation_protocol',BASE/'scripts/_evaluation_protocol.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def definition():
    return json.loads((BASE/'configs/pp1_evaluation_protocol_v1.json').read_text(encoding='utf-8'))


def manifest():
    groups=[];rows=[]
    roles=['forecast_train']*32+['conformal_calibration']*20+['policy_calibration']*12+['outer_test']*16
    for n,role in enumerate(roles):
        gid=f'g{n}';groups.append({'group_id':gid,'split_role':role})
        rows.extend({'scenario_id':f'{gid}-s{i}','group_id':gid,'split_role':role} for i in range(37))
    return {'groups':groups,'scenarios':rows}


def test_declared_protocol_validates():
    data=definition();assert MODULE.validate_definition(data)==data
    assert data['outer_execution_authorized'] is False


@pytest.mark.parametrize('count,targets,expected',[(17,1,18),(18,1,19),(19,1,19),(37,1,37),(37,3,38),(58,3,59),(59,3,59)])
def test_exact_finite_sample_rank(count,targets,expected):
    assert MODULE.conformal_rank(count,targets)==expected


def test_outer_manifest_has_sixteen_unique_groups_and_592_cases():
    groups,rows=MODULE.verify_manifest(manifest())
    assert len(groups)==16 and len(rows)==592


def test_duplicate_group_identity_rejected():
    data=manifest();data['groups'][0]=copy.deepcopy(data['groups'][1])
    with pytest.raises(ValueError,match='duplicate groups'):
        MODULE.verify_manifest(data)


def test_duplicate_scenario_rejected():
    data=manifest();data['scenarios'][0]=copy.deepcopy(data['scenarios'][1])
    with pytest.raises(ValueError,match='unique scenario'):
        MODULE.verify_manifest(data)


def test_group_cannot_cross_folds():
    data=manifest();data['scenarios'][0]['split_role']='outer_test'
    with pytest.raises(ValueError,match='Group-fold'):
        MODULE.verify_manifest(data)


@pytest.mark.parametrize('target,value', [('future_global_margin',-.01),('future_slice_margin',-.01),('future_nonaffected_margin',-.03)])
def test_unchanged_safety_boundaries_required(target,value):
    data=definition();data['safety_boundaries'][target]=value
    with pytest.raises(ValueError,match='boundaries cannot change'):
        MODULE.validate_definition(data)


def test_definitions_do_not_authorize_outer_execution():
    data=definition();data['outer_execution_authorized']=True
    with pytest.raises(ValueError,match='cannot authorize'):
        MODULE.validate_definition(data)


def test_bonferroni_rank_cannot_be_clipped_to_available_groups():
    data=definition();data['bonferroni']['rank_clipping_allowed']=True
    with pytest.raises(ValueError,match='fallback cannot be bypassed'):
        MODULE.validate_definition(data)


def test_feature_ablation_cannot_claim_component_removal():
    data=definition();data['ablations'][0]['scope']='whole_C1_removed'
    with pytest.raises(ValueError,match='whole-component removal'):
        MODULE.validate_definition(data)


def test_bootstrap_pairing_is_mandatory():
    data=definition();data['statistics']['paired_draws_shared_across_methods']=False
    with pytest.raises(ValueError,match='bootstrap definition'):
        MODULE.validate_definition(data)


def test_undefined_conditional_risk_cannot_be_zero():
    data=definition();data['statistics']['zero_deployment_conditional_risk']='zero'
    with pytest.raises(ValueError,match='Undefined-statistic'):
        MODULE.validate_definition(data)


def test_b8_a8_equivalence_declared():
    data=definition();assert data['ablations'][7]['reuse']=='B8'


def test_blocked_target_imputation_rejected():
    data=definition();data['shared_controls']['blocked_target_imputation']=True
    with pytest.raises(ValueError,match='never impute'):
        MODULE.validate_definition(data)
