"""Feature removals, training-fold isolation and actual saved-forecaster behavior."""
import io
import joblib
import numpy as np
import pandas as pd
import pytest
from c4.forecasting.benchmark_feature_ablation import VARIANTS, QUALITY_COLUMNS, ablation_columns, project_features
from c4.forecasting.benchmark_n1 import BenchmarkN1Config, BenchmarkQuantileForecaster, TARGETS, grouped_folds


def tables():
    ids=[f's{i}' for i in range(12)]
    columns=['c1_predicted_harm','c1_harm_width','c2_diagnosis_stability',*QUALITY_COLUMNS,
             'c3_n_synthetic_rows','c3_repair_approved','c4_global_candidate_minus_champion_accuracy','c4_holdout_only']
    x=pd.DataFrame({c:np.arange(12,dtype=float)/20 for c in columns},index=ids)
    x['c3_repair_approved']=1.
    x['c4_holdout_only']=[.1,.2,.3]+[0.]*9
    y=pd.DataFrame({t:np.arange(12,dtype=float)/1000 for t in TARGETS},index=ids)
    i=pd.DataFrame({'scenario_id':ids,'group_id':[f'g{n//3}' for n in range(12)],'split_role':['forecast_train']*12})
    return x,y,i


def test_A1_removes_only_direct_c1_columns():
    x,_,_=tables();kept,dropped=ablation_columns(x.columns,'A1')
    assert dropped==['c1_predicted_harm','c1_harm_width']
    assert all(not c.startswith('c1_') for c in kept)
    assert 'c2_diagnosis_stability' in kept and 'c3_repair_approved' in kept


def test_A2_retains_other_diagnosis_and_gate_inputs():
    x,_,_=tables();kept,dropped=ablation_columns(x.columns,'A2')
    assert dropped==['c2_diagnosis_stability']
    assert 'c1_predicted_harm' in kept and 'c3_repair_approved' in kept


def test_A3_removes_four_quality_fields_but_retains_repair_size_and_approval():
    x,_,_=tables();kept,dropped=ablation_columns(x.columns,'A3')
    assert set(dropped)==set(QUALITY_COLUMNS)
    assert {'c3_n_synthetic_rows','c3_repair_approved'} <= set(kept)


@pytest.mark.parametrize('variant',VARIANTS)
def test_projection_preserves_scenario_order_without_mutating_input(variant):
    x,_,_=tables();before=x.copy(deep=True)
    projected=project_features(x,variant,x.columns)
    assert projected.index.equals(x.index)
    assert projected.columns.tolist()==ablation_columns(x.columns,variant)[0]
    projected.iloc[0,0]=999.
    pd.testing.assert_frame_equal(x,before)


def test_wrong_projection_order_rejected():
    x,_,_=tables()
    with pytest.raises(ValueError,match='schema/order differs'):
        project_features(x.iloc[:,::-1],'A1',x.columns)


def test_extra_future_target_feature_rejected_by_projection():
    x,_,_=tables();raw=x.columns.tolist();x[TARGETS[0]]=0.
    with pytest.raises(ValueError,match='schema/order differs'):
        project_features(x,'A1',raw)


@pytest.mark.parametrize('variant',VARIANTS)
def test_missing_required_removed_feature_rejected(variant):
    x,_,_=tables()
    columns=[c for c in x.columns if not c.startswith('c1_')] if variant=='A1' else [c for c in x.columns if c!=('c2_diagnosis_stability' if variant=='A2' else QUALITY_COLUMNS[0])]
    with pytest.raises(ValueError,match='required|Required'):
        ablation_columns(columns,variant)


def test_duplicate_feature_names_rejected():
    with pytest.raises(ValueError,match='Unique'):
        ablation_columns(['c1_x','c1_x'],'A1')


def test_unknown_variant_cannot_silently_use_full_model():
    with pytest.raises(ValueError,match='Only A1'):
        ablation_columns(['c1_x','c4_x'],'A4')


@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_lightgbm_fit_and_byte_reload_predict_identically(variant):
    x,y,i=tables();projected=project_features(x,variant,x.columns)
    # Small synthetic test configuration only; research script uses the original frozen settings.
    cfg=BenchmarkN1Config(n_estimators=3,num_leaves=3,min_child_samples=2,cv_splits=2)
    fitted=BenchmarkQuantileForecaster('lightgbm',cfg).fit(projected,y,i)
    assert len(fitted.models)==9
    assert fitted.input_columns==projected.columns.tolist()
    assert not set(ablation_columns(x.columns,variant)[1]) & set(fitted.selected_columns)
    buffer=io.BytesIO();joblib.dump(fitted,buffer);restored=joblib.load(io.BytesIO(buffer.getvalue()))
    pd.testing.assert_frame_equal(fitted.predict(projected),restored.predict(projected),check_exact=True)


def test_fold_filter_never_uses_heldout_only_feature_variation():
    x,y,i=tables();projected=project_features(x,'A1',x.columns)
    train=np.flatnonzero(i.group_id.ne('g0'));valid=np.flatnonzero(i.group_id.eq('g0'))
    cfg=BenchmarkN1Config(n_estimators=3,num_leaves=3,min_child_samples=2,cv_splits=2)
    model=BenchmarkQuantileForecaster('lightgbm',cfg).fit(projected.iloc[train],y.iloc[train],i.iloc[train])
    assert 'c4_holdout_only' in model.constant_columns
    assert 'c4_holdout_only' not in model.selected_columns
    assert len(model.predict(projected.iloc[valid]))==3


@pytest.mark.parametrize('role',['conformal_calibration','conformal_extension','policy_calibration','outer_test'])
def test_nontraining_fold_cannot_fit_an_ablation(role):
    x,y,i=tables();i.loc[0,'split_role']=role
    with pytest.raises(ValueError,match='Only forecast_train'):
        BenchmarkQuantileForecaster('lightgbm').fit(project_features(x,'A3',x.columns),y,i)


def test_grouped_folds_are_shared_by_all_feature_variants():
    x,_,i=tables()
    expected=[(a.tolist(),b.tolist()) for a,b in grouped_folds(i,2)]
    for variant in VARIANTS:
        projected=project_features(x,variant,x.columns)
        assert projected.index.tolist()==i.scenario_id.tolist()
        actual=[(a.tolist(),b.tolist()) for a,b in grouped_folds(i,2)]
        assert actual==expected
        for train,valid in actual:
            assert not set(i.iloc[train].group_id) & set(i.iloc[valid].group_id)
