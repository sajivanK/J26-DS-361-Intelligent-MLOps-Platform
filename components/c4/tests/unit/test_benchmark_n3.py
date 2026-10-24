"""Boundary semantics and the frozen fallback must hold independently of labels."""
import ast
import inspect
from pathlib import Path

import pytest

from c4.risk.benchmark_n3 import BOUNDARIES, TARGETS, assess_safety


def safe():
    return {t: [b, min(1., b+.1)] for t, b in BOUNDARIES.items()}


def test_inclusive_lower_boundary():
    r = assess_safety(safe(), auto_deploy_supported=True)
    assert r['release_signal'] == 'AUTO_DEPLOY'
    assert r['release_authorized'] is False
    assert all(v['status'] == 'SAFE' for v in r['dimensions'].values())


@pytest.mark.parametrize('target', TARGETS)
def test_upper_equal_boundary_is_uncertain(target):
    intervals = safe()
    b = BOUNDARIES[target]
    intervals[target] = [b-.1, b]
    assert assess_safety(intervals, auto_deploy_supported=True)['release_signal'] == 'HUMAN_REVIEW'


@pytest.mark.parametrize('target', TARGETS)
def test_one_unsafe_dimension_escalates(target):
    intervals = safe()
    b = BOUNDARIES[target]
    intervals[target] = [b-.1, b-.001]
    assert assess_safety(intervals, auto_deploy_supported=True)['release_signal'] == 'ESCALATE'


def test_fallback_all_uncertain():
    r = assess_safety({t: [-1., 1.] for t in TARGETS}, auto_deploy_supported=False)
    assert r['release_signal'] == 'HUMAN_REVIEW'
    assert all(v['status'] == 'UNCERTAIN' for v in r['dimensions'].values())


@pytest.mark.parametrize('pair', [[.2,.1], [-1.01,1], [float('nan'),1], [0,float('inf')], [True,1], ['0',1]])
def test_invalid_bounds_rejected(pair):
    intervals = safe()
    intervals[TARGETS[0]] = pair
    with pytest.raises(ValueError):
        assess_safety(intervals, auto_deploy_supported=True)


def test_missing_dimension_rejected():
    with pytest.raises(ValueError):
        assess_safety({}, auto_deploy_supported=True)


def test_no_partial_fallback_bypass():
    with pytest.raises(ValueError):
        assess_safety(safe(), auto_deploy_supported=False)


def test_support_must_be_boolean():
    with pytest.raises(ValueError):
        assess_safety(safe(), auto_deploy_supported='False')


def test_future_labels_not_api_arguments():
    assert set(inspect.signature(assess_safety).parameters) == {'intervals', 'auto_deploy_supported'}


def test_assessment_script_does_not_load_target_tables_or_fit_models():
    script = Path(__file__).resolve().parents[2] / 'scripts/assess_n3_benchmark.py'
    tree = ast.parse(script.read_text(encoding='utf-8'))
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value,str)]
    assert not any(s in strings for s in ('targets.csv', 'outcomes.json', 'future_predictions.csv'))
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls) & {'fit', 'predict', 'sample'}
