"""Recovery retains the failed group and all original remaining work."""
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from resume_portable_training_batch import remaining_after_failure


def evidence():
    claim = {'group_ids_in_fixed_order': ['done', 'failed', 'next']}
    failure = {'group_id': 'failed', 'completed_group_records': [{'group_id': 'done', 'meta_rows': 2}]}
    return claim, failure


def test_resume_reuses_completion_and_keeps_failed_group():
    claim, failure = evidence()
    previous = remaining_after_failure(claim, failure, ['failed', 'next'])
    assert previous == failure['completed_group_records']


def test_failed_group_cannot_be_skipped():
    claim, failure = evidence()
    with pytest.raises(ValueError, match='remaining group order'):
        remaining_after_failure(claim, failure, ['next'])


def test_resume_cannot_reorder_original_remaining_groups():
    claim, failure = evidence()
    with pytest.raises(ValueError, match='remaining group order'):
        remaining_after_failure(claim, failure, ['next', 'failed'])


def test_duplicated_completion_is_rejected():
    claim, failure = evidence()
    failure['completed_group_records'] *= 2
    with pytest.raises(ValueError, match='duplicated'):
        remaining_after_failure(claim, failure, ['failed', 'next'])
