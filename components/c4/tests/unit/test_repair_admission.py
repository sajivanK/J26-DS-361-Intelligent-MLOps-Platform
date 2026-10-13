import pytest

from c4.experiments.repair_admission import assess_repair_admission, continue_after_precheck


def complete_fixture():
    return (
        {'data_role': 'release_validation', 'source_declarations': {'c1': 'real', 'c2': 'real'},
         'producer_authenticity_verified': True, 'c1_evidence': {'engine_version': '1.2.1'}},
        {'validation_status': 'VALIDATED', 'repair_package_approved': True, 'privacy_status': 'PASS'},
        {'protocol_status': 'frozen', 'screen_status': 'PASS', 'candidate_training_authorized': True},
    )


def test_current_development_case_is_blocked():
    upstream, package, assessment = complete_fixture()
    upstream['source_declarations'] = {'c1': 'mock', 'c2': 'mock'}
    upstream['producer_authenticity_verified'] = False
    package.update(validation_status='UNVALIDATED', repair_package_approved=False, privacy_status='NOT_EVALUATED')
    assessment.update(protocol_status='draft', screen_status='FAIL', candidate_training_authorized=False)
    result = assess_repair_admission(upstream, package, assessment)
    assert result['admission_status'] == 'BLOCKED'
    assert result['preflight_decision'] == 'ESCALATE'
    assert len(result['failed_checks']) == 6


def test_blocked_case_never_calls_downstream_workflow():
    inputs = complete_fixture()
    inputs[1]['repair_package_approved'] = False
    result = assess_repair_admission(*inputs)
    def forbidden():
        pytest.fail('Downstream workflow was called for a blocked package.')
    assert continue_after_precheck(result, forbidden) is None


def test_precheck_pass_is_not_training_or_release_authorization():
    result = assess_repair_admission(*complete_fixture())
    assert result['admission_status'] == 'READY_FOR_STANDARD_VALIDATION'
    assert result['candidate_training_authorized'] is False
    assert result['final_release_authorized'] is False
    assert result['standard_contract_and_runtime_gates_required'] is True


def test_precheck_pass_can_reach_standard_validation_only():
    result = assess_repair_admission(*complete_fixture())
    assert continue_after_precheck(result, lambda: 'standard-validation-required') == 'standard-validation-required'


def test_unknown_privacy_blocks_even_with_approval_flag():
    inputs = complete_fixture()
    inputs[1]['privacy_status'] = 'NOT_EVALUATED'
    assert 'privacy_approved' in assess_repair_admission(*inputs)['failed_checks']


def test_draft_screen_pass_still_blocks():
    inputs = complete_fixture()
    inputs[2]['protocol_status'] = 'draft'
    assert 'validation_protocol_frozen' in assess_repair_admission(*inputs)['failed_checks']


def test_fake_c1_cannot_be_used_as_real_producer_evidence():
    inputs = complete_fixture()
    inputs[0]['c1_evidence']['engine_version'] = 'fake-1.1'
    assert 'upstream_producer_evidence_verified' in assess_repair_admission(*inputs)['failed_checks']


def test_future_context_is_rejected():
    inputs = complete_fixture()
    inputs[0]['data_role'] = 'sealed_future'
    with pytest.raises(ValueError, match='release-validation'):
        assess_repair_admission(*inputs)
