"""Fail-closed offline admission before the approved-package C4 workflow.

This precheck does not replace UpstreamInput validation or run_hard_gates.
It handles development metadata that cannot yet form an approved C3 input.
"""


def assess_repair_admission(upstream, generation, assessment):
    if upstream.get('data_role') != 'release_validation':
        raise ValueError('Admission requires a release-validation evidence context.')
    origins = upstream.get('source_declarations', {})
    c1 = upstream.get('c1_evidence', {})
    actual_sources = (
        origins.get('c1') == 'real' and origins.get('c2') == 'real'
        and not str(c1.get('engine_version', '')).lower().startswith('fake')
        and upstream.get('producer_authenticity_verified') is True
    )
    checks = {
        'upstream_producer_evidence_verified': actual_sources,
        'repair_package_approved': (
            generation.get('repair_package_approved') is True
            and generation.get('validation_status') == 'VALIDATED'
        ),
        'privacy_approved': generation.get('privacy_status') == 'PASS',
        'validation_protocol_frozen': assessment.get('protocol_status') == 'frozen',
        'screen_passed': assessment.get('screen_status') == 'PASS',
        'assessment_authorizes_training': assessment.get('candidate_training_authorized') is True,
    }
    reasons = {
        'upstream_producer_evidence_verified': 'Mock or unauthenticated upstream evidence is not a real C1/C2-backed scenario.',
        'repair_package_approved': 'The repair package is not validated and approved.',
        'privacy_approved': 'The package has no approved privacy PASS.',
        'validation_protocol_frozen': 'The applied development protocol is not frozen.',
        'screen_passed': 'The applied development screen did not pass.',
        'assessment_authorizes_training': 'The saved assessment does not authorize approved-package training.',
    }
    failed = [key for key, passed in checks.items() if not passed]
    ready = not failed
    return {
        'purpose': 'development_repair_admission_precheck',
        'admission_status': 'READY_FOR_STANDARD_VALIDATION' if ready else 'BLOCKED',
        'preflight_decision': None if ready else 'ESCALATE',
        'checks': checks, 'failed_checks': failed,
        'reasons': [reasons[key] for key in failed],
        'repair_precheck_passed': ready,
        # No training authorization is created by this precheck itself.
        'candidate_training_authorized': False,
        'standard_contract_and_runtime_gates_required': True,
        'final_release_authorized': False,
        'n1_n4_executed': False,
    }


def continue_after_precheck(result, standard_validation_workflow):
    """Blocked metadata must never reach the standard workflow callback."""
    if result['admission_status'] == 'BLOCKED':
        return None
    if result['admission_status'] != 'READY_FOR_STANDARD_VALIDATION':
        raise ValueError('Unknown admission state.')
    return standard_validation_workflow()
