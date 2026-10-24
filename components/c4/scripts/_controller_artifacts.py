"""Verify and load the frozen pilot artifacts; never read future target tables."""
import io
import json
from pathlib import Path
import joblib
from _common import C4_ROOT, checked_bytes, checked_json, require, sha
from _policy_pipeline import VERSION, POLICY_SPEC_HASH, verify_policy_spec, verify_context_support
from c4.conformal.benchmark_n2 import JointCalibration
from c4.risk.benchmark_controller import FrozenPilotController

POLICY_HASH = 'c838173e2410715a6e607269bfd0743ffeff745494c17a6467b8035915dcd0e8'
SNAPSHOT_HASH = 'd29e2f073c6dfe35241ebd0835cddccff6be08bc9faba1255dd6a2a06e93c4fd'
CONTROLLER_VERSION = 'c4-pilot-controller-v1'
SOURCE_FILES = ['scripts/_controller_artifacts.py', 'scripts/freeze_c4_controller.py',
    'scripts/verify_c4_controller.py', 'scripts/run_c4_controller.py', 'src/c4/risk/benchmark_controller.py',
    'configs/controller_acceptance_fixtures_v1.json']


def controller_directory(root):
    return root / 'c4_artifacts/controllers/adult' / VERSION / CONTROLLER_VERSION


def bindings(root):
    frozen, specdir = verify_policy_spec(root)
    support, support_hash = verify_context_support(root)
    policy_dir = root / 'c4_artifacts/policies/adult' / VERSION / 'policy-fit-v1'
    md, md_hash = checked_json(policy_dir)
    require(md_hash == POLICY_HASH and md['policy_specification_sha256'] == POLICY_SPEC_HASH
            and md['policy_snapshot_metadata_sha256'] == SNAPSHOT_HASH
            and md['policy_context_support_sha256'] == support_hash and md['n1_binding'] == frozen['n1_binding']
            and md['n2_metadata_sha256'] == frozen['n2_metadata_sha256']
            and md['n3_metadata_sha256'] == frozen['n3_metadata_sha256'], 'Frozen policy/model binding differs.')
    for relative, digest in md['source_sha256'].items(): checked_bytes(C4_ROOT / relative, digest)
    selection = json.loads(checked_bytes(policy_dir / 'selection.json', md['files']['selection.json']))
    require(selection['policy_specification_sha256'] == POLICY_SPEC_HASH
            and selection['policy_context_support_sha256'] == support_hash
            and selection['policy_snapshot_metadata_sha256'] == SNAPSHOT_HASH
            and selection['n1_binding'] == frozen['n1_binding'], 'Selected policies differ.')
    policies = {m: {family: result['selected_policy'] for family, result in families.items()}
                for m, families in selection['methods'].items()}
    # This controller version binds the actual measured unsupported-policy result.
    require(all(result['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY'
                and result['selected_policy']['kind'] == 'review_only'
                for families in selection['methods'].values() for result in families.values()),
            'Measured policy fallback differs; do not silently replace the frozen version.')
    n1dir = root / 'c4_artifacts/forecasting/adult' / VERSION / 'adult-n1-training-v1/n1-fit-v1'
    n1md, n1hash = checked_json(n1dir)
    spec, spec_hash = checked_json(n1dir.parent, 'specification')
    require(n1hash == frozen['n1_binding']['run_metadata_sha256']
            and spec_hash == frozen['n1_binding']['specification_sha256'], 'Frozen N1 binding differs.')
    n2dir = root / 'c4_artifacts/conformal/adult' / VERSION / 'n2-joint-v2'
    n2md, n2hash = checked_json(n2dir)
    require(n2hash == frozen['n2_metadata_sha256'], 'Frozen N2 binding differs.')
    calibration = json.loads(checked_bytes(n2dir / 'calibration.json', n2md['files']['calibration.json']))
    return {'scope': 'Adult accuracy-margin pilot; simulated upstream evidence',
        'policy_specification_sha256': POLICY_SPEC_HASH, 'policy_metadata_sha256': POLICY_HASH,
        'policy_snapshot_metadata_sha256': SNAPSHOT_HASH, 'policy_context_support_sha256': support_hash,
        'n1_binding': frozen['n1_binding'], 'n2_metadata_sha256': n2hash, 'n3_metadata_sha256': frozen['n3_metadata_sha256'],
        'raw_columns': spec['binding']['raw_columns'], 'selected_policies': policies,
        'unsupported_contexts': support['unsupported_contexts'], 'primary_method': selection['primary_method'],
        'primary_family': 'adaptive', 'automatic_deployment_enabled': False,
        'production_release_authorized': False}, (n1dir, n1md, calibration)


def load_controller(root):
    frozen, freeze_hash = checked_json(controller_directory(root), 'freeze')
    for relative, digest in frozen['source_sha256'].items(): checked_bytes(C4_ROOT / relative, digest)
    current, (n1dir, md, calibration) = bindings(root)
    require(frozen['binding'] == current, 'Frozen controller artifact bindings changed.')
    models = {m: joblib.load(io.BytesIO(checked_bytes(n1dir / (m+'.joblib'), md['files'][m+'.joblib'])))
              for m in ('lightgbm', 'linear')}
    fitted = {m: JointCalibration.from_dict(calibration['methods'][m]) for m in models}
    controller = FrozenPilotController(models, fitted, current['selected_policies'], current['raw_columns'], current['unsupported_contexts'])
    return controller, current, freeze_hash


def source_hashes():
    return {relative: sha((C4_ROOT / relative).read_bytes()) for relative in SOURCE_FILES}
