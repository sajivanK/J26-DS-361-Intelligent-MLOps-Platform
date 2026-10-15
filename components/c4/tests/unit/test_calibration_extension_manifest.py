"""Extension scope, pairing, RNG provenance and deterministic construction."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    root=Path(__file__).resolve().parents[2]
    # In the repository these are existing files; local package QA provides their paths.
    import os
    scripts=Path(os.environ.get('C4_BASE_SCRIPT_DIR',str(root/'scripts')))
    config=Path(os.environ.get('C4_BASE_PROTOCOL_CONFIG',str(root/'configs/benchmark_protocol.json')))
    base=load(scripts/'_benchmark_manifest.py','base_manifest_test')
    with patch.dict(sys.modules,{'_benchmark_manifest':base}):
        helper=load(root/'scripts/_calibration_extension_manifest.py','extension_manifest_test')
    protocol=json.loads(config.read_text())
    definition=json.loads((root/'configs/calibration_extension_v2.json').read_text())
    return helper,base.build_manifest(protocol),definition


def test_fixed_inventory_and_disjoint_ids():
    h,original,d=inputs(); ext=h.build_extension(original,d)
    assert len(ext['groups'])==20 and len(ext['scenarios'])==740
    assert not {r['scenario_id'] for r in original['scenarios']} & {r['scenario_id'] for r in ext['scenarios']}
    assert all(g['split_role']=='conformal_extension' for g in ext['groups'])


def test_deterministic_no_mutation():
    h,o,d=inputs(); before=copy.deepcopy(o)
    assert h.build_extension(o,d)==h.build_extension(o,d)
    assert o==before


def test_design_and_paired_seeds_retained():
    h,o,d=inputs(); ext=h.build_extension(o,d)
    for g in ext['groups']:
        rows=[r for r in ext['scenarios'] if r['group_id']==g['group_id']]
        assert len(rows)==37 and sum(r['is_no_drift_control'] for r in rows)==1
        bases={r['shared_base_package_id']:r for r in rows if r['repair_condition']=='targeted_repair'}
        assert len(bases)==19
        for row in rows:
            if row['repair_condition']=='label_corrupted_repair':
                b=bases[row['shared_base_package_id']]
                assert row['generator_seed']==b['generator_seed']
                assert row['drift_seeds']==b['drift_seeds']
                assert row['label_flip_fraction']==.2


@pytest.mark.parametrize('key,value', [('alpha',.1),('split_role','outer_test'),('expected_groups',2),
    ('replication_indices',[0,1,2,3,4]),('full_execution_enabled',True)])
def test_scope_changes_rejected(key,value):
    h,o,d=inputs();d[key]=value
    with pytest.raises(ValueError):h.build_extension(o,d)


def test_seed_collision_fails_without_replacement():
    h,o,d=inputs();h.seed=lambda *args:0
    with pytest.raises(ValueError,match='collision'):h.build_extension(o,d)


def test_original_manifest_incomplete_rejected():
    h,o,d=inputs();o['groups'].pop()
    with pytest.raises(ValueError):h.build_extension(o,d)
