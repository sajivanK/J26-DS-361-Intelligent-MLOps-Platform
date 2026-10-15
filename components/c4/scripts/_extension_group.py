"""Admission scope bound to the independently frozen calibration extension."""
import json
from pathlib import Path
from _common import C4_ROOT, checked_bytes, checked_json, require
from _conformal_group import verify_n1 as verify_base_n1
from _training_group import corrupt_labels
from _calibration_extension_manifest import build_extension

VERSION='adult-simulation-pp1-v2'
EXTENSION='calibration-support-extension-v2'
EXTENSION_HASH='b4e5acc10a991aed3f2bcd735fb60ab5bda55d80f38e9bde3796343df2115507'
MANIFEST_HASH='f784cbd5dd88f796493df442916a6973ce138a14bc7ea1587aeedade99c8a584'


def extension_groups(manifest):
    groups=manifest['groups']
    expected=[f'adult-calext-v2-ns{ns}-r{i:03d}' for ns in (42,123,2026,3407) for i in (20,21,22,23,24)]
    require([g['group_id'] for g in groups]==expected and
            all(g['split_role']=='conformal_extension' for g in groups), 'Wrong fixed extension groups/order/fold.')
    return expected


def extension_group_cases(manifest, group_id):
    require(group_id in extension_groups(manifest),'Only a frozen conformal-extension group is allowed.')
    rows=[r for r in manifest['scenarios'] if r['group_id']==group_id]
    require(len(rows)==37 and len({r['scenario_id'] for r in rows})==37 and
            all(r['split_role']=='conformal_extension' for r in rows),'Wrong extension scenario inventory or fold.')
    drifted=[r for r in rows if not r['is_no_drift_control']]
    expected={(m,s,d,q) for m in ('marginal_multi','interaction','combined') for s in (.2,.5,.8)
              for d in ('exact_slice','broadened_slice') for q in ('targeted_repair','label_corrupted_repair')}
    require({(r['mechanism'],r['magnitude'],r['diagnosis_condition'],r['repair_condition']) for r in drifted}==expected,
            'Extension factorial differs.')
    controls=[r for r in rows if r['is_no_drift_control']]
    require(len(controls)==1 and controls[0]['mechanism']=='none' and controls[0]['magnitude']==0 and
            controls[0]['diagnosis_condition']=='exact_slice' and controls[0]['repair_condition']=='targeted_repair',
            'Extension control differs.')
    return sorted(rows,key=lambda r:(r['shared_base_package_id'],r['repair_condition']!='targeted_repair',r['scenario_id']))


def batch_groups(manifest, number):
    require(type(number) is int and 1<=number<=5,'Batch number must be 1 through 5.')
    return extension_groups(manifest)[(number-1)*4:number*4]


def load_extension(root):
    root=Path(root)
    n1=verify_base_n1(root)
    base=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    frozen,base_hash=checked_json(base,'freeze')
    require(base_hash==n1['protocol_freeze_sha256'] and frozen['full_execution_enabled'] is False,'Base protocol differs.')
    original=json.loads(checked_bytes(base/'scenario_manifest.json',frozen['scenario_manifest_sha256']))
    directory=root/'c4_artifacts/calibration_extensions/adult'/VERSION/EXTENSION
    ext,ext_hash=checked_json(directory,'freeze')
    require(ext_hash==EXTENSION_HASH and ext['n1_binding']==n1 and
            ext['base_protocol_freeze_sha256']==base_hash and ext['full_execution_enabled'] is False,
            'Extension freeze or N1 binding differs.')
    for relative,expected in ext['source_sha256'].items():checked_bytes(C4_ROOT/relative,expected)
    definition=json.loads(checked_bytes(directory/'definition.json',ext['definition_sha256']))
    manifest=json.loads(checked_bytes(directory/'scenario_manifest.json',ext['scenario_manifest_sha256']))
    require(ext['scenario_manifest_sha256']==MANIFEST_HASH and manifest==build_extension(original,definition),
            'Extension differs from its frozen deterministic design.')
    protocol_bytes=checked_bytes(directory/'base_protocol.json',ext['base_protocol_sha256'])
    require(protocol_bytes==checked_bytes(base/'protocol.json',frozen['protocol_sha256']),'Base protocol copy differs.')
    protocol=json.loads(protocol_bytes)
    require(ext['unchanged_admission_definition']==protocol['admission'] and ext['unchanged_N2_definition']==protocol['N2'],
            'Admission or N2 definitions changed.')
    for kind,path in (('n2',root/'c4_artifacts/conformal/adult'/VERSION/'n2-joint-v1'),
                      ('n3',root/'c4_artifacts/safety/adult'/VERSION/'n3-safety-v1')):
        stage,digest=checked_json(path)
        require(digest==ext['original_'+kind+'_metadata_sha256'] and stage['n1_binding']==n1,
                'Original '+kind+' artifact changed.')
    extension_groups(manifest)
    return frozen,base_hash,protocol,manifest,MANIFEST_HASH,ext_hash


def verify_n1(root):
    load_extension(root)
    return verify_base_n1(root)
