"""Recovery accepts exactly the known newline fault and preserves original bytes."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import pandas as pd
import pytest


def helper():
    common = ModuleType('_common')
    def require(ok, message):
        if not ok:raise ValueError(message)
    def sha(contents):return hashlib.sha256(contents).hexdigest()
    def checked_bytes(path, expected):
        contents=Path(path).read_bytes()
        require(sha(contents)==expected,'Checksum mismatch')
        return contents
    common.require=require;common.sha=sha;common.checked_bytes=checked_bytes
    common.C4_ROOT=Path(__file__).resolve().parents[2]
    for name in ('parser','resolve_data_root','checked_json','save_json','git_context','utc_now'):
        setattr(common,name,lambda *a,**k:None)
    outer=ModuleType('_outer_pipeline');outer.VERSION='adult-simulation-pp1-v2'
    outer.verified_phase=lambda *a,**k:None
    audit=ModuleType('audit_outer_snapshot');audit.collect=lambda *a:None
    source=common.C4_ROOT/'scripts/recover_outer_snapshot_save.py'
    spec=importlib.util.spec_from_file_location('isolated_outer_snapshot_recovery',source)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'_common':common,'_outer_pipeline':outer,'audit_outer_snapshot':audit}):
        spec.loader.exec_module(module)
    return module


def test_exact_crlf_translation_is_accepted():
    h=helper();canonical=b'# Evidence\n\n56 cases\n'
    h.verify_newline_mismatch(canonical.replace(b'\n',b'\r\n'),canonical,h.sha(canonical))


@pytest.mark.parametrize('actual',[b'# Changed\r\n',b'\xef\xbb\xbf# Evidence\r\n',b'# Evidence\r\n\n',b'# Evidence\n\n'])
def test_arbitrary_changes_mixed_newlines_bom_or_absent_fault_rejected(actual):
    h=helper();canonical=b'# Evidence\n\n'
    with pytest.raises(ValueError,match='not exactly'):h.verify_newline_mismatch(actual,canonical,h.sha(canonical))


def test_unknown_expected_markdown_hash_cannot_be_bypassed():
    h=helper()
    with pytest.raises(ValueError,match='does not pin'):h.verify_newline_mismatch(b'x\r\n',b'x\n','wrong')


def fixture(tmp_path,h):
    old=tmp_path/'original';old.mkdir()
    frame=pd.DataFrame({'value':[.25,-.5]},index=pd.Index(['s1','s2'],name='scenario_id'))
    contents=frame.to_csv(index=True,lineterminator='\n').encode()
    note=b'# Evidence\n56 cases\n'
    (old/'features.csv').write_bytes(contents)
    (old/h.EVIDENCE).write_bytes(note.replace(b'\n',b'\r\n'))
    md={'files':{'features.csv':h.sha(contents),h.EVIDENCE:h.sha(note)}}
    return old,md,{'features.csv':frame},note


def test_reconstruction_preserves_every_csv_and_only_canonicalizes_note(tmp_path):
    h=helper();old,md,frames,note=fixture(tmp_path,h)
    before={p.name:p.read_bytes() for p in old.iterdir()}
    payloads,hashes=h.verified_payloads(old,md,frames,note)
    assert payloads['features.csv']==before['features.csv'] and payloads[h.EVIDENCE]==note
    assert hashes[h.EVIDENCE]==h.sha(before[h.EVIDENCE])
    assert {p.name:p.read_bytes() for p in old.iterdir()}==before
    new=tmp_path/'recovered';files=h.write_payloads(new,payloads)
    assert files==md['files']
    assert (new/h.EVIDENCE).read_bytes()==note
    assert {p.name:p.read_bytes() for p in old.iterdir()}==before


def test_saved_table_must_match_new_reconstruction_even_with_valid_checksum(tmp_path):
    h=helper();old,md,frames,note=fixture(tmp_path,h)
    frames['features.csv'].loc['s1','value']=.9
    with pytest.raises(ValueError,match='reconstructed'):h.verified_payloads(old,md,frames,note)


def test_unexpected_extra_file_in_metadata_rejected(tmp_path):
    h=helper();old,md,frames,note=fixture(tmp_path,h);md['files']['extra.csv']='unknown'
    with pytest.raises(ValueError,match='inventory'):h.verified_payloads(old,md,frames,note)


def test_existing_recovery_is_never_overwritten(tmp_path):
    h=helper();output=tmp_path/'already-there';output.mkdir();(output/'marker').write_bytes(b'preserve')
    with pytest.raises(FileExistsError):h.write_payloads(output,{'new.txt':b'new'})
    assert (output/'marker').read_bytes()==b'preserve' and not (output/'new.txt').exists()


def test_original_literal_extracted_without_executing_audit_module(tmp_path):
    h=helper();h.C4_ROOT=tmp_path;(tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/audit_outer_snapshot.py').write_bytes(b"raise RuntimeError('must not execute')\ndef main():\n    note = '# Evidence\\n56 cases\\n'\n")
    assert h.original_note()==b'# Evidence\n56 cases\n'
