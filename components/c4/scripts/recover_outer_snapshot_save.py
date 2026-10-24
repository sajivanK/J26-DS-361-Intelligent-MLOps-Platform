"""Recover only the verified Windows Markdown newline mismatch into a new snapshot.

Preserves the original snapshot and frozen implementations. Reconstructs features
and paired targets from already saved evidence; no fitting or future prediction.
"""
import ast
from pathlib import Path
from _common import (C4_ROOT, parser, resolve_data_root, checked_json, checked_bytes,
                     require, sha, save_json, git_context, utc_now)
from _outer_pipeline import VERSION, verified_phase
from audit_outer_snapshot import collect

EXECUTION_HASH = 'e88d5702380b872e66b5b07e8cd9179a3863b379ad49377bde193a71803ab08d'
REGISTRY_HASH = '8630e0ad7e39b2ca2b9b4921fe531f6964e1ef48f2dc14de9c1350a5629b2493'
BATCH_HASH = '895c903553b2bcc1ac654f72386897ddada9b4fbd061c82925a2608bdc49261d'
EVIDENCE = 'C4_OUTER_SNAPSHOT_EVIDENCE.md'
OLD_VERSION = 'outer-all16-outcome-audit-v1'
NEW_VERSION = 'outer-all16-outcome-audit-recovered-v2'


def original_note():
    """Read the literal from the unchanged execution-pinned original audit script."""
    tree = ast.parse((C4_ROOT/'scripts/audit_outer_snapshot.py').read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id == 'note' for t in n.targets)]
    require(len(assignments) == 1, 'Cannot identify the original frozen Markdown literal.')
    note = ast.literal_eval(assignments[0].value)
    require(isinstance(note,str) and '\r' not in note and note.endswith('\n'),
            'Original evidence literal has an unexpected newline representation.')
    return note.encode('utf-8')


def verify_newline_mismatch(actual, canonical, expected_hash):
    require(sha(canonical) == expected_hash, 'Original metadata does not pin the frozen Markdown literal.')
    require(actual != canonical and actual == canonical.replace(b'\n', b'\r\n'),
            'Evidence mismatch is not exactly LF-to-CRLF translation; stop and investigate.')


def verified_payloads(directory, metadata, frames, canonical):
    require(set(metadata['files']) == set(frames) | {EVIDENCE}, 'Unexpected original snapshot file inventory.')
    payloads, original_hashes = {}, {}
    for filename, frame in frames.items():
        actual = checked_bytes(directory/filename, metadata['files'][filename])
        reconstructed = frame.to_csv(index=frame.index.name == 'scenario_id', lineterminator='\n').encode('utf-8')
        require(actual == reconstructed, 'Saved table differs from reconstructed frozen evidence: '+filename)
        payloads[filename] = actual
        original_hashes[filename] = sha(actual)
    actual = (directory/EVIDENCE).read_bytes()
    verify_newline_mismatch(actual, canonical, metadata['files'][EVIDENCE])
    original_hashes[EVIDENCE] = sha(actual)
    payloads[EVIDENCE] = canonical
    return payloads, original_hashes


def write_payloads(output, payloads):
    """Exclusive binary writes: the hashed and persisted bytes are identical."""
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, contents in payloads.items():
        with (output/filename).open('xb') as stream:
            stream.write(contents)
        files[filename] = sha(contents)
        checked_bytes(output/filename, files[filename])
    return files


def main():
    p = parser(__doc__)
    p.add_argument('--finish', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    parent = root/'c4_artifacts/meta_datasets/adult'/VERSION
    original = parent/OLD_VERSION
    output = parent/NEW_VERSION
    with verified_phase(root, 'audit') as phase:
        require(phase['sha256'] == EXECUTION_HASH, 'Different outer execution checkpoint.')
        metadata, metadata_hash = checked_json(original)
        original_metadata_bytes = (original/'metadata.json').read_bytes()
        original_sidecar_bytes = (original/'metadata.sha256').read_bytes()
        batch_dir = root/'c4_artifacts/benchmark_runs/adult'/VERSION/'outer_candidate_batches/outer-candidate-all16-v1'
        batch, batch_hash = checked_json(batch_dir, 'batch_results')
        require(batch_hash == BATCH_HASH and batch['completed_scenarios'] == 56 and
                batch['outer_execution_freeze_sha256'] == EXECUTION_HASH, 'Completed outer batch differs.')
        require(metadata['purpose'] == 'verified_outer_snapshot' and metadata['fold'] == 'outer_test' and
                metadata['outer_execution_freeze_sha256'] == EXECUTION_HASH and
                metadata['outer_pre_release_registry_sha256'] == REGISTRY_HASH and
                [metadata[k] for k in ('rows','assigned_groups','nonempty_groups','all_blocked_groups','admission_records','blocked')]
                == [56,16,14,2,592,536] and
                all(metadata[k] is False for k in ('N1_refitted','N2_refitted','policy_reselected',
                                                   'blocked_targets_imputed','production_release_authorized')),
                'Original snapshot scope/counts/state differs.')
        frames, links, registry_hash = collect(root)
        require(registry_hash == REGISTRY_HASH and links == metadata['source_links'] and
                list(frames['features.csv'].columns) == metadata['feature_columns'] and
                list(frames['targets.csv'].columns) == metadata['target_columns'],
                'Reconstructed outer evidence differs from original snapshot metadata.')
        payloads, original_hashes = verified_payloads(original, metadata, frames, original_note())
        print('ALL 56 OUTER CASES / SAVED TABLES / PAIRED TARGETS / ORIGINAL BINDINGS: PASS')
        print('SOLE MISMATCH VERIFIED: Markdown LF -> Windows CRLF; original files preserved.')
        print('RETRAINING / FUTURE PREDICTION / POLICY RESELECTION: False')
        require(not output.exists(), 'Recovery version already exists; preserve it.')
        if not args.finish:
            print('PLAN ONLY. Use --finish to save a separate verified snapshot.')
            return
        git = git_context()
        require(not git['working_tree_dirty'], 'Commit the additive recovery before --finish.')
        files = write_payloads(output, payloads)
        result = {**metadata, 'created_at_utc':utc_now(), **git, 'files':files,
            'snapshot_version':NEW_VERSION, 'recovery':{
                'kind':'exact_windows_markdown_newline_serialization_recovery',
                'original_snapshot_directory':str(original), 'original_metadata_sha256':metadata_hash,
                'original_actual_file_sha256':original_hashes,
                'original_expected_markdown_sha256':metadata['files'][EVIDENCE],
                'original_snapshot_modified':False, 'original_frozen_scripts_modified':False,
                'all_saved_tables_copied_byte_identically':True,
                'source_script_sha256':sha(Path(__file__).read_bytes()),
                'completed_outer_batch_sha256':BATCH_HASH}}
        digest = save_json(output/'metadata.json',result)
        for filename, expected in files.items():
            checked_bytes(output/filename,expected)
        for filename, expected in original_hashes.items():
            checked_bytes(original/filename,expected)
        require((original/'metadata.json').read_bytes() == original_metadata_bytes and
                (original/'metadata.sha256').read_bytes() == original_sidecar_bytes,
                'Original snapshot metadata changed during recovery.')
        checked_json(output)
        print('RECOVERED SNAPSHOT FILES / CHECKSUMS / ORIGINAL PRESERVATION: PASS')
        print('OUTPUT:',output)
        print('METADATA SHA256:',digest)
        print('NEXT: final outer comparison statistics using this recovered snapshot.')


if __name__ == '__main__':
    main()
