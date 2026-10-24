"""Shared helpers for scripts run from components/c4/scripts."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

C4_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = C4_ROOT.parents[1]
sys.path.insert(0, str(C4_ROOT / 'src'))

from c4.data import resolve_data_root


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--data-root', type=Path, default=None,
                        help='External dataset root; defaults to C4_DATA_ROOT.')
    return result


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(path, expected):
    contents = Path(path).read_bytes()
    require(sha(contents) == expected, f'Checksum mismatch: {path}')
    return contents


def checked_json(directory, stem='metadata'):
    directory = Path(directory)
    expected = (directory / f'{stem}.sha256').read_text(encoding='utf-8').strip()
    contents = checked_bytes(directory / f'{stem}.json', expected)
    return json.loads(contents), expected


def save_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as file:
        file.write(contents)
    checksum = sha(contents)
    with path.with_suffix('.sha256').open('x', encoding='utf-8') as file:
        file.write(checksum + '\n')
    return checksum


def git_context():
    def run(arguments):
        return subprocess.check_output(['git', *arguments], cwd=REPO_ROOT, text=True).strip()
    return {'git_revision': run(['rev-parse', 'HEAD']),
            'working_tree_dirty': bool(run(['status', '--porcelain']))}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def default_yaml():
    return C4_ROOT / 'configs' / 'scenario_config.yaml'


def name(value):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise argparse.ArgumentTypeError('Use an ID starting with an ASCII letter or number.')
    return value


def manifest_root(root, manifest_version):
    return root / 'c4_artifacts' / 'splits' / 'adult' / manifest_version
