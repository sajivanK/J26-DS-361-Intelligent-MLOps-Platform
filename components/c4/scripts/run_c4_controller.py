"""Infer a research decision from a JSON evidence vector and explicit hard-gate results."""
import json
from pathlib import Path
from _common import parser, resolve_data_root, require
from _controller_artifacts import load_controller


def main():
    p = parser(__doc__)
    p.add_argument('--input-json', type=Path, required=True)
    p.add_argument('--method', choices=['lightgbm', 'linear'], default='lightgbm')
    p.add_argument('--family', choices=['global', 'adaptive'], default='adaptive')
    args = p.parse_args(); controller, _, digest = load_controller(resolve_data_root(args.data_root))
    payload = json.loads(args.input_json.read_text(encoding='utf-8'))
    require(set(payload) == {'evidence_id', 'features', 'hard_gates'}, 'Expected evidence_id, features and hard_gates only.')
    result = controller.decide(payload['features'], payload['hard_gates'], evidence_id=payload['evidence_id'],
                               method=args.method, family=args.family)
    print(json.dumps({'controller_freeze_sha256': digest, **result}, indent=2, allow_nan=False))


if __name__ == '__main__': main()
