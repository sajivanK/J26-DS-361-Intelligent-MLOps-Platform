"""Exercise actual parser statements without loading research artifacts."""
import argparse
import ast
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def parse_script(filename, arguments):
    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = []
    for statement in main.body:
        statements.append(statement)
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "parse_args" for node in ast.walk(statement)):
            break
    else:
        raise AssertionError("No parser invocation found")
    namespace = {"parser": lambda description: argparse.ArgumentParser(description=description),
                 "__doc__": ast.get_docstring(tree)}
    with patch.object(sys, "argv", [filename, *arguments]):
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["args"]


def test_case_freeze_accepts_scenario_id_and_freeze_flag():
    args = parse_script("freeze_policy_case.py", ["--scenario-id", "example-policy-scenario", "--freeze"])
    assert args.scenario_id == "example-policy-scenario"
    assert args.freeze is True


def test_case_freeze_requires_scenario_id():
    with pytest.raises(SystemExit) as error:
        parse_script("freeze_policy_case.py", ["--freeze"])
    assert error.value.code == 2


def test_context_freeze_accepts_whole_fold_flag_without_scenario():
    args = parse_script("freeze_policy_context_support.py", ["--freeze"])
    assert args.freeze is True
    assert not hasattr(args, "scenario_id")


def test_context_freeze_rejects_per_scenario_argument():
    with pytest.raises(SystemExit) as error:
        parse_script("freeze_policy_context_support.py", ["--scenario-id", "example-policy-scenario", "--freeze"])
    assert error.value.code == 2
