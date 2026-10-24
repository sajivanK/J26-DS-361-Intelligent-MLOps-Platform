"""
Contract checks: every component output is checked against the agreed contract files in
shared/contracts/ (required fields, types, allowed values). The contract files are the single
source of truth; this module only reads them.
J26-DS-361 | Intelligent MLOps Platform
"""
import copy
import json
from functools import lru_cache

from .settings import CONTRACTS_DIR

# output name -> contract file
CONTRACT_FILES = {
    "c1": "c1_output_schema.json",
    "c2_to_c3": "c2_output_to_c3_schema.json",
    "c2_to_c4": "c2_output_to_c4_schema.json",
    "c3": "c3_output_schema.json",
    "c4": "c4_output_schema.json",
}


@lru_cache(maxsize=None)
def load(name):
    with open(CONTRACTS_DIR / CONTRACT_FILES[name], encoding="utf-8") as f:
        return json.load(f)


def fields(name):
    c = load(name)
    return c.get("fields") or c.get("output_fields") or {}


def example(name, key=None):
    """A deep copy of the contract's example output (key picks one of several, e.g. C4)."""
    c = load(name)
    ex = c.get("example_output") or c.get("example_outputs")
    if key is not None:
        ex = ex[key]
    return copy.deepcopy(ex)


def _type_ok(value, t):
    t = (t or "").replace(" ", "")
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "float":
        return is_num
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "object":
        return isinstance(value, dict)
    if t == "array[float,float]":
        return (isinstance(value, list) and len(value) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value))
    if t == "array[string]":
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if t.startswith("array"):
        return isinstance(value, list)
    return True                                   # unknown type name: do not block


def validate(name, obj):
    """Return a list of problems (empty list = output matches the contract)."""
    if not isinstance(obj, dict):
        return [f"{name}: output is {type(obj).__name__}, expected a JSON object"]
    errors = []
    for field, spec in fields(name).items():
        if field not in obj:
            if spec.get("required"):
                errors.append(f"{name}.{field}: missing (required)")
            continue
        value = obj[field]
        if value is None:                         # v1.1: only fields marked "nullable" may be null
            if not spec.get("nullable"):
                errors.append(f"{name}.{field}: null is not allowed")
            continue
        if not _type_ok(value, spec.get("type")):
            errors.append(f"{name}.{field}: expected {spec.get('type')}, got {type(value).__name__}")
            continue
        allowed = spec.get("allowed_values")
        if allowed:
            values = value if isinstance(value, list) else [value]
            bad = [v for v in values if v not in allowed]
            if bad:
                errors.append(f"{name}.{field}: {bad} not in allowed values {allowed}")
    return errors
