"""Restricted AST-based evaluator for Component 2 slice rules.

No eval(), exec(), or pandas.DataFrame.query() is used.
Only allowlisted AST operations, columns, and literals are supported.
"""

import ast
import operator
from typing import Any
import pandas as pd

MAX_RULE_LENGTH = 2000
MAX_MEMBERSHIP_ITEMS = 100


class UnsafeSliceRuleError(ValueError):
    """Raised for invalid or unsupported slice expressions."""


COMPARISON_OPERATORS = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.In: lambda a, b: a.isin(b),
    ast.NotIn: lambda a, b: ~a.isin(b),
}


def _eval_literal(node: ast.AST) -> Any:
    """Evaluate a literal or negative number safely."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            val = _eval_literal(node.operand)
            if isinstance(val, (int, float)):
                return -val
        elif isinstance(node.op, ast.UAdd):
            val = _eval_literal(node.operand)
            if isinstance(val, (int, float)):
                return +val
        raise UnsafeSliceRuleError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_MEMBERSHIP_ITEMS:
            raise UnsafeSliceRuleError(
                f"Membership list exceeds maximum of {MAX_MEMBERSHIP_ITEMS} items."
            )
        return [_eval_literal(elem) for elem in node.elts]
    raise UnsafeSliceRuleError(f"Expected literal value, found {type(node).__name__}")


def _eval_operand(node: ast.AST, df: pd.DataFrame) -> pd.Series | Any:
    """Evaluate an operand in a comparison (column reference or literal)."""
    if isinstance(node, ast.Name):
        col_name = node.id
        if col_name not in df.columns:
            raise UnsafeSliceRuleError(f"Unknown column referenced in rule: '{col_name}'")
        return df[col_name]
    return _eval_literal(node)


def _eval_ast_node(node: ast.AST, df: pd.DataFrame) -> pd.Series | Any:
    """Recursively evaluate an AST node against df columns."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, df)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            combined_mask = None
            for val_node in node.values:
                sub_mask = _eval_ast_node(val_node, df)
                if not isinstance(sub_mask, pd.Series):
                    raise UnsafeSliceRuleError("Condition did not produce a boolean mask.")
                combined_mask = sub_mask if combined_mask is None else (combined_mask & sub_mask)
            return combined_mask
        elif isinstance(node.op, ast.Or):
            combined_mask = None
            for val_node in node.values:
                sub_mask = _eval_ast_node(val_node, df)
                if not isinstance(sub_mask, pd.Series):
                    raise UnsafeSliceRuleError("Condition did not produce a boolean mask.")
                combined_mask = sub_mask if combined_mask is None else (combined_mask | sub_mask)
            return combined_mask
        raise UnsafeSliceRuleError(f"Unsupported boolean operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            sub_mask = _eval_ast_node(node.operand, df)
            if not isinstance(sub_mask, pd.Series):
                raise UnsafeSliceRuleError("not operator requires a boolean condition.")
            return ~sub_mask
        return _eval_literal(node)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise UnsafeSliceRuleError("Chained comparisons are not allowed in slice rules.")

        left_val = _eval_operand(node.left, df)
        curr_left = left_val
        overall_mask = None

        for op, comp_node in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in COMPARISON_OPERATORS:
                raise UnsafeSliceRuleError(f"Unsupported comparison operator: {op_type.__name__}")
            comp_func = COMPARISON_OPERATORS[op_type]
            right_val = _eval_operand(comp_node, df)

            try:
                step_mask = comp_func(curr_left, right_val)
            except Exception as exc:
                raise UnsafeSliceRuleError(f"Comparison evaluation error: {exc}") from exc

            if not isinstance(step_mask, pd.Series):
                raise UnsafeSliceRuleError("Comparison did not involve a DataFrame column.")

            overall_mask = step_mask if overall_mask is None else (overall_mask & step_mask)
            curr_left = right_val

        return overall_mask

    if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp, ast.Lambda, ast.Dict)):
        raise UnsafeSliceRuleError(f"Disallowed expression construct: {type(node).__name__}")

    raise UnsafeSliceRuleError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_slice_rule(df: pd.DataFrame, rule: str) -> pd.Series:
    """Return a boolean mask for a restricted C2 slice rule."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise UnsafeSliceRuleError("A non-empty DataFrame is required.")
    if not isinstance(rule, str) or not rule.strip():
        raise UnsafeSliceRuleError("Slice rule must be a non-empty string.")
    if len(rule) > MAX_RULE_LENGTH:
        raise UnsafeSliceRuleError(
            f"Slice rule exceeds maximum length of {MAX_RULE_LENGTH} characters."
        )

    import re
    cleaned_rule = rule.strip()
    cleaned_rule = re.sub(r"\bAND\b", "and", cleaned_rule)
    cleaned_rule = re.sub(r"\bOR\b", "or", cleaned_rule)
    cleaned_rule = re.sub(r"\bNOT\b", "not", cleaned_rule)

    try:
        parsed = ast.parse(cleaned_rule, mode="eval")
    except SyntaxError as exc:
        raise UnsafeSliceRuleError(f"Syntax error in slice rule: {exc}") from exc

    result = _eval_ast_node(parsed, df)
    if not isinstance(result, pd.Series):
        raise UnsafeSliceRuleError("Slice rule must evaluate to a boolean Series.")
    return result.astype(bool)


def extract_affected_slice(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Extract affected records matching the given restricted slice rule."""
    mask = evaluate_slice_rule(df, rule)
    return df.loc[mask].copy()
