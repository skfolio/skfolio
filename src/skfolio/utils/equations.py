"""Equation module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from skfolio.exceptions import (
    DuplicateGroupsError,
    EquationToMatrixError,
    FactorNotFoundError,
    GroupNotFoundError,
)
from skfolio.typing import ArrayLike, FloatArray, StrArray

__all__ = ["equations_to_matrix", "group_cardinalities_to_matrix"]

_EQUALITY_OPERATORS = {"==", "="}
_INEQUALITY_OPERATORS = {">=", "<="}
_COMPARISON_OPERATORS = _EQUALITY_OPERATORS.union(_INEQUALITY_OPERATORS)
_SUB_ADD_OPERATORS = {"-", "+"}
_MUL_OPERATORS = {"*"}
_OPERATORS = _COMPARISON_OPERATORS.union(_SUB_ADD_OPERATORS, _MUL_OPERATORS)
_NAME_END = ""
_OPERATOR_CHARS = frozenset("+-*<=>")
_NUMBER_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class _EquationContext:
    """Precomputed equation parsing data."""

    n_assets: int
    group_indices: dict[str, np.ndarray]
    factor_indices: dict[str, np.ndarray]
    name_trie: dict[str, Any]
    loading_matrix: FloatArray | None


def equations_to_matrix(
    groups: ArrayLike,
    equations: ArrayLike,
    sum_to_one: bool = False,
    raise_if_group_missing: bool = False,
    names: tuple[str, str] = ("groups", "equations"),
    loading_matrix: ArrayLike | None = None,
    factor_groups: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Convert a list of linear equations into the left and right matrices of the
    inequality A <= B and equality A == B.

    Supports both asset group constraints and factor exposure constraints when
    `loading_matrix` and `factor_groups` are provided.

    Parameters
    ----------
    groups : array-like of shape (n_groups, n_assets)
        2D array of assets groups.

        For example:

             groups = np.array(
                [
                    ["SPX", "SX5E", "NKY", "TLT"],
                    ["Equity", "Equity", "Equity", "Bond"],
                    ["US", "Europe", "Japan", "US"],
                ]
            )

    equations : array-like of shape (n_equations,)
         1D array of equations.

         Example of valid equation patterns:
            * "number_1 * group_1 + number_3 <= number_4 * group_3 + number_5"
            * "group_1 == number * group_2"
            * "group_1 <= number"
            * "group_1 == number"

        "group_1" and "group_2" are the group names defined in `groups`.
        The second expression means that the sum of all assets in "group_1" should be
        less or equal to "number" times the sum of all assets in "group_2".

        When `loading_matrix` and `factor_groups` are provided, factor names and
        factor families can also be used in equations for factor exposure constraints.

        For example:

             equations = [
                "Equity <= 3 * Bond",
                "US >= 1.5",
                "Europe >= 0.5 * Japan",
                "Japan == 1",
                "3*SPX + 5*SX5E == 2*TLT + 3",
                "Momentum <= 0.3",  # Factor exposure constraint
                "style <= 0.5",     # Factor family constraint
            ]

    sum_to_one : bool
        If this is set to True, all elements in a group sum to one (used in the `views`
        of the Black-Litterman model).

    raise_if_group_missing : bool, default=False
        If this is set to True, an error is raised when a group is not found in the
        groups, otherwise only a warning is shown.
        The default is False.

    names : tuple[str, str], default=('groups', 'equations')
        The group and equation names used in error messages.
        The default is `('groups', 'equations')`.

    loading_matrix : array-like of shape (n_assets, n_factors), optional
        The factor loading matrix where each column represents a factor's exposure
        across assets. Required when using factor constraints.

    factor_groups : array-like of shape (n_factor_groups, n_factors), optional
        2D array of factor groups, similar to `groups` but for factors.

        For example:

             factor_groups = np.array(
                [
                    ["Momentum", "Value", "Size"],  # factor names
                    ["style", "style", "style"],    # factor families
                ]
            )

    Returns
    -------
    left_equality: ndarray of shape (n_equations_equality, n_assets)
    right_equality: ndarray of shape (n_equations_equality,)
        The left and right matrices of the equality A = B.

    left_inequality: ndarray of shape (n_equations_inequality, n_assets)
    right_inequality: ndarray of shape (n_equations_inequality,)
        The left and right matrices of the inequality A <= B.
    """
    groups = _validate_groups(groups, name=names[0])
    equations = _validate_equations(equations, name=names[1])

    _, n_assets = groups.shape

    # Validate and convert loading_matrix and factor_groups
    if loading_matrix is not None:
        loading_matrix = np.asarray(loading_matrix)
        if loading_matrix.ndim != 2:
            raise ValueError(
                f"`loading_matrix` must be a 2D array, got {loading_matrix.ndim}D array"
            )
        if loading_matrix.shape[0] != n_assets:
            raise ValueError(
                f"`loading_matrix` must have {n_assets} rows (n_assets), "
                f"got {loading_matrix.shape[0]}"
            )

    if factor_groups is not None:
        factor_groups = _validate_factor_groups(factor_groups, name="factor_groups")
        if loading_matrix is None:
            raise ValueError(
                "`loading_matrix` must be provided when `factor_groups` is provided"
            )
        if factor_groups.shape[1] != loading_matrix.shape[1]:
            raise ValueError(
                f"`factor_groups` columns ({factor_groups.shape[1]}) must match "
                f"`loading_matrix` columns ({loading_matrix.shape[1]})"
            )

    context = _build_equation_context(
        groups=groups,
        loading_matrix=loading_matrix,
        factor_groups=factor_groups,
    )

    a_equality = []
    b_equality = []

    a_inequality = []
    b_inequality = []

    for string in equations:
        try:
            left, right, is_inequality = _string_to_equation(
                groups=groups,
                string=string,
                sum_to_one=sum_to_one,
                context=context,
            )
            if is_inequality:
                a_inequality.append(left)
                b_inequality.append(right)
            else:
                a_equality.append(left)
                b_equality.append(right)
        except GroupNotFoundError as e:
            if raise_if_group_missing:
                raise
            warnings.warn(str(e), stacklevel=2)
        except FactorNotFoundError:
            # Always raise for factor constraints
            raise
    return (
        np.array(a_equality, dtype=float)
        if a_equality
        else np.empty((0, n_assets), dtype=float),
        np.array(b_equality, dtype=float),
        np.array(a_inequality, dtype=float)
        if a_inequality
        else np.empty((0, n_assets), dtype=float),
        np.array(b_inequality, dtype=float),
    )


def group_cardinalities_to_matrix(
    groups: ArrayLike,
    group_cardinalities: dict[str, int],
    raise_if_group_missing: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Convert group cardinality constraints into an inequality matrix.

    Parameters
    ----------
    groups : array-like of shape (n_groups, n_assets)
        2D array of assets groups.

        For example:

             groups = np.array(
                [
                    ["Equity", "Equity", "Equity", "Bond"],
                    ["US", "Europe", "Japan", "US"],
                ]
            )

    group_cardinalities : dict[str, int]
        Dictionary of cardinality constraint per group.
        For example: {"Equity": 1, "US": 3}

    raise_if_group_missing : bool, default=False
        If this is set to True, an error is raised when a group is not found in the
        groups, otherwise only a warning is shown.
        The default is False.

    Returns
    -------
    left_inequality : ndarray of shape (n_constraints, n_assets)
    right_inequality : ndarray of shape (n_constraints,)
        The left and right matrices of the cardinality inequality.
    """
    groups = _validate_groups(groups, name="group")
    _, n_assets = groups.shape
    group_indices = _build_column_indices(groups)

    a_inequality = []
    b_inequality = []

    for group, card in group_cardinalities.items():
        try:
            arr = _matching_array_from_indices(
                indices=group_indices,
                key=group,
                n_assets=n_assets,
                sum_to_one=False,
            )
            a_inequality.append(arr)
            b_inequality.append(card)

        except GroupNotFoundError as e:
            if raise_if_group_missing:
                raise
            warnings.warn(str(e), stacklevel=2)
    return (
        np.array(a_inequality, dtype=float)
        if a_inequality
        else np.empty((0, n_assets), dtype=float),
        np.array(b_inequality, dtype=float),
    )


def _validate_groups(groups: ArrayLike, name: str = "groups") -> StrArray:
    """Validate group dimensions and duplicate labels across levels."""
    groups = np.asarray(groups)
    if groups.ndim != 2:
        raise ValueError(
            f"`{name} must be a 2D array, got {groups.ndim}D array instead."
        )
    n = len(groups)
    group_sets = [set(groups[i]) for i in range(n)]
    for i in range(n - 1):
        for e in group_sets[i]:
            for j in range(i + 1, n):
                if e in group_sets[j]:
                    raise DuplicateGroupsError(
                        f"'{e}' appear in two levels: {groups[i].tolist()} "
                        f"and {groups[j].tolist()}. "
                        f"{name} must be in only one level."
                    )

    return groups


def _validate_factor_groups(
    factor_groups: ArrayLike, name: str = "factor_groups"
) -> StrArray:
    """Validate factor names and optional factor families."""
    factor_groups = np.asarray(factor_groups)
    if factor_groups.ndim != 2:
        raise ValueError(
            f"`{name} must be a 2D array, got {factor_groups.ndim}D array instead."
        )

    if factor_groups.shape[0] > 2:
        return _validate_groups(factor_groups, name=name)

    factor_names = factor_groups[0]
    factor_families = factor_groups[1] if factor_groups.shape[0] == 2 else None
    _validate_factor_names_and_families(
        factor_names=factor_names,
        factor_families=factor_families,
        name=name,
    )
    return factor_groups


def _validate_factor_names_and_families(
    factor_names: ArrayLike,
    factor_families: ArrayLike | None,
    name: str = "factor_groups",
) -> None:
    """Validate factor names and family labels used for factor constraints."""
    factor_names = np.asarray(factor_names)
    if factor_names.ndim != 1:
        raise ValueError(
            f"`factor_names` must be a 1D array, got {factor_names.ndim}D array."
        )
    if np.unique(factor_names).size != factor_names.size:
        raise DuplicateGroupsError(f"Factor names in `{name}` must be unique.")

    if factor_families is None:
        return

    factor_families = np.asarray(factor_families)
    if factor_families.ndim != 1:
        raise ValueError(
            f"`factor_families` must be a 1D array, got {factor_families.ndim}D array."
        )
    if factor_families.shape != factor_names.shape:
        raise ValueError(
            f"`factor_families` shape {factor_families.shape} must match "
            f"`factor_names` shape {factor_names.shape}."
        )

    for label in np.intersect1d(factor_names, factor_families):
        factor_idx = np.flatnonzero(factor_names == label)
        family_idx = np.flatnonzero(factor_families == label)
        if factor_idx.size == 1 and np.array_equal(factor_idx, family_idx):
            continue

        family_factor_names = factor_names[family_idx].tolist()
        raise DuplicateGroupsError(
            f"Factor name '{label}' in `{name}` is also used as a family name, "
            f"but the family contains {family_factor_names}. A factor name can "
            "match its family name only when the family contains that factor "
            "alone."
        )


def _validate_equations(equations: ArrayLike, name: str = "equations") -> StrArray:
    """Validate equation dimensions."""
    equations = np.asarray(equations)

    if equations.ndim != 1:
        raise ValueError(
            f"`{name}` must be a 1D array, got {equations.ndim}D array instead."
        )
    return equations


def _build_column_indices(values: StrArray) -> dict[str, np.ndarray]:
    """Map each label to the columns where it appears."""
    indices: dict[str, list[int]] = {}
    for row in values:
        for col, value in enumerate(row):
            indices.setdefault(str(value), []).append(col)

    return {
        key: np.array(sorted(set(cols)), dtype=int) for key, cols in indices.items()
    }


def _build_name_trie(names: set[str]) -> dict[str, Any]:
    """Build a trie used to match full names before operators."""
    root: dict[str, Any] = {}
    for name in names:
        if name == "":
            continue
        node = root
        for char in name:
            node = node.setdefault(char, {})
        node[_NAME_END] = name
    return root


def _build_equation_context(
    groups: StrArray,
    loading_matrix: FloatArray | None = None,
    factor_groups: StrArray | None = None,
) -> _EquationContext:
    """Precompute lookup data shared by all parsed equations."""
    n_assets = groups.shape[1]
    group_indices = _build_column_indices(groups)
    factor_indices = (
        _build_column_indices(factor_groups) if factor_groups is not None else {}
    )

    collisions = set(group_indices).intersection(factor_indices)
    if collisions:
        label = sorted(collisions)[0]
        raise DuplicateGroupsError(
            f"'{label}' exists in both groups and factor_groups. Names must be "
            "unique across groups and factor_groups."
        )

    names = set(group_indices).union(factor_indices)
    return _EquationContext(
        n_assets=n_assets,
        group_indices=group_indices,
        factor_indices=factor_indices,
        name_trie=_build_name_trie(names),
        loading_matrix=None
        if loading_matrix is None
        else np.asarray(loading_matrix, dtype=float),
    )


def _matching_array_from_indices(
    indices: dict[str, np.ndarray],
    key: str,
    n_assets: int,
    sum_to_one: bool,
) -> FloatArray:
    """Return a selector array from precomputed column indices."""
    try:
        columns = indices[key]
    except KeyError:
        raise GroupNotFoundError(f"Unable to find '{key}' in groups") from None

    arr = np.zeros(n_assets, dtype=float)
    arr[columns] = 1.0 / len(columns) if sum_to_one else 1.0
    return arr


def _matching_array(values: StrArray, key: str, sum_to_one: bool) -> FloatArray:
    """Return the columns matching a label as a selector array."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"`values` must be a 2D array, got {values.ndim}D array.")
    return _matching_array_from_indices(
        indices=_build_column_indices(values),
        key=key,
        n_assets=values.shape[1],
        sum_to_one=sum_to_one,
    )


def _matching_array_from_context(
    context: _EquationContext,
    key: str,
    sum_to_one: bool,
) -> FloatArray:
    """Match key in groups or factor groups from a precomputed context."""
    if key in context.group_indices:
        return _matching_array_from_indices(
            indices=context.group_indices,
            key=key,
            n_assets=context.n_assets,
            sum_to_one=sum_to_one,
        )

    if key in context.factor_indices:
        if context.loading_matrix is None:
            raise FactorNotFoundError(
                f"Factor '{key}' found in factor_groups but loading_matrix is None."
            )
        return context.loading_matrix[:, context.factor_indices[key]].sum(axis=1)

    raise GroupNotFoundError(f"Unable to find '{key}' in groups or factor_groups")


def _matching_array_with_factors(
    groups: StrArray,
    key: str,
    sum_to_one: bool,
    loading_matrix: FloatArray | None,
    factor_groups: StrArray | None,
) -> FloatArray:
    """Return an asset selector or factor loading vector for key."""
    context = _build_equation_context(
        groups=np.asarray(groups),
        loading_matrix=loading_matrix,
        factor_groups=None if factor_groups is None else np.asarray(factor_groups),
    )
    return _matching_array_from_context(
        context=context,
        key=key,
        sum_to_one=sum_to_one,
    )


def _is_token_boundary(string: str, position: int) -> bool:
    """Return True when a name or number can end at position."""
    return (
        position == len(string)
        or string[position].isspace()
        or string[position] in _OPERATOR_CHARS
    )


def _match_name(
    string: str,
    position: int,
    name_trie: dict[str, Any],
) -> tuple[str, int] | None:
    """Return the longest valid name starting at position."""
    node = name_trie
    match = None
    match_end = position
    i = position
    while i < len(string) and string[i] in node:
        node = node[string[i]]
        i += 1
        if _NAME_END in node:
            match = node[_NAME_END]
            match_end = i

    if match is not None and _is_token_boundary(string, match_end):
        return match, match_end
    return None


def _read_unknown_token(string: str, position: int) -> tuple[str, int]:
    """Read an unknown token until the next whitespace or operator."""
    i = position
    while (
        i < len(string) and not string[i].isspace() and string[i] not in _OPERATOR_CHARS
    ):
        i += 1
    return string[position:i], i


def _tokenize_equation_string(
    string: str,
    name_trie: dict[str, Any] | None = None,
) -> list[str]:
    """Tokenize an equation string with exact name matching."""
    tokens = []
    name_trie = {} if name_trie is None else name_trie
    i = 0
    while i < len(string):
        if string[i].isspace():
            i += 1
            continue

        name_match = _match_name(string, i, name_trie)
        if name_match is not None:
            token, i = name_match
            tokens.append(token)
            continue

        char = string[i]
        if char in {"<", ">"}:
            if i + 1 < len(string) and string[i + 1] == "=":
                tokens.append(char + "=")
                i += 2
                continue
            raise EquationToMatrixError(
                f"{char} is an invalid comparison operator. "
                f"Valid comparison operators are: {list(_COMPARISON_OPERATORS)}"
            )

        if char == "=":
            if i + 1 < len(string) and string[i + 1] == "=":
                tokens.append("==")
                i += 2
            else:
                tokens.append("=")
                i += 1
            continue

        if char in _SUB_ADD_OPERATORS or char in _MUL_OPERATORS:
            tokens.append(char)
            i += 1
            continue

        number_match = _NUMBER_PATTERN.match(string, i)
        if number_match is not None and _is_token_boundary(string, number_match.end()):
            tokens.append(number_match.group())
            i = number_match.end()
            continue

        token, i = _read_unknown_token(string, i)
        tokens.append(token)

    return tokens


def _split_equation_string(
    string: str,
    name_trie: dict[str, Any] | None = None,
) -> list[str]:
    """Split an equation string into names, numbers, and operators."""
    tokens = _tokenize_equation_string(string=string, name_trie=name_trie)
    n_comparisons = sum(token in _COMPARISON_OPERATORS for token in tokens)
    if n_comparisons == 0:
        raise EquationToMatrixError(
            f"The string must contain a comparison operator: "
            f"{list(_COMPARISON_OPERATORS)}"
        )
    if n_comparisons > 1:
        raise EquationToMatrixError(
            f"The string must contain only one comparison operator, found "
            f"{n_comparisons}."
        )
    return tokens


def _string_to_number(string: str, err_msg: str) -> float:
    """Convert a token to a number."""
    try:
        return float(string)
    except ValueError:
        raise GroupNotFoundError(
            f"{err_msg}: the group or factor '{string}' is missing"
        ) from None


def _parse_expression(
    tokens: list[str],
    context: _EquationContext,
    err_msg: str,
    sum_to_one: bool,
) -> tuple[FloatArray, float]:
    """Parse one side of a linear equation."""
    vector = np.zeros(context.n_assets, dtype=float)
    constant = 0.0
    sign = 1
    expect_term = True
    position = 0

    while position < len(tokens):
        token = tokens[position]

        if token in _SUB_ADD_OPERATORS:
            sign = 1 if token == "+" else -1
            position += 1
            expect_term = True
            continue

        if not expect_term:
            raise EquationToMatrixError(
                f"{err_msg}: the character '{token}' is wrongly positioned"
            )

        if token in _OPERATORS:
            raise EquationToMatrixError(
                f"{err_msg}: the character '{token}' is wrongly positioned"
            )

        position += 1
        if token in context.group_indices or token in context.factor_indices:
            arr = _matching_array_from_context(
                context=context,
                key=token,
                sum_to_one=sum_to_one,
            )
            number = 1.0
        else:
            number = _string_to_number(token, err_msg=err_msg)
            arr = None

        if position < len(tokens) and tokens[position] in _MUL_OPERATORS:
            position += 1
            if position == len(tokens):
                raise EquationToMatrixError(
                    f"{err_msg}: the character 'None' is wrongly positioned"
                )
            token = tokens[position]
            if arr is None:
                if (
                    token not in context.group_indices
                    and token not in context.factor_indices
                ):
                    raise EquationToMatrixError(
                        f"{err_msg}: the character '{token}' is wrongly positioned"
                    )
                arr = _matching_array_from_context(
                    context=context,
                    key=token,
                    sum_to_one=sum_to_one,
                )
            else:
                number = _string_to_number(token, err_msg=err_msg)
            position += 1

        if arr is None:
            constant += sign * number
        else:
            vector += sign * number * arr

        sign = 1
        expect_term = False

    if expect_term:
        raise EquationToMatrixError(
            f"{err_msg}: the character 'None' is wrongly positioned"
        )

    return vector, constant


def _string_to_equation(
    groups: StrArray,
    string: str,
    sum_to_one: bool,
    loading_matrix: FloatArray | None = None,
    factor_groups: StrArray | None = None,
    context: _EquationContext | None = None,
) -> tuple[FloatArray, float, bool]:
    """Convert a string into the left vector, right scalar, and constraint type."""
    err_msg = f"Wrong pattern encountered while converting the string '{string}'"
    context = (
        _build_equation_context(
            groups=groups,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        if context is None
        else context
    )

    tokens = _split_equation_string(
        string=string,
        name_trie=context.name_trie,
    )
    comparison_index = next(
        i for i, token in enumerate(tokens) if token in _COMPARISON_OPERATORS
    )
    operator = tokens[comparison_index]
    left_tokens = tokens[:comparison_index]
    right_tokens = tokens[comparison_index + 1 :]

    left_vector, left_constant = _parse_expression(
        tokens=left_tokens,
        context=context,
        err_msg=err_msg,
        sum_to_one=sum_to_one,
    )
    right_vector, right_constant = _parse_expression(
        tokens=right_tokens,
        context=context,
        err_msg=err_msg,
        sum_to_one=sum_to_one,
    )

    is_inequality = operator in _INEQUALITY_OPERATORS
    if operator == ">=":
        left = right_vector - left_vector
        right = left_constant - right_constant
    else:
        left = left_vector - right_vector
        right = right_constant - left_constant

    return left, right, is_inequality
