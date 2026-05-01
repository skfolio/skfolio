"""Equation module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import warnings

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
_NON_MUL_OPERATORS = _COMPARISON_OPERATORS.union(_SUB_ADD_OPERATORS)
_OPERATORS = _NON_MUL_OPERATORS.union(_MUL_OPERATORS)
_COMPARISON_OPERATOR_SIGNS = {">=": -1, "<=": 1, "==": 1, "=": 1}
_SUB_ADD_OPERATOR_SIGNS = {"+": 1, "-": -1}


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
        factor_groups = _validate_groups(factor_groups, name="factor_groups")
        if loading_matrix is None:
            raise ValueError(
                "`loading_matrix` must be provided when `factor_groups` is provided"
            )
        if factor_groups.shape[1] != loading_matrix.shape[1]:
            raise ValueError(
                f"`factor_groups` columns ({factor_groups.shape[1]}) must match "
                f"`loading_matrix` columns ({loading_matrix.shape[1]})"
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
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
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
    """Convert a list of linear equations into the left and right matrices of the
    inequality A <= B and equality A == B.

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
    left_inequality: ndarray of shape (n_constraints, n_assets)
    right_inequality: ndarray of shape (n_constraints,)
        The left and right matrices of the cardinality inequality.
    """
    groups = _validate_groups(groups, name="group")

    a_inequality = []
    b_inequality = []

    for group, card in group_cardinalities.items():
        try:
            arr = _matching_array(values=groups, key=group, sum_to_one=False)
            a_inequality.append(arr)
            b_inequality.append(card)

        except GroupNotFoundError as e:
            if raise_if_group_missing:
                raise
            warnings.warn(str(e), stacklevel=2)
    return (
        np.array(a_inequality),
        np.array(b_inequality),
    )


def _validate_groups(groups: ArrayLike, name: str = "groups") -> StrArray:
    """Validate groups by checking its dim and if group names don't appear in multiple
    levels and convert to numpy array.

    Parameters
    ----------
    groups : array-like of shape (n_groups, n_assets)
        2D-array of strings.

    Returns
    -------
    groups : ndarray of shape (n_groups, n_assets)
        2D-array of strings.
    """
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
                        f"'{e}' appear in two levels: {list(groups[i])} "
                        f"and {list(groups[i])}. "
                        f"{name} must be in only one level."
                    )

    return groups


def _validate_equations(equations: ArrayLike, name: str = "equations") -> StrArray:
    """Validate equations by checking its dim and convert to numpy array.

    Parameters
    ----------
    equations : array-like of shape (n_equations,)
        1D array of equations.

    Returns
    -------
    equations : ndarray of shape (n_equations,)
        1D array of equations.
    """
    equations = np.asarray(equations)

    if equations.ndim != 1:
        raise ValueError(
            f"`{name}` must be a 1D array, got {equations.ndim}D array instead."
        )
    return equations


def _matching_array(values: StrArray, key: str, sum_to_one: bool) -> FloatArray:
    """Takes in a 2D array of strings, a key string, and a boolean flag.
    It returns a 1D array where the value is 1 if there is a match between the key and
    any value in the 2D array, and 0 otherwise. The returned array can be scaled to
    have a sum of one if the flag is set to True.

    Parameters
    ----------
    values : ndarray of shape (n, m)
        2D-array of strings.

    key : str
        String to match in the values.

    sum_to_one : bool
        If this is set to True, the matching 1D-array is scaled to have a sum of one.

    Returns
    -------
    matching_array : ndarray of shape (n, )
        Matching 1D-array.
    """
    arr = np.any(values == key, axis=0)
    if not arr.any():
        raise EquationToMatrixError(f"Unable to find '{key}' in '{values}'")
    if sum_to_one:
        s = arr.sum()
    else:
        s = 1
    return arr / s


def _matching_array_with_factors(
    groups: StrArray,
    key: str,
    sum_to_one: bool,
    loading_matrix: FloatArray | None,
    factor_groups: FloatArray | None,
) -> FloatArray:
    """Match key in groups or factor_groups and return coefficient array.

    For asset groups, returns a binary selector (1 if asset in group, 0 otherwise).
    For factors, returns the corresponding loading vector(s) from loading_matrix.

    Parameters
    ----------
    groups : ndarray of shape (n_group_levels, n_assets)
        2D array of asset groups.

    key : str
        String to match in groups or factor_groups.

    sum_to_one : bool
        If True, the result is scaled to sum to one (applied only to asset groups, not factor exposures).

    loading_matrix : ndarray of shape (n_assets, n_factors) or None
        Factor loading matrix.

    factor_groups : ndarray of shape (n_factor_group_levels, n_factors) or None
        2D array of factor groups.

    Returns
    -------
    arr : ndarray of shape (n_assets,)
        Coefficient array for the constraint.

    Raises
    ------
    DuplicateGroupsError
        If key exists in both groups and factor_groups.
    FactorNotFoundError
        If key is in factor_groups but loading_matrix is None.
    GroupNotFoundError
        If key is not found in either groups or factor_groups.
    """
    group_names = set(groups.flatten())
    in_groups = key in group_names

    in_factors = False
    if factor_groups is not None:
        factor_names = set(factor_groups.flatten())
        in_factors = key in factor_names

    # Check for collision
    if in_groups and in_factors:
        raise DuplicateGroupsError(
            f"'{key}' exists in both asset groups and factor_groups. "
            "Names must be unique across groups and factor_groups."
        )

    if in_groups:
        # Original behavior: binary selector
        arr = np.any(groups == key, axis=0).astype(float)
        if sum_to_one:
            arr = arr / arr.sum()
        return arr

    if in_factors:
        if loading_matrix is None:
            raise FactorNotFoundError(
                f"Factor '{key}' found in factor_groups but loading_matrix is None."
            )
        # Sum loading vectors for all matching factors
        factor_match = np.any(factor_groups == key, axis=0)
        arr = loading_matrix[:, factor_match].sum(axis=1)
        return arr

    # Not found anywhere
    raise GroupNotFoundError(f"Unable to find '{key}' in groups or factor_groups")


def _comparison_operator_sign(operator: str) -> int:
    """Convert the operators '>=', "==" and '<=' into the corresponding integer
    values -1, 1 and 1, respectively.

    Parameters
    ----------
    operator : str
        Operator: '>=' or '<='.

    Returns
    -------
    value : int
        Operator sign: 1 or -1.
    """
    try:
        return _COMPARISON_OPERATOR_SIGNS[operator]
    except KeyError:
        raise EquationToMatrixError(
            f"operator '{operator}' is not valid. It should be '<=' or '>='"
        ) from None


def _sub_add_operator_sign(operator: str) -> int:
    """Convert the operators '+' and '-' into 1 or -1.

    Parameters
    ----------
    operator : str
       Operator: '+' and '-'.

    Returns
    -------
    value : int
       Operator sign: 1 or -1.
    """
    try:
        return _SUB_ADD_OPERATOR_SIGNS[operator]
    except KeyError:
        raise EquationToMatrixError(
            f"operator '{operator}' is not valid. It should be be '+' or '-'"
        ) from None


def _string_to_float(string: str) -> float:
    """Convert the factor string into a float.

    Parameters
    ----------
    string : str
       The factor string.

    Returns
    -------
    value : int
       The factor string converted to float.
    """
    try:
        return float(string)
    except ValueError:
        raise EquationToMatrixError(f"Unable to convert {string} into float") from None


def _split_equation_string(string: str) -> list[str]:
    """Split an equation strings by operators."""
    comp_pattern = "(?=" + "|".join([".+\\" + e for e in _COMPARISON_OPERATORS]) + ")"
    if not bool(re.match(comp_pattern, string)):
        raise EquationToMatrixError(
            f"The string must contains a comparison operator: "
            f"{list(_COMPARISON_OPERATORS)}"
        )

    # Regex to match only '>' and '<' but not '<=' or '>='
    invalid_pattern = r"(?<!<)(?<!<=)>(?!=)|(?<!>)<(?!=)"
    invalid_matches = re.findall(invalid_pattern, string)

    if len(invalid_matches) > 0:
        raise EquationToMatrixError(
            f"{invalid_matches[0]} is an invalid comparison operator. "
            f"Valid comparison operators are: {list(_COMPARISON_OPERATORS)}"
        )

    # '==' needs to be before '='
    operators = sorted(_OPERATORS, reverse=True)
    pattern = "((?:" + "|".join(["\\" + e for e in operators]) + "))"
    res = [x.strip() for x in re.split(pattern, string)]
    res = [x for x in res if x != ""]
    return res


def _string_to_equation(
    groups: StrArray,
    string: str,
    sum_to_one: bool,
    loading_matrix: FloatArray | None = None,
    factor_groups: FloatArray | None = None,
) -> tuple[FloatArray, float, bool]:
    """Convert a string to a left 1D-array and right float of the form:
    `groups @ left <= right` or `groups @ left == right` and return whether it's an
    equality or inequality.

    Parameters
    ----------
    groups : ndarray of shape (n_groups, n_assets)
        Groups 2D-array

    string : str
        String to convert

    sum_to_one : bool
        If this is set to True, the 1D-array is scaled to have a sum of one.

    loading_matrix : ndarray of shape (n_assets, n_factors) or None
        Factor loading matrix for factor constraints.

    factor_groups : ndarray of shape (n_factor_group_levels, n_factors) or None
        2D array of factor groups.

    Returns
    -------
    left : 1D-array of shape (n_assets,)
    right : float
    is_inequality : bool
    """
    n = groups.shape[1]
    err_msg = f"Wrong pattern encountered while converting the string '{string}'"

    iterator = iter(_split_equation_string(string))
    group_names = set(groups.flatten())
    factor_group_names = (
        set(factor_groups.flatten()) if factor_groups is not None else set()
    )

    def is_group_or_factor(name: str) -> bool:
        return name in group_names or name in factor_group_names

    left = np.zeros(n)
    right = 0
    main_sign = 1
    comparison_sign = None
    is_inequality = None
    e = next(iterator, None)
    i = 0
    while True:
        i += 1
        if i > 1e6:
            raise RecursionError(err_msg)
        if e is None:
            break
        sign = 1
        if e in _COMPARISON_OPERATORS:
            if e in _INEQUALITY_OPERATORS:
                is_inequality = True
            else:
                is_inequality = False
            main_sign = -1
            comparison_sign = _comparison_operator_sign(e)
            e = next(iterator, None)
            if e in _SUB_ADD_OPERATORS:
                sign *= _sub_add_operator_sign(e)
                e = next(iterator, None)
        elif e in _SUB_ADD_OPERATORS:
            sign *= _sub_add_operator_sign(e)
            e = next(iterator, None)
        elif e in _MUL_OPERATORS:
            raise EquationToMatrixError(
                f"{err_msg}: the character '{e}' is wrongly positioned"
            )
        sign *= main_sign
        # next can only be a number or a group
        if e is None or e in _OPERATORS:
            raise EquationToMatrixError(
                f"{err_msg}: the character '{e}' is wrongly positioned"
            )
        if is_group_or_factor(e):
            arr = _matching_array_with_factors(
                groups=groups,
                key=e,
                sum_to_one=sum_to_one,
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )
            # next can only be a '*' or an ['-', '+', '>=', '<=', '==', '='] or None
            e = next(iterator, None)
            if e is None or e in _NON_MUL_OPERATORS:
                left += sign * arr
            elif e in _MUL_OPERATORS:
                # next can only be a number
                e = next(iterator, None)
                try:
                    number = float(e)
                except ValueError:
                    raise GroupNotFoundError(
                        f"{err_msg}: the group '{e}' is missing from the groups"
                        f" {groups}"
                    ) from None

                left += number * sign * arr
                e = next(iterator, None)
            else:
                raise EquationToMatrixError(
                    f"{err_msg}: the character '{e}' is wrongly positioned"
                )
        else:
            try:
                number = float(e)
            except ValueError:
                raise GroupNotFoundError(
                    f"{err_msg}: the group '{e}' is missing from the groups {groups}"
                ) from None
            # next can only be a '*' or an operator or None
            e = next(iterator, None)
            if e in _MUL_OPERATORS:
                # next can only be a group or factor
                e = next(iterator, None)
                if not is_group_or_factor(e):
                    raise EquationToMatrixError(
                        f"{err_msg}: the character '{e}' is wrongly positioned"
                    )
                arr = _matching_array_with_factors(
                    groups=groups,
                    key=e,
                    sum_to_one=sum_to_one,
                    loading_matrix=loading_matrix,
                    factor_groups=factor_groups,
                )
                left += number * sign * arr
                e = next(iterator, None)
            elif e is None or e in _NON_MUL_OPERATORS:
                right += number * sign
            else:
                raise EquationToMatrixError(
                    f"{err_msg}: the character '{e}' is wrongly positioned"
                )

    left *= comparison_sign
    right *= -comparison_sign

    return left, right, is_inequality
