"""Equation module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import re
import warnings

import numpy as np
import numpy.typing as npt

from skfolio.exceptions import (
    DuplicateGroupsError,
    EquationToMatrixError,
    GroupNotFoundError,
)

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
    groups: npt.ArrayLike,
    equations: npt.ArrayLike,
    sum_to_one: bool = False,
    raise_if_group_missing: bool = False,
    names: tuple[str, str] = ("groups", "equations"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of linear equations into the left and right matrices of the
    inequality A <= B and equality A == B.

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

        For example:

             equations = [
                "Equity <= 3 * Bond",
                "US >= 1.5",
                "Europe >= 0.5 * Japan",
                "Japan == 1",
                "3*SPX + 5*SX5E == 2*TLT + 3",
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

    Returns
    -------
    left_equality: ndarray of shape (n_equations_equality, n_assets)
    right_equality: ndarray of shape (n_equations_equality,)
        The left and right matrices of the inequality A <= B.

    left_inequality: ndarray of shape (n_equations_inequality, n_assets)
    right_inequality: ndarray of shape (n_equations_inequality,)
        The left and right matrices of the equality A == B.
    """
    groups = _validate_groups(groups, name=names[0])
    equations = _validate_equations(equations, name=names[1])

    _, n_assets = groups.shape

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
    groups: npt.ArrayLike,
    group_cardinalities: dict[str, int],
    raise_if_group_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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


def _validate_groups(groups: npt.ArrayLike, name: str = "groups") -> np.ndarray:
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


def _validate_equations(
    equations: npt.ArrayLike, name: str = "equations"
) -> np.ndarray:
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


def _matching_array(values: np.ndarray, key: str, sum_to_one: bool) -> np.ndarray:
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
    groups: np.ndarray,
    string: str,
    sum_to_one: bool,
) -> tuple[np.ndarray, float, bool]:
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

    def is_group(name: str) -> bool:
        return name in group_names

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
        if is_group(e):
            arr = _matching_array(values=groups, key=e, sum_to_one=sum_to_one)
            # next can only be a '*' or an ['-', '+', '>=', '<=', '==', '='] or None
            e = next(iterator, None)
            if e is None or e in _NON_MUL_OPERATORS:
                left += sign * arr
            elif e in _MUL_OPERATORS:
                # next can only a number
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
                # next can only a group
                e = next(iterator, None)
                if not is_group(e):
                    raise EquationToMatrixError(
                        f"{err_msg}: the character '{e}' is wrongly positioned"
                    )
                arr = _matching_array(values=groups, key=e, sum_to_one=sum_to_one)
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
