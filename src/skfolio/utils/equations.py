"""Equation module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import re
import warnings

import numpy as np
import numpy.typing as npt

from skfolio.exceptions import EquationToMatrixError, GroupNotFoundError

__all__ = ["equations_to_matrix"]


def equations_to_matrix(
    groups: npt.ArrayLike,
    equations: npt.ArrayLike,
    sum_to_one: bool = False,
    raise_if_group_missing: bool = False,
    names: tuple[str, str] = ("groups", "equations"),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of linear equations into the left and right matrices of the
    inequality A <= B.

    Parameters
    ----------
    groups : array-like of shape (n_groups, n_assets)
        2D array of assets groups.

        Examples:
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
            * "group_1 >= number * group_2"
            * "group_1 <= number"
            * "group_1 >= number"

        "group_1" and "group_2" are the group names defined in `groups`.
        The second expression means that the sum of all assets in "group_1" should be
        less or equal to "number" times the sum of all assets in "group_2".

        Examples:
             equations = [
                "Equity <= 3 * Bond",
                "US >= 1.5",
                "Europe >= 0.5 * Japan",
                "Japan <= 1",
                "3*SPX + 5*SX5E <= 2*TLT + 3",
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
    left: ndarray of shape (n_equations, n_assets)
    right: ndarray of shape (n_equations,)
        The left and right matrices of the inequality A <= B.
        If none of the group inside the equations are part of the groups, `None` is
        returned.
    """
    groups = np.asarray(groups)
    equations = np.asarray(equations)
    if groups.ndim != 2:
        raise ValueError(
            f"`{names[0]}` must be a 2D array, got {groups.ndim}D array instead."
        )
    if equations.ndim != 1:
        raise ValueError(
            f"`{names[1]}` must be a 1D array, got {equations.ndim}D array instead."
        )

    n_equations = len(equations)
    n_assets = groups.shape[1]
    a = np.zeros((n_equations, n_assets))
    b = np.zeros(n_equations)
    for i, string in enumerate(equations):
        try:
            left, right = _string_to_equation(
                groups=groups,
                string=string,
                sum_to_one=sum_to_one,
            )
            a[i] = left
            b[i] = right
        except GroupNotFoundError as e:
            if raise_if_group_missing:
                raise
            warnings.warn(str(e), stacklevel=2)
    return a, b


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
        s = np.sum(arr)
    else:
        s = 1
    return arr / s


_operator_mapping = {">=": -1, "<=": 1, "==": 1, "=": 1}
_operator_signs = {"+": 1, "-": -1}


def _inequality_operator_sign(operator: str) -> int:
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
        return _operator_mapping[operator]
    except KeyError:
        raise EquationToMatrixError(
            f"operator '{operator}' is not valid. It should be '<=' or '>='"
        ) from None


def _operator_sign(operator: str) -> int:
    """Convert the operators '+' and '-' into 1 or -1

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
        return _operator_signs[operator]
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


def _string_to_equation(
    groups: np.ndarray,
    string: str,
    sum_to_one: bool,
) -> tuple[np.ndarray, float]:
    """Convert a string to a left 1D-array and right float of the form:
    `groups @ left <= right`.

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
    left: 1D-array of shape (n_assets,)
    right: float
    """
    n = groups.shape[1]
    operators = ["-", "+", "*", ">=", "<=", "==", "="]
    invalid_operators = [">", "<"]
    pattern = re.compile(r"((?:" + "|\\".join(operators) + r"))")
    invalid_pattern = re.compile(r"((?:" + "|\\".join(invalid_operators) + r"))")
    err_msg = f"Wrong pattern encountered while converting the string '{string}'"

    res = re.split(pattern, string)
    res = [x.strip() for x in res]
    res = [x for x in res if x != ""]
    iterator = iter(res)
    group_names = set(groups.flatten())

    def is_group(name: str) -> bool:
        return name in group_names

    left = np.zeros(n)
    right = 0
    main_sign = 1
    inequality_sign = None
    e = next(iterator, None)
    i = 0
    while True:
        i += 1
        if i > 1e6:
            raise RecursionError(err_msg)
        if e is None:
            break
        sign = 1
        if e in [">=", "<=", "==", "="]:
            main_sign = -1
            inequality_sign = _inequality_operator_sign(e)
            e = next(iterator, None)
            if e in ["-", "+"]:
                sign *= _operator_sign(e)
                e = next(iterator, None)
        elif e in ["-", "+"]:
            sign *= _operator_sign(e)
            e = next(iterator, None)
        elif e == "*":
            raise EquationToMatrixError(
                f"{err_msg}: the character '{e}' is wrongly positioned"
            )
        sign *= main_sign
        # next can only be a number or a group
        if e is None or e in operators:
            raise EquationToMatrixError(
                f"{err_msg}: the character '{e}' is wrongly positioned"
            )
        if is_group(e):
            arr = _matching_array(values=groups, key=e, sum_to_one=sum_to_one)
            # next can only be a '*' or an ['-', '+', '>=', '<=', '==', '='] or None
            e = next(iterator, None)
            if e is None or e in ["-", "+", ">=", "<=", "==", "="]:
                left += sign * arr
            elif e == "*":
                # next can only a number
                e = next(iterator, None)
                try:
                    number = float(e)
                except ValueError:
                    invalid_ops = invalid_pattern.findall(e)
                    if len(invalid_ops) > 0:
                        raise EquationToMatrixError(
                            f"{invalid_ops[0]} is an invalid operator. Valid operators"
                            f" are: {operators}"
                        ) from None
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
                invalid_ops = invalid_pattern.findall(e)
                if len(invalid_ops) > 0:
                    raise EquationToMatrixError(
                        f"{invalid_ops[0]} is an invalid operator. Valid operators are:"
                        f" {operators}"
                    ) from None
                raise GroupNotFoundError(
                    f"{err_msg}: the group '{e}' is missing from the groups {groups}"
                ) from None
            # next can only be a '*' or an operator or None
            e = next(iterator, None)
            if e == "*":
                # next can only a group
                e = next(iterator, None)
                if not is_group(e):
                    raise EquationToMatrixError(
                        f"{err_msg}: the character '{e}' is wrongly positioned"
                    )
                arr = _matching_array(values=groups, key=e, sum_to_one=sum_to_one)
                left += number * sign * arr
                e = next(iterator, None)
            elif e is None or e in ["-", "+", ">=", "<=", "==", "="]:
                right += number * sign
            else:
                raise EquationToMatrixError(
                    f"{err_msg}: the character '{e}' is wrongly positioned"
                )

    left *= inequality_sign
    right *= -inequality_sign

    return left, right
