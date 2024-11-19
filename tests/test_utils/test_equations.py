import numpy as np
import pytest

from skfolio.exceptions import DuplicateGroupsError, EquationToMatrixError
from skfolio.utils.equations import (
    _COMPARISON_OPERATORS,
    _matching_array,
    _split_equation_string,
    _string_to_equation,
    equations_to_matrix,
    group_cardinalities_to_matrix,
)


@pytest.fixture
def groups():
    groups = np.array(
        [["a", "a", "b", "b"], ["c", "c", "j", "c"], ["d", "e", "d", "e"]]
    )
    return groups


@pytest.fixture
def group_cardinalities():
    group_cardinalities = {"a": 2, "b": 3, "e": 4}
    return group_cardinalities


def test_split_equation_string():
    for op in _COMPARISON_OPERATORS:
        for c in ["", " ", "   "]:
            string = "left" + c + op + c + "right"
            res = _split_equation_string(string)
            assert len(res) == 3


def test_split_equation_string_error():
    with pytest.raises(EquationToMatrixError):
        _split_equation_string("a*b")

    with pytest.raises(EquationToMatrixError):
        _split_equation_string("a>3")

    with pytest.raises(EquationToMatrixError):
        _split_equation_string("a<3")


def test_string_to_equation():
    string = "-5 - 3.5 * a + b - 2*c + 2 <= -1 + e*2.1 + f +6.5"
    groups = np.array([["a", "b", "c", "e", "f"]])
    left, right, is_inequality = _string_to_equation(
        groups=groups, string=string, sum_to_one=False
    )
    np.testing.assert_array_almost_equal(left, np.array([-3.5, 1.0, -2.0, -2.1, -1.0]))
    assert is_inequality is True
    assert right == 8.5

    string = "5 - 3.5 * a + a - c*2 + 2 +f >= -1 + e*2.1 + f +6.5"
    groups = np.array([["a", "a", "c", "e", "f"]])
    left, right, is_inequality = _string_to_equation(
        groups=groups, string=string, sum_to_one=False
    )
    assert is_inequality is True
    np.testing.assert_array_almost_equal(left, np.array([2.5, 2.5, 2, 2.1, 0]))
    assert right == 1.5

    string = "5 - 3.5 * a + a - c*2 + 2 +f == -1 + e*2.1 + f +6.5"
    groups = np.array([["a", "a", "c", "e", "f"]])
    left, right, is_inequality = _string_to_equation(
        groups=groups, string=string, sum_to_one=False
    )
    assert is_inequality is False
    np.testing.assert_array_almost_equal(left, np.array([-2.5, -2.5, -2, -2.1, 0]))
    assert right == -1.5


def test_matching_array(groups):
    arr = _matching_array(values=groups, key="a", sum_to_one=False)

    assert np.array_equal(arr, np.array([1.0, 1.0, 0.0, 0.0]))

    arr = _matching_array(values=groups, key="e", sum_to_one=False)
    assert np.array_equal(arr, np.array([0.0, 1.0, 0.0, 1.0]))

    arr = _matching_array(values=groups, key="a", sum_to_one=True)

    assert np.array_equal(arr, np.array([0.5, 0.5, 0.0, 0.0]))


def test_equations_to_matrix_inequality(groups):
    equations = ["a <= 2 * b ", "a <= 1.2", "3*c >= 0", "d >= 3 ", " e >=  .5*d"]
    a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(groups=groups, equations=equations)
    assert len(a_eq) == 0
    assert len(b_eq) == 0
    np.testing.assert_array_almost_equal(
        a_ineq,
        np.array(
            [
                [1.0, 1.0, -2.0, -2.0],
                [1.0, 1.0, 0.0, 0.0],
                [-3.0, -3.0, 0.0, -3.0],
                [-1.0, 0.0, -1.0, 0.0],
                [0.5, -1.0, 0.5, -1.0],
            ]
        ),
    )

    np.testing.assert_array_almost_equal(b_ineq, np.array([0.0, 1.2, 0, -3.0, 0.0]))


def test_equations_to_matrix_equality(groups):
    equations = ["a == 2 * b ", "a = 1.2", "3*c = 0", "d == 3 ", " e ==  .5*d"]
    a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(groups=groups, equations=equations)
    assert len(a_ineq) == 0
    assert len(b_ineq) == 0
    np.testing.assert_array_almost_equal(
        a_eq,
        np.array(
            [
                [1.0, 1.0, -2.0, -2.0],
                [1.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 3.0],
                [1.0, 0.0, 1.0, 0.0],
                [-0.5, 1.0, -0.5, 1.0],
            ]
        ),
    )

    np.testing.assert_array_almost_equal(b_eq, np.array([0.0, 1.2, 0, 3.0, 0.0]))


def test_equations_to_matrix_mix(groups):
    equations = ["a <= 2 * b ", "a = 1.2", "3*c = 0", "d == 3 ", " e >=  .5*d"]
    a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(groups=groups, equations=equations)

    np.testing.assert_array_almost_equal(
        a_eq,
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 3.0],
                [1.0, 0.0, 1.0, 0.0],
            ]
        ),
    )

    np.testing.assert_array_almost_equal(b_eq, np.array([1.2, 0, 3.0]))

    np.testing.assert_array_almost_equal(
        a_ineq,
        np.array(
            [
                [1.0, 1.0, -2.0, -2.0],
                [0.5, -1.0, 0.5, -1.0],
            ]
        ),
    )

    np.testing.assert_array_almost_equal(b_ineq, np.array([0.0, 0.0]))


def test_equations_to_matrix_error(groups):
    for c in [["a == "], ["a <= 2*bb"], ["a <= 2*b*c"]]:
        with pytest.raises(EquationToMatrixError):
            equations_to_matrix(groups=groups, equations=c)


def test_equations_to_matrix_duplicate_groups_error():
    groups = np.array(
        [["a", "a", "b", "b"], ["c", "a", "c", "a"], ["d", "e", "d", "e"]]
    )

    equations = ["a <= 2 * b ", "a <= 1.2", "d >= 3 ", " e >=  .5*d"]

    with pytest.raises(DuplicateGroupsError):
        _ = equations_to_matrix(groups=groups, equations=equations)


def test_views():
    groups = np.array(
        [
            [
                "Health Care",
                "Health Care",
                "Utilities",
                "Industrials",
                "Financials",
                "Industrials",
                "Energy",
            ]
        ]
    )

    equations = [
        "Health Care - Financials == 0.03 ",
        "Industrials - Utilities== 0.04",
        "Energy = 0.06 ",
    ]
    a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
        groups=groups, equations=equations, sum_to_one=True
    )
    assert len(a_ineq) == 0
    assert len(b_ineq) == 0
    assert np.array_equal(
        a_eq,
        np.array(
            [
                [0.5, 0.5, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.5, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert np.array_equal(b_eq, np.array([0.03, 0.04, 0.06]))


def test_group_cardinalities_to_matrix(groups, group_cardinalities):
    a, b = group_cardinalities_to_matrix(
        groups=groups, group_cardinalities=group_cardinalities
    )

    assert np.array_equal(
        a, np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    )

    assert np.array_equal(b, np.array([2, 3, 4]))


def test_group_cardinalities_to_matrix_error(groups, group_cardinalities):
    with pytest.raises(EquationToMatrixError):
        _ = group_cardinalities_to_matrix(
            groups=groups, group_cardinalities={"x": 5}, raise_if_group_missing=True
        )
