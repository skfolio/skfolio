import numpy as np

from skfolio.exceptions import EquationToMatrixError
from skfolio.utils.equations import _string_to_equation, equations_to_matrix


def test_string_to_equation():
    string = "-5 - 3.5 * a + b - 2*c + 2 <= -1 + e*2.1 + f +6.5"
    groups = np.array([["a", "b", "c", "e", "f"]])
    left, right = _string_to_equation(groups=groups, string=string, sum_to_one=False)
    np.testing.assert_array_almost_equal(left, np.array([-3.5, 1.0, -2.0, -2.1, -1.0]))
    assert right == 8.5
    string = "5 - 3.5 * a + a - c*2 + 2 +f >= -1 + e*2.1 + f +6.5"
    groups = np.array([["a", "a", "c", "e", "f"]])
    left, right = _string_to_equation(groups=groups, string=string, sum_to_one=False)
    np.testing.assert_array_almost_equal(left, np.array([2.5, 2.5, 2, 2.1, 0]))
    assert right == 1.5


def test_equations_to_matrix():
    groups = np.array(
        [["a", "a", "b", "b"], ["c", "a", "c", "a"], ["d", "e", "d", "e"]]
    )

    equations = ["a <= 2 * b ", "a <= 1.2", "d >= 3 ", " e >=  .5*d"]
    a, b = equations_to_matrix(groups=groups, equations=equations)
    np.testing.assert_array_almost_equal(
        a,
        np.array([
            [1.0, 1.0, -2.0, -1.0],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.5, -1.0, 0.5, -1.0],
        ]),
    )

    np.testing.assert_array_almost_equal(b, np.array([0.0, 1.2, -3.0, 0.0]))

    for c in [["a == "], ["a <= 2*bb"], ["a <= 2*b*c"]]:
        try:
            equations_to_matrix(groups=groups, equations=c)
            raise
        except EquationToMatrixError:
            pass


def views():
    groups = np.array([[
        "Health Care",
        "Health Care",
        "Utilities",
        "Industrials",
        "Financials",
        "Industrials",
        "Energy",
    ]])

    equations = [
        "Health Care - Financials == 0.03 ",
        "Industrials - Utilities== 0.04",
        "Energy == 0.06 ",
    ]
    a, b = equations_to_matrix(groups=groups, equations=equations, sum_to_one=True)
    assert np.array_equal(
        a,
        np.array([
            [0.5, 0.5, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]),
    )
    assert np.array_equal(b, np.array([0.03, 0.04, 0.06]))
