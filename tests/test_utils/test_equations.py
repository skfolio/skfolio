import numpy as np
import pytest

from skfolio.exceptions import (
    DuplicateGroupsError,
    EquationToMatrixError,
    FactorNotFoundError,
    GroupNotFoundError,
)
from skfolio.utils.equations import (
    _COMPARISON_OPERATORS,
    _matching_array,
    _matching_array_with_factors,
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


@pytest.fixture
def factor_groups():
    """Factor groups with factor names and factor families."""
    return np.array(
        [
            ["Momentum", "Value", "Size"],  # factor names
            ["style", "style", "style"],  # factor families
        ]
    )


@pytest.fixture
def loading_matrix():
    """Loading matrix of shape (n_assets=4, n_factors=3)."""
    return np.array(
        [
            [0.5, 0.2, -0.1],  # asset 0
            [0.3, 0.4, 0.2],  # asset 1
            [-0.2, 0.1, 0.5],  # asset 2
            [0.1, -0.3, 0.3],  # asset 3
        ]
    )


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


class TestMatchingArrayWithFactors:
    """Tests for _matching_array_with_factors function."""

    def test_matching_asset_group(self, groups, loading_matrix, factor_groups):
        """Test matching an asset group returns binary selector."""
        arr = _matching_array_with_factors(
            groups=groups,
            key="a",
            sum_to_one=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        np.testing.assert_array_equal(arr, np.array([1.0, 1.0, 0.0, 0.0]))

    def test_matching_asset_group_sum_to_one(
        self, groups, loading_matrix, factor_groups
    ):
        """Test matching an asset group with sum_to_one=True."""
        arr = _matching_array_with_factors(
            groups=groups,
            key="a",
            sum_to_one=True,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        np.testing.assert_array_equal(arr, np.array([0.5, 0.5, 0.0, 0.0]))

    def test_matching_factor_name(self, groups, loading_matrix, factor_groups):
        """Test matching a factor name returns the loading vector."""
        arr = _matching_array_with_factors(
            groups=groups,
            key="Momentum",
            sum_to_one=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        # Should return the first column of loading_matrix (Momentum)
        np.testing.assert_array_almost_equal(arr, np.array([0.5, 0.3, -0.2, 0.1]))

    def test_matching_factor_family(self, groups, loading_matrix, factor_groups):
        """Test matching a factor family returns sum of all matching loading vectors."""
        arr = _matching_array_with_factors(
            groups=groups,
            key="style",
            sum_to_one=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        # Should return sum of all columns (all factors are 'style')
        expected = loading_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(arr, expected)

    def test_collision_raises_error(self, loading_matrix):
        """Test that collision between groups and factor_groups raises error."""
        groups = np.array([["Momentum", "Value", "Size", "Other"]])  # Collision!
        factor_groups = np.array([["Momentum", "Value", "Size"]])

        with pytest.raises(DuplicateGroupsError):
            _matching_array_with_factors(
                groups=groups,
                key="Momentum",
                sum_to_one=False,
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )

    def test_factor_without_loading_matrix_raises_error(self, groups, factor_groups):
        """Test that factor constraint without loading_matrix raises error."""
        with pytest.raises(FactorNotFoundError):
            _matching_array_with_factors(
                groups=groups,
                key="Momentum",
                sum_to_one=False,
                loading_matrix=None,
                factor_groups=factor_groups,
            )

    def test_not_found_raises_error(self, groups, loading_matrix, factor_groups):
        """Test that unknown key raises GroupNotFoundError."""
        with pytest.raises(GroupNotFoundError):
            _matching_array_with_factors(
                groups=groups,
                key="Unknown",
                sum_to_one=False,
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )

    def test_no_factor_groups_only_asset_groups(self, groups):
        """Test with no factor_groups - should work like original _matching_array."""
        arr = _matching_array_with_factors(
            groups=groups,
            key="a",
            sum_to_one=False,
            loading_matrix=None,
            factor_groups=None,
        )
        np.testing.assert_array_equal(arr, np.array([1.0, 1.0, 0.0, 0.0]))


class TestStringToEquationWithFactors:
    """Tests for _string_to_equation with factor constraints."""

    def test_factor_constraint(self, groups, loading_matrix, factor_groups):
        """Test parsing a factor constraint equation."""
        string = "Momentum <= 0.3"
        left, right, is_inequality = _string_to_equation(
            groups=groups,
            string=string,
            sum_to_one=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        # Momentum is column 0 of loading_matrix
        np.testing.assert_array_almost_equal(left, np.array([0.5, 0.3, -0.2, 0.1]))
        assert right == 0.3
        assert is_inequality is True

    def test_mixed_asset_and_factor_constraint(
        self, groups, loading_matrix, factor_groups
    ):
        """Test parsing an equation with both asset groups and factors."""
        string = "a + 2*Momentum <= 0.5"
        left, right, is_inequality = _string_to_equation(
            groups=groups,
            string=string,
            sum_to_one=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        # a: [1, 1, 0, 0] + 2 * Momentum: 2*[0.5, 0.3, -0.2, 0.1]
        expected = np.array([1.0, 1.0, 0.0, 0.0]) + 2 * np.array([0.5, 0.3, -0.2, 0.1])
        np.testing.assert_array_almost_equal(left, expected)
        assert right == 0.5
        assert is_inequality is True

    def test_factor_family_constraint(self, groups, loading_matrix, factor_groups):
        """Test parsing a factor family constraint equation."""
        string = "style == 0"
        left, right, is_inequality = _string_to_equation(
            groups=groups,
            string=string,
            sum_to_one=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        # style matches all factors, so sum of all loading columns
        expected = loading_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(left, expected)
        assert right == 0
        assert is_inequality is False


class TestEquationsToMatrixWithFactors:
    """Tests for equations_to_matrix with factor constraints."""

    def test_factor_inequality_constraints(self, groups, loading_matrix, factor_groups):
        """Test factor inequality constraints."""
        equations = ["Momentum <= 0.3", "Value >= -0.1"]
        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=groups,
            equations=equations,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )

        assert len(a_eq) == 0
        assert len(b_eq) == 0

        # Momentum <= 0.3
        np.testing.assert_array_almost_equal(a_ineq[0], np.array([0.5, 0.3, -0.2, 0.1]))
        assert b_ineq[0] == pytest.approx(0.3)

        # Value >= -0.1 becomes -Value <= 0.1
        np.testing.assert_array_almost_equal(
            a_ineq[1], -np.array([0.2, 0.4, 0.1, -0.3])
        )
        assert b_ineq[1] == pytest.approx(0.1)

    def test_factor_equality_constraints(self, groups, loading_matrix, factor_groups):
        """Test factor equality constraints."""
        equations = ["Momentum == 0", "Size = 0.2"]
        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=groups,
            equations=equations,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )

        assert len(a_ineq) == 0
        assert len(b_ineq) == 0

        # Momentum == 0
        np.testing.assert_array_almost_equal(a_eq[0], np.array([0.5, 0.3, -0.2, 0.1]))
        assert b_eq[0] == pytest.approx(0.0)

        # Size = 0.2
        np.testing.assert_array_almost_equal(a_eq[1], np.array([-0.1, 0.2, 0.5, 0.3]))
        assert b_eq[1] == pytest.approx(0.2)

    def test_mixed_asset_and_factor_constraints(
        self, groups, loading_matrix, factor_groups
    ):
        """Test mixing asset group and factor constraints."""
        equations = ["a <= 0.5", "Momentum <= 0.3", "b == 0.4"]
        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=groups,
            equations=equations,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )

        # Equality: b == 0.4
        assert len(a_eq) == 1
        np.testing.assert_array_almost_equal(a_eq[0], np.array([0.0, 0.0, 1.0, 1.0]))
        assert b_eq[0] == pytest.approx(0.4)

        # Inequalities: a <= 0.5, Momentum <= 0.3
        assert len(a_ineq) == 2
        np.testing.assert_array_almost_equal(a_ineq[0], np.array([1.0, 1.0, 0.0, 0.0]))
        assert b_ineq[0] == pytest.approx(0.5)
        np.testing.assert_array_almost_equal(a_ineq[1], np.array([0.5, 0.3, -0.2, 0.1]))
        assert b_ineq[1] == pytest.approx(0.3)

    def test_backward_compatibility_no_factors(self, groups):
        """Test that equations_to_matrix works without factor parameters."""
        equations = ["a <= 0.5", "b == 0.4"]
        a_eq, _b_eq, a_ineq, _ = equations_to_matrix(
            groups=groups,
            equations=equations,
        )

        assert len(a_eq) == 1
        assert len(a_ineq) == 1
        np.testing.assert_array_almost_equal(a_eq[0], np.array([0.0, 0.0, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(a_ineq[0], np.array([1.0, 1.0, 0.0, 0.0]))


class TestEquationsToMatrixFactorValidation:
    """Tests for validation errors in equations_to_matrix with factors."""

    def test_loading_matrix_wrong_shape_ndim(self, groups, factor_groups):
        """Test that 1D loading_matrix raises error."""
        loading_matrix = np.array([0.1, 0.2, 0.3])  # 1D instead of 2D
        with pytest.raises(ValueError, match="must be a 2D array"):
            equations_to_matrix(
                groups=groups,
                equations=["Momentum <= 0.3"],
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )

    def test_loading_matrix_wrong_n_assets(self, groups, factor_groups):
        """Test that loading_matrix with wrong n_assets raises error."""
        # groups has 4 assets, but loading_matrix has 3 rows
        loading_matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        with pytest.raises(ValueError, match="must have 4 rows"):
            equations_to_matrix(
                groups=groups,
                equations=["Momentum <= 0.3"],
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )

    def test_factor_groups_without_loading_matrix(self, groups, factor_groups):
        """Test that factor_groups without loading_matrix raises error."""
        with pytest.raises(
            ValueError, match=r"loading_matrix.*must be provided.*factor_groups"
        ):
            equations_to_matrix(
                groups=groups,
                equations=["a <= 0.5"],
                loading_matrix=None,
                factor_groups=factor_groups,
            )

    def test_factor_groups_loading_matrix_column_mismatch(self, groups):
        """Test that mismatched columns raises error."""
        loading_matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        factor_groups = np.array([["F1", "F2", "F3"]])  # 3 factors but 2 columns

        with pytest.raises(ValueError, match=r"columns.*must match"):
            equations_to_matrix(
                groups=groups,
                equations=["F1 <= 0.3"],
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )

    def test_collision_between_groups_and_factor_groups(self, loading_matrix):
        """Test that collision between groups and factor_groups raises error."""
        groups = np.array([["Momentum", "Other1", "Other2", "Other3"]])
        factor_groups = np.array([["Momentum", "Value", "Size"]])

        with pytest.raises(DuplicateGroupsError):
            equations_to_matrix(
                groups=groups,
                equations=["Momentum <= 0.3"],
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
            )

    def test_factor_constraint_raises_when_factor_not_found(
        self, groups, loading_matrix, factor_groups
    ):
        """Test that unknown factor in equation raises FactorNotFoundError."""
        with pytest.raises(GroupNotFoundError):
            equations_to_matrix(
                groups=groups,
                equations=["UnknownFactor <= 0.3"],
                loading_matrix=loading_matrix,
                factor_groups=factor_groups,
                raise_if_group_missing=True,
            )
