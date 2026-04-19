import numpy as np
import pytest

from skfolio.preprocessing._transformer._cross_sectional import CSWinsorizer


@pytest.fixture
def X_with_outlier():
    return np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 100.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ]
    )


class TestRegression:
    """Exact-match regression tests against frozen reference values."""

    def test_default_percentiles(self, X_with_outlier):
        transformed = CSWinsorizer(low=0.2, high=0.8).fit_transform(X_with_outlier)
        expected = np.array(
            [
                [1.8, 2.0, 3.0, 4.0, 23.2],
                [18.0, 20.0, 30.0, 40.0, 42.0],
            ]
        )
        np.testing.assert_allclose(transformed, expected, rtol=1e-6)

    def test_estimation_weights(self, X_with_outlier):
        cs_weights = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        transformed = CSWinsorizer(low=0.2, high=0.8).fit_transform(
            X_with_outlier, cs_weights=cs_weights
        )
        expected = np.array(
            [
                [1.6, 2.0, 3.0, 3.4, 3.4],
                [18.0, 20.0, 30.0, 40.0, 42.0],
            ]
        )
        np.testing.assert_allclose(transformed, expected, rtol=1e-6)

    def test_nan_preservation(self):
        X = np.array([[1.0, np.nan, 3.0, 4.0, 100.0]])
        transformed = CSWinsorizer(low=0.2, high=0.8).fit_transform(X)
        expected = np.array([[2.2, np.nan, 3.0, 4.0, 42.4]])

        np.testing.assert_allclose(
            transformed[0, [0, 2, 3, 4]], expected[0, [0, 2, 3, 4]], rtol=1e-6
        )
        assert np.isnan(transformed[0, 1])

    def test_all_nan_row_raises(self):
        X = np.array(
            [[1.0, np.nan, 3.0, 4.0, 100.0], [np.nan, np.nan, np.nan, np.nan, np.nan]]
        )

        with pytest.raises(ValueError, match="estimation asset"):
            CSWinsorizer(low=0.2, high=0.8).fit_transform(X)


class TestProperties:
    """Behavioral properties that must always hold."""

    def test_does_not_mutate_input(self, X_with_outlier):
        X_before = X_with_outlier.copy()
        transformed = CSWinsorizer(low=0.2, high=0.8).fit_transform(X_with_outlier)

        np.testing.assert_array_equal(X_with_outlier, X_before)
        assert not np.shares_memory(transformed, X_with_outlier)

    def test_constant_row_passthrough(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        transformed = CSWinsorizer(low=0.2, high=0.8).fit_transform(X)
        np.testing.assert_array_equal(transformed, X)

    def test_single_finite_passthrough(self):
        X = np.array([[np.nan, np.nan, 7.0, np.nan, np.nan]])
        transformed = CSWinsorizer(low=0.2, high=0.8).fit_transform(X)
        assert transformed[0, 2] == 7.0
        assert np.all(np.isnan(transformed[0, [0, 1, 3, 4]]))

    def test_groups_argument_is_ignored(self, X_with_outlier):
        cs_groups = np.array([[0, 0, 1, 1, 1], [1, 1, 0, 0, -1]])
        winsorizer = CSWinsorizer(low=0.2, high=0.8)

        with_groups = winsorizer.fit_transform(X_with_outlier, cs_groups=cs_groups)
        without_groups = winsorizer.fit_transform(X_with_outlier)
        np.testing.assert_array_equal(with_groups, without_groups)

    def test_multi_row_independence(self):
        X_single = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
        X_multi = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 100.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        )
        winsorizer = CSWinsorizer(low=0.2, high=0.8)

        np.testing.assert_array_equal(
            winsorizer.fit_transform(X_single)[0],
            winsorizer.fit_transform(X_multi)[0],
        )


class TestValidation:
    """Parameter and input validation."""

    @pytest.mark.parametrize(
        ("low", "high"),
        [
            (-0.1, 0.8),
            (0.2, 1.1),
            (0.8, 0.2),
        ],
    )
    def test_invalid_percentile_bounds(self, low, high):
        with pytest.raises(ValueError, match="low"):
            CSWinsorizer(low=low, high=high).fit_transform(np.ones((1, 5)))

    def test_weights_shape_mismatch(self):
        X = np.ones((1, 5))
        cs_weights = np.ones((1, 3))

        with pytest.raises(ValueError, match="cs_weights"):
            CSWinsorizer().fit_transform(X, cs_weights=cs_weights)
