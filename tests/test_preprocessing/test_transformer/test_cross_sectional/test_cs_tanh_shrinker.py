from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.preprocessing import CSTanhShrinker


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

    def test_default_knee(self, X_with_outlier):
        out = CSTanhShrinker(knee=3.0).fit_transform(X_with_outlier)
        expected = np.array(
            [
                [1.124719, 2.01651569, 3.0, 3.98348431, 7.4478],
                [11.24719004, 20.1651569, 30.0, 39.8348431, 48.75280996],
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_estimation_weights(self, X_with_outlier):
        w = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        out = CSTanhShrinker(knee=3.0).fit_transform(X_with_outlier, cs_weights=w)
        expected = np.array(
            [
                [1.05439397, 2.0020956, 2.9979044, 3.94560603, 6.9478],
                [11.24719004, 20.1651569, 30.0, 39.8348431, 48.75280996],
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_nan_preservation(self):
        X = np.array([[1.0, np.nan, 3.0, 4.0, 100.0]])
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        expected = np.array(
            [
                [1.11079222, np.nan, 3.00093399, 3.99906601, 10.1717],
            ]
        )
        np.testing.assert_allclose(
            out[:, [0, 2, 3, 4]], expected[:, [0, 2, 3, 4]], rtol=1e-5
        )
        assert np.isnan(out[0, 1])


class TestProperties:
    """Behavioral properties that must always hold."""

    def test_monotonicity(self):
        X = np.array([[1, 2, 3, 4, 5, 50, 100, 200, 500]], dtype=float)
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        assert np.all(np.diff(out[0]) > 0)

    def test_constant_row_passthrough(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        np.testing.assert_array_equal(out, X)

    def test_single_finite_passthrough(self):
        X = np.array([[np.nan, np.nan, 7.0, np.nan, np.nan]])
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        assert out[0, 2] == 7.0
        assert np.all(np.isnan(out[0, [0, 1, 3, 4]]))

    def test_center_near_identity(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(1, 200))
        out = CSTanhShrinker(knee=4.0).fit_transform(X)
        mask = np.abs(X) < 1.0
        np.testing.assert_allclose(out[mask], X[mask], atol=0.05)

    def test_atol_threshold_is_independent_of_knee(self):
        X = np.array([[0.0, 0.0, 1e-9, 1e-9, 2e-9]])
        out_small_knee = CSTanhShrinker(knee=2.0, atol=5e-9).fit_transform(X)
        out_large_knee = CSTanhShrinker(knee=10.0, atol=5e-9).fit_transform(X)

        np.testing.assert_array_equal(out_small_knee, X)
        np.testing.assert_array_equal(out_large_knee, X)

    def test_outlier_is_shrunk(
        self,
    ):
        X = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        assert out[0, 4] < 100.0

    def test_preserves_scale(self):
        X = np.array([[1000.0, 2000.0, 3000.0, 4000.0, 5000.0]])
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        assert np.all(out > 500)

    def test_dataframe_input(self):
        X = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 100.0]],
            columns=["a", "b", "c", "d", "e"],
        )
        out = CSTanhShrinker(knee=3.0).fit_transform(X)
        assert out.shape == (1, 5)

    def test_multi_row_independence(self):
        X1 = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
        X2 = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 100.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        )
        t = CSTanhShrinker(knee=3.0)
        np.testing.assert_array_equal(t.fit_transform(X1)[0], t.fit_transform(X2)[0])


class TestValidation:
    """Parameter and input validation."""

    @pytest.mark.parametrize("knee", [0.0, np.nan, np.inf, -np.inf])
    def test_knee_must_be_finite_and_positive(self, knee):
        with pytest.raises(ValueError, match="knee"):
            CSTanhShrinker(knee=knee).fit_transform(np.ones((1, 5)))

    @pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, -np.inf])
    def test_atol_must_be_finite_and_non_negative(self, atol):
        with pytest.raises(ValueError, match="atol"):
            CSTanhShrinker(atol=atol).fit_transform(np.ones((1, 5)))

    def test_weights_nan(self):
        X = np.ones((1, 5))
        w = np.array([[1.0, np.nan, 1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="cs_weights"):
            CSTanhShrinker().fit_transform(X, cs_weights=w)

    def test_weights_negative(self):
        X = np.ones((1, 5))
        w = np.array([[1.0, -1.0, 1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="cs_weights"):
            CSTanhShrinker().fit_transform(X, cs_weights=w)

    def test_weights_shape_mismatch(self):
        X = np.ones((1, 5))
        w = np.ones((1, 3))
        with pytest.raises(ValueError, match="cs_weights"):
            CSTanhShrinker().fit_transform(X, cs_weights=w)

    def test_groups_argument_is_ignored(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
        cs_groups = np.array([[0, 0, 1, 1, 1]])

        shrinker = CSTanhShrinker()
        with_groups = shrinker.fit_transform(X, cs_groups=cs_groups)
        without_groups = shrinker.fit_transform(X)
        np.testing.assert_array_equal(with_groups, without_groups)

    def test_all_nan_row_raises(self):
        X = np.array(
            [[1.0, 2.0, 3.0, 4.0, 100.0], [np.nan, np.nan, np.nan, np.nan, np.nan]]
        )

        with pytest.raises(ValueError, match="estimation asset"):
            CSTanhShrinker().fit_transform(X)
