import numpy as np
import pytest

from skfolio.preprocessing import CSStandardScaler


@pytest.fixture
def grouped_inputs():
    X = np.array([[1.0, 2.0, 4.0, 10.0], [4.0, 1.0, 3.0, 7.0]])
    cs_weights = np.array([[0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]])
    cs_groups = np.array([[0, 0, 1, 1], [0, 0, 1, -1]])
    return X, cs_weights, cs_groups


def _assert_standardized_estimation_universe(
    transformed: np.ndarray, cs_weights: np.ndarray
) -> None:
    est_mask = np.isfinite(transformed) & (cs_weights > 0)

    for row_idx in range(transformed.shape[0]):
        values = transformed[row_idx, est_mask[row_idx]]
        weights = cs_weights[row_idx, est_mask[row_idx]]
        weighted_mean = np.average(values, weights=weights)
        equal_weighted_std = np.sqrt(
            np.sum((values - weighted_mean) ** 2) / (values.size - 1)
        )

        np.testing.assert_allclose(weighted_mean, 0.0, atol=1e-12)
        np.testing.assert_allclose(equal_weighted_std, 1.0, atol=1e-12)


def _eqw_bessel_std_around(values: np.ndarray, mean: float) -> float:
    """sqrt(sum((values - mean)^2) / (N - 1)) on a 1D estimation slice.

    Mirrors `_masked_equal_weighted_std` exactly: equal-weighted, Bessel-
    corrected, around an externally supplied (typically weighted) mean.
    """
    n = values.size
    if n <= 1:
        return 0.0
    residuals = values - mean
    return float(np.sqrt(np.sum(residuals * residuals) / (n - 1)))


def _reference_transform(
    X: np.ndarray,
    cs_weights: np.ndarray | None = None,
    cs_groups: np.ndarray | None = None,
    min_group_size: int = 8,
    atol: float = 1e-12,
) -> np.ndarray:
    """Plain per-row, per-group reference used for property tests.

    Mirrors the documented contract: weighted mean and unbiased equal-weighted
    std (around the weighted mean) on the estimation universe per (row, group),
    global fallback for missing-group and small-group cells, then global
    recenter to weighted mean zero and rescale by the estimation-universe
    second moment around zero with Bessel correction.
    """
    X = np.asarray(X, dtype=float)
    n_obs, n_assets = X.shape
    finite = np.isfinite(X)

    if cs_weights is None:
        weights = finite.astype(float)
    else:
        weights = np.where(finite, np.asarray(cs_weights, dtype=float), 0.0)

    estimation = finite & (weights > 0)
    out = np.full_like(X, np.nan, dtype=float)

    for t in range(n_obs):
        x_row, w_row = X[t], weights[t]
        est_row, finite_row = estimation[t], finite[t]
        est_idx = np.flatnonzero(est_row)
        if est_idx.size == 0:
            raise ValueError("each observation needs at least one estimation asset")

        mu_global = float(np.average(x_row[est_idx], weights=w_row[est_idx]))
        sd_global = _eqw_bessel_std_around(x_row[est_idx], mu_global)

        mu_cell = np.full(n_assets, mu_global, dtype=float)
        sd_cell = np.full(n_assets, sd_global, dtype=float)

        if cs_groups is not None:
            g_row = np.asarray(cs_groups[t], dtype=int)
            for g in np.unique(g_row[g_row != -1]):
                members = (g_row == g) & est_row
                cnt = int(members.sum())
                if cnt < min_group_size:
                    continue
                cell_mask = g_row == g
                mu_g = float(np.average(x_row[members], weights=w_row[members]))
                mu_cell[cell_mask] = mu_g
                sd_cell[cell_mask] = _eqw_bessel_std_around(x_row[members], mu_g)

        z = np.zeros(n_assets, dtype=float)
        good = finite_row & (sd_cell > atol)
        z[good] = (x_row[good] - mu_cell[good]) / sd_cell[good]
        z = np.where(finite_row, z, np.nan)

        if cs_groups is None:
            out[t] = z
            continue

        mu_bmk = float(np.average(z[est_idx], weights=w_row[est_idx]))
        z_centered = z - mu_bmk
        # Implementation: sqrt(sum(centered^2)/(N-1)) -- second moment around 0
        # with Bessel correction, NOT centered around the equal-weighted mean.
        sd_global_final = _eqw_bessel_std_around(z_centered[est_idx], 0.0)
        z_final = np.zeros(n_assets, dtype=float)
        if sd_global_final > atol:
            z_final = z_centered / sd_global_final
        out[t] = np.where(finite_row, z_final, np.nan)

    return out


class TestRegression:
    """Exact-match regression tests against frozen reference values."""

    def test_global_standardization(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0], [4.0, 1.0, 3.0, 7.0]])
        transformed = CSStandardScaler().fit_transform(X)
        expected = np.array(
            [
                [
                    -0.80622577482985,
                    -0.55815630565144,
                    -0.06201736729460,
                    1.42639944777590,
                ],
                [0.1, -1.1, -0.3, 1.3],
            ]
        )
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_grouped_standardization_with_weights_and_missing_group(
        self, grouped_inputs
    ):
        X, cs_weights, cs_groups = grouped_inputs
        transformed = CSStandardScaler(min_group_size=2).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        expected = np.array(
            [
                [
                    -0.70710678118655,
                    0.70710678118655,
                    3.53553390593274,
                    12.02081528017131,
                ],
                [
                    1.00250132323966,
                    -0.97982707108555,
                    0.18689571335547,
                    3.92480998046478,
                ],
            ]
        )
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)


class TestProperties:
    """Behavioral properties that must always hold."""

    def test_estimation_universe_is_centered_and_scaled(self, grouped_inputs):
        X, cs_weights, cs_groups = grouped_inputs
        transformed = CSStandardScaler(min_group_size=2).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        _assert_standardized_estimation_universe(transformed, cs_weights)

    def test_grouped_weighted_path_without_fallback(self):
        X = np.array([[1.0, 3.0, 10.0, 14.0]])
        cs_weights = np.array([[0.2, 0.8, 0.3, 0.7]])
        cs_groups = np.array([[0, 0, 1, 1]])

        transformed = CSStandardScaler(min_group_size=2).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )

        assert np.all(np.isfinite(transformed))
        _assert_standardized_estimation_universe(transformed, cs_weights)

    def test_grouped_path_without_weights(self):
        X = np.array([[1.0, 3.0, 10.0, 14.0]])
        cs_groups = np.array([[0, 0, 1, 1]])

        transformed = CSStandardScaler(min_group_size=2).fit_transform(
            X, cs_groups=cs_groups
        )

        assert np.all(np.isfinite(transformed))
        _assert_standardized_estimation_universe(
            transformed, np.isfinite(X).astype(np.float64)
        )

    def test_grouped_nan_reduces_group_count_and_preserves_nan(self):
        X = np.array([[1.0, np.nan, 10.0, 14.0]])
        cs_weights = np.array([[0.2, 0.8, 0.3, 0.7]])
        cs_groups = np.array([[0, 0, 1, 1]])

        transformed = CSStandardScaler(min_group_size=2).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )

        assert np.isnan(transformed[0, 1])
        assert np.all(np.isfinite(transformed[0, [0, 2, 3]]))
        _assert_standardized_estimation_universe(transformed, cs_weights)

    def test_grouped_zero_weight_cells_do_not_affect_estimation_assets(self):
        X_left = np.array([[1.0, 3.0, 10.0, 14.0]])
        X_right = np.array([[1.0, 3.0, -500.0, 800.0]])
        cs_weights = np.array([[0.4, 0.6, 0.0, 0.0]])
        cs_groups = np.array([[0, 0, 0, 0]])

        scaler = CSStandardScaler(min_group_size=2)
        transformed_left = scaler.fit_transform(
            X_left, cs_weights=cs_weights, cs_groups=cs_groups
        )
        transformed_right = scaler.fit_transform(
            X_right, cs_weights=cs_weights, cs_groups=cs_groups
        )

        np.testing.assert_allclose(
            transformed_left[0, cs_weights[0] > 0],
            transformed_right[0, cs_weights[0] > 0],
            rtol=1e-12,
        )

    def test_all_missing_groups_match_global_standardization(self):
        X = np.array([[1.0, 3.0, 10.0, 14.0]])
        cs_weights = np.array([[0.2, 0.8, 0.3, 0.7]])
        cs_groups = np.full(X.shape, -1, dtype=int)

        transformed = CSStandardScaler(min_group_size=2).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        expected = CSStandardScaler().fit_transform(X, cs_weights=cs_weights)

        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_nan_preservation(self):
        X = np.array([[1.0, np.nan, 4.0, 10.0]])
        transformed = CSStandardScaler().fit_transform(X)

        assert np.isnan(transformed[0, 1])
        np.testing.assert_allclose(
            transformed[0, [0, 2, 3]],
            np.array([-0.87287156094397, -0.21821789023599, 1.09108945117996]),
            rtol=1e-12,
        )

    def test_multi_row_independence(self):
        X_single = np.array([[1.0, 2.0, 4.0, 10.0]])
        X_multi = np.array([[1.0, 2.0, 4.0, 10.0], [4.0, 1.0, 3.0, 7.0]])
        scaler = CSStandardScaler()

        np.testing.assert_array_equal(
            scaler.fit_transform(X_single)[0], scaler.fit_transform(X_multi)[0]
        )

    def test_zero_weight_assets_still_receive_output(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_weights = np.array([[0.5, 0.5, 0.0, 0.0]])

        transformed = CSStandardScaler().fit_transform(X, cs_weights=cs_weights)

        np.testing.assert_allclose(
            transformed,
            np.array(
                [
                    [
                        -0.70710678118655,
                        0.70710678118655,
                        3.53553390593274,
                        12.02081528017131,
                    ]
                ]
            ),
            rtol=1e-12,
        )

    def test_constant_row_returns_zeros(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        transformed = CSStandardScaler().fit_transform(X)
        np.testing.assert_array_equal(transformed, np.zeros_like(X))


class TestGroupedFallback:
    """Cover the count-based and missing-group fallback paths explicitly."""

    def test_count_based_fallback_uses_global_stats(self):
        """A valid group with too few estimation members falls back to global."""
        # Group 0 has 2 members, group 1 has 6. With min_group_size=4 only
        # group 0 falls back to global stats.
        X = np.array([[1.0, 2.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]])
        cs_groups = np.array([[0, 0, 1, 1, 1, 1, 1, 1]])

        actual = CSStandardScaler(min_group_size=4).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=4)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_count_based_fallback_disabled_with_min_group_size_one(self):
        """With min_group_size=1 every valid group keeps its own stats."""
        X = np.array([[1.0, 3.0, 10.0, 14.0, 22.0]])
        cs_groups = np.array([[0, 0, 1, 1, 2]])

        actual = CSStandardScaler(min_group_size=1).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_mixed_fallback_and_valid_groups_match_reference(self):
        """Some groups fall back, others use their own stats; weights present."""
        X = np.array([[1.0, 2.0, 3.0, 10.0, 12.0, 14.0, 16.0, 18.0]])
        cs_weights = np.array([[0.1, 0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1, 1, 1]])

        actual = CSStandardScaler(min_group_size=4).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        expected = _reference_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups, min_group_size=4
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_all_nan_group_falls_back_and_preserves_nan(self):
        """A group whose members are all NaN has count=0 and falls back."""
        X = np.array([[1.0, 2.0, 4.0, np.nan, np.nan, np.nan]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        out = CSStandardScaler(min_group_size=2).fit_transform(X, cs_groups=cs_groups)
        assert np.all(np.isnan(out[0, 3:]))
        assert np.all(np.isfinite(out[0, :3]))
        _assert_standardized_estimation_universe(out, np.isfinite(X).astype(np.float64))

    def test_constant_group_yields_zero_within_step(self):
        """A constant group above min_group_size keeps z=0 after the within step."""
        # Group 0 is constant at 5.0 with 3 members; group 1 has dispersion.
        # Within: group 0 std = 0 -> cells become 0; group 1 -> [-1, 0, 1].
        # Global recenter: weighted mean of [0, 0, 0, -1, 0, 1] is 0.
        # Global EW std (ddof=1): SS=2 over N-1=5 -> sqrt(2/5).
        X = np.array([[5.0, 5.0, 5.0, 0.0, 2.0, 4.0]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        out = CSStandardScaler(min_group_size=3).fit_transform(X, cs_groups=cs_groups)
        s = np.sqrt(2.0 / 5.0)
        expected = np.array([[0.0, 0.0, 0.0, -1.0 / s, 0.0, 1.0 / s]])
        np.testing.assert_allclose(out, expected, rtol=1e-12)


class TestStatisticalContract:
    """Numerical contracts that protect against silent statistical bugs."""

    def test_bessel_correction_used_in_within_group_std(self):
        """Within-group std uses N-1 (sample), not N (population).

        Two groups of different size make the Bessel correction observable
        through the global rescale. A regression to the population variant
        would change every output.
        """
        # Group A = [0, 2, 4] (size 3, sample std=2);
        # Group B = [10, 14] (size 2, sample std=2*sqrt(2)).
        X = np.array([[0.0, 2.0, 4.0, 10.0, 14.0]])
        cs_groups = np.array([[0, 0, 0, 1, 1]])

        out = CSStandardScaler(min_group_size=2).fit_transform(X, cs_groups=cs_groups)
        # Within-group z then global EW (ddof=1) rescale by sqrt(3)/2.
        expected = np.array(
            [
                [
                    -2.0 / np.sqrt(3.0),
                    0.0,
                    2.0 / np.sqrt(3.0),
                    -np.sqrt(2.0 / 3.0),
                    np.sqrt(2.0 / 3.0),
                ]
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_grouped_invariant_to_group_relabeling(self):
        """Permuting group labels does not change the output."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 30))
        cs_groups = rng.integers(0, 4, size=X.shape)
        cs_weights = rng.random(X.shape) + 0.1

        relabel = np.array([3, 1, 0, 2])
        cs_groups_relabeled = relabel[cs_groups]

        scaler = CSStandardScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
        b = scaler.fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups_relabeled
        )
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_invariant_to_asset_permutation(self):
        """Permuting assets permutes the output identically."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 20))
        cs_groups = rng.integers(0, 3, size=X.shape)
        cs_weights = rng.random(X.shape) + 0.1

        perm = rng.permutation(X.shape[1])
        scaler = CSStandardScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
        b = scaler.fit_transform(
            X[:, perm], cs_weights=cs_weights[:, perm], cs_groups=cs_groups[:, perm]
        )
        np.testing.assert_allclose(a[:, perm], b, rtol=1e-12)

    def test_grouped_invariant_to_per_row_weight_scaling(self):
        """Scaling row weights by a positive constant does not change output."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((4, 16))
        cs_groups = rng.integers(0, 3, size=X.shape)
        cs_weights = rng.random(X.shape) + 0.1
        scales = np.array([[1.0], [2.5], [0.1], [37.0]])

        scaler = CSStandardScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
        b = scaler.fit_transform(X, cs_weights=cs_weights * scales, cs_groups=cs_groups)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_cross_row_independence(self):
        """The composite-key bincount must not leak across rows."""
        X_a = np.array([[1.0, 3.0, 10.0, 14.0]])
        X_b = np.array([[100.0, -50.0, 7.0, 9.0]])
        cs_groups_a = np.array([[0, 0, 1, 1]])
        cs_groups_b = np.array([[1, 1, 0, 0]])

        X_multi = np.vstack([X_a, X_b])
        cs_groups_multi = np.vstack([cs_groups_a, cs_groups_b])

        scaler = CSStandardScaler(min_group_size=2)
        out_multi = scaler.fit_transform(X_multi, cs_groups=cs_groups_multi)
        out_a = scaler.fit_transform(X_a, cs_groups=cs_groups_a)
        out_b = scaler.fit_transform(X_b, cs_groups=cs_groups_b)

        np.testing.assert_allclose(out_multi[0], out_a[0], rtol=1e-12)
        np.testing.assert_allclose(out_multi[1], out_b[0], rtol=1e-12)

    def test_grouped_sparse_group_labels(self):
        """Sparse integer labels are remapped without affecting the output."""
        X = np.array([[1.0, 2.0, 4.0, 10.0, 14.0, 18.0]])
        cs_groups_dense = np.array([[0, 0, 0, 1, 1, 1]])
        cs_groups_sparse = np.array([[7, 7, 7, 999_999, 999_999, 999_999]])

        scaler = CSStandardScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_groups=cs_groups_dense)
        b = scaler.fit_transform(X, cs_groups=cs_groups_sparse)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_object_dtype_groups_accepted(self):
        """Object arrays of Python ints behave like an int array."""
        X = np.array([[1.0, 2.0, 4.0, 10.0, 14.0, 18.0]])
        cs_groups_int = np.array([[0, 0, 0, 1, 1, 1]])
        cs_groups_obj = np.array([[0, 0, 0, 1, 1, 1]], dtype=object)

        scaler = CSStandardScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_groups=cs_groups_int)
        b = scaler.fit_transform(X, cs_groups=cs_groups_obj)
        np.testing.assert_allclose(a, b, rtol=1e-12)


class TestReferenceEquivalence:
    """Compare against a plain Python reference on randomized inputs."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 2024])
    def test_random_inputs_with_weights_groups_and_nans(self, seed):
        rng = np.random.default_rng(seed)
        n_obs, n_assets = 6, 25

        X = rng.standard_normal((n_obs, n_assets))
        X[rng.random(X.shape) < 0.10] = np.nan

        cs_weights = rng.random(X.shape)
        cs_weights[rng.random(X.shape) < 0.10] = 0.0

        cs_groups = rng.integers(0, 5, size=X.shape)
        cs_groups[rng.random(X.shape) < 0.05] = -1

        # Guarantee at least one estimation asset per row.
        for t in range(n_obs):
            if not (np.isfinite(X[t]) & (cs_weights[t] > 0)).any():
                X[t, 0] = 1.0
                cs_weights[t, 0] = 1.0
                cs_groups[t, 0] = 0

        actual = CSStandardScaler(min_group_size=3).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        expected = _reference_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups, min_group_size=3
        )
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )

    def test_random_inputs_no_weights(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((4, 20))
        cs_groups = rng.integers(-1, 4, size=X.shape)

        actual = CSStandardScaler(min_group_size=3).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=3)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )


class TestValidation:
    """Parameter and input validation."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_group_size": 0}, "min_group_size"),
            ({"atol": -1.0}, "atol"),
        ],
    )
    def test_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CSStandardScaler(**kwargs).fit_transform(np.ones((1, 4)))

    def test_groups_must_be_integer(self):
        X = np.ones((1, 4))
        cs_groups = np.array([[0.0, 0.0, 1.0, 1.0]])

        with pytest.raises(ValueError, match="integer array"):
            CSStandardScaler().fit_transform(X, cs_groups=cs_groups)

    def test_groups_shape_mismatch(self):
        X = np.ones((1, 4))
        cs_groups = np.ones((1, 3), dtype=int)

        with pytest.raises(ValueError, match="same shape"):
            CSStandardScaler().fit_transform(X, cs_groups=cs_groups)

    def test_groups_must_be_greater_than_minus_one(self):
        X = np.ones((1, 4))
        cs_groups = np.array([[0, -2, 1, 1]])

        with pytest.raises(ValueError, match=">= -1"):
            CSStandardScaler().fit_transform(X, cs_groups=cs_groups)

    @pytest.mark.parametrize(
        ("cs_weights", "match"),
        [
            (np.array([[0.5, -0.1, 0.3, 0.3]]), "non-negative"),
            (np.array([[0.5, np.nan, 0.3, 0.3]]), "NaN"),
            (np.ones((1, 3)), "same shape"),
        ],
    )
    def test_invalid_weights(self, cs_weights, match):
        X = np.ones((1, 4))

        with pytest.raises(ValueError, match=match):
            CSStandardScaler().fit_transform(X, cs_weights=cs_weights)

    def test_empty_estimation_universe_raises(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        cs_weights = np.zeros_like(X)

        with pytest.raises(ValueError, match="estimation asset"):
            CSStandardScaler().fit_transform(X, cs_weights=cs_weights)
