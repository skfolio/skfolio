from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from skfolio.preprocessing import CSGaussianRankScaler
from skfolio.typing import FloatArray, IntArray


def _midrank_percentile(sample: FloatArray, query: float) -> float:
    """Reference midrank percentile of `query` against `sample`, clipped to [0.5/n, 1 - 0.5/n]."""
    n = sample.size
    num_below = float((sample < query).sum())
    num_equal = float((sample == query).sum())
    raw = (num_below + 0.5 * num_equal) / n
    return float(np.clip(raw, 0.5 / n, 1.0 - 0.5 / n))


def _reference_percentile_rank(
    X: FloatArray,
    cs_weights: FloatArray | None = None,
    cs_groups: IntArray | None = None,
    min_group_size: int = 8,
) -> FloatArray:
    """Plain per-row, per-group percentile rank reference."""
    X = np.asarray(X, dtype=float)
    n_obs, _ = X.shape
    finite = np.isfinite(X)

    if cs_weights is None:
        weights = finite.astype(float)
    else:
        weights = np.where(finite, np.asarray(cs_weights, dtype=float), 0.0)

    estimation = finite & (weights > 0)
    out = np.full_like(X, np.nan)

    for t in range(n_obs):
        global_idx = np.flatnonzero(estimation[t])
        if global_idx.size == 0:
            raise ValueError("each observation needs at least one estimation asset")
        global_sample = X[t, global_idx]

        for i in np.flatnonzero(finite[t]):
            if cs_groups is None:
                sample = global_sample
            else:
                g = int(cs_groups[t, i])
                if g == -1:
                    sample = global_sample
                else:
                    members = (np.asarray(cs_groups[t]) == g) & estimation[t]
                    if int(members.sum()) < min_group_size:
                        sample = global_sample
                    else:
                        sample = X[t, members]
            out[t, i] = _midrank_percentile(sample, X[t, i])

    return out


def _reference_gaussian_rank(
    X: FloatArray,
    cs_weights: FloatArray | None = None,
    cs_groups: IntArray | None = None,
    min_group_size: int = 8,
    scale: bool = True,
    atol: float = 1e-12,
) -> FloatArray:
    """Plain reference: percentile rank, then ndtri, then recenter and optional rescale."""
    X = np.asarray(X, dtype=float)
    p = _reference_percentile_rank(
        X,
        cs_weights=cs_weights,
        cs_groups=cs_groups,
        min_group_size=min_group_size,
    )
    Z = ndtri(p)

    finite = np.isfinite(X)
    if cs_weights is None:
        weights = finite.astype(float)
    else:
        weights = np.where(finite, np.asarray(cs_weights, dtype=float), 0.0)
    estimation = finite & (weights > 0)

    out = np.full_like(Z, np.nan)
    for t in range(X.shape[0]):
        est = estimation[t]
        if not est.any():
            continue
        mu = float(np.average(Z[t, est], weights=weights[t, est]))
        centered = Z[t] - mu
        if not scale:
            out[t] = np.where(finite[t], centered, np.nan)
            continue
        n_est = int(est.sum())
        if n_est > 1:
            ss = float((centered[est] ** 2).sum())
            sigma = float(np.sqrt(ss / (n_est - 1)))
        else:
            sigma = 0.0
        if sigma > atol:
            out[t] = np.where(finite[t], centered / sigma, np.nan)
        else:
            out[t] = np.where(finite[t], 0.0, np.nan)
    return out


def _assert_centered_and_scaled(
    transformed: FloatArray,
    cs_weights: FloatArray,
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


class TestRegression:
    """Exact-match regression tests against frozen reference values."""

    def test_scale_false_matches_gaussianized_percentiles(self):
        X = np.array([[10.0, 20.0, 30.0, 40.0]])
        transformed = CSGaussianRankScaler(scale=False).fit_transform(X)
        expected = ndtri(np.array([[0.125, 0.375, 0.625, 0.875]]))
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_group_fallback_uses_global_rank(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_weights = np.array([[1.0, 1.0, 0.0, 0.0]])
        cs_groups = np.array([[0, 0, 1, 1]])

        transformed = CSGaussianRankScaler(
            min_group_size=2,
            scale=False,
        ).fit_transform(
            X,
            cs_weights=cs_weights,
            cs_groups=cs_groups,
        )

        expected = ndtri(np.array([[0.25, 0.75, 0.75, 0.75]]))
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)


class TestProperties:
    """Behavioral properties that must always hold."""

    def test_constant_row_maps_to_zero(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        transformed = CSGaussianRankScaler().fit_transform(X)
        np.testing.assert_array_equal(transformed, np.zeros_like(X))

    def test_constant_row_with_scale_false_also_maps_to_zero(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        transformed = CSGaussianRankScaler(scale=False).fit_transform(X)
        np.testing.assert_allclose(transformed, np.zeros_like(X), atol=1e-12)

    def test_estimation_universe_is_centered_and_scaled(self):
        X = np.array([[10.0, 20.0, 30.0, 40.0], [4.0, 1.0, 3.0, 7.0]])
        cs_weights = np.array([[1.0, 1.0, 1.0, 1.0], [0.2, 0.3, 0.5, 0.0]])

        transformed = CSGaussianRankScaler().fit_transform(X, cs_weights=cs_weights)

        _assert_centered_and_scaled(transformed, cs_weights)

    def test_scale_false_recenters_to_weighted_zero_mean(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 30))
        cs_weights = rng.random(X.shape) + 0.1

        transformed = CSGaussianRankScaler(scale=False).fit_transform(
            X, cs_weights=cs_weights
        )

        for row_idx in range(X.shape[0]):
            est = cs_weights[row_idx] > 0
            weighted_mean = np.average(
                transformed[row_idx, est], weights=cs_weights[row_idx, est]
            )
            np.testing.assert_allclose(weighted_mean, 0.0, atol=1e-12)

    def test_zero_weight_assets_still_receive_output(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_weights = np.array([[1.0, 1.0, 0.0, 0.0]])

        transformed = CSGaussianRankScaler(scale=False).fit_transform(
            X,
            cs_weights=cs_weights,
        )

        assert np.all(np.isfinite(transformed))

    def test_output_is_finite_on_estimation_universe(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((6, 50))
        X[rng.random(X.shape) < 0.10] = np.nan

        transformed = CSGaussianRankScaler().fit_transform(X)

        finite_in = np.isfinite(X)
        assert np.all(np.isfinite(transformed[finite_in]))
        assert np.all(np.isnan(transformed[~finite_in]))

    def test_nan_preservation(self):
        X = np.array([[1.0, np.nan, 3.0, 5.0]])
        transformed = CSGaussianRankScaler(scale=False).fit_transform(X)

        assert np.isnan(transformed[0, 1])
        np.testing.assert_allclose(
            transformed[0, [0, 2, 3]],
            ndtri(np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])),
            rtol=1e-12,
        )

    def test_multi_row_independence(self):
        X_single = np.array([[1.0, 2.0, 4.0, 10.0]])
        X_multi = np.array([[1.0, 2.0, 4.0, 10.0], [4.0, 1.0, 3.0, 7.0]])
        scaler = CSGaussianRankScaler()

        np.testing.assert_array_equal(
            scaler.fit_transform(X_single)[0],
            scaler.fit_transform(X_multi)[0],
        )

    def test_all_missing_groups_match_no_groups_path(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((4, 25))
        cs_groups = np.full(X.shape, -1, dtype=int)

        with_groups = CSGaussianRankScaler(min_group_size=2).fit_transform(
            X, cs_groups=cs_groups
        )
        without_groups = CSGaussianRankScaler().fit_transform(X)

        np.testing.assert_allclose(with_groups, without_groups, rtol=1e-12)


class TestGroupedFallback:
    """Cover the count-based and missing-group fallback paths explicitly."""

    def test_grouped_happy_path_without_fallback(self):
        X = np.array([[1.0, 3.0, 5.0, 10.0, 12.0, 14.0]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        transformed = CSGaussianRankScaler(min_group_size=3, scale=False).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_gaussian_rank(
            X, cs_groups=cs_groups, min_group_size=3, scale=False
        )

        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_min_group_size_one_disables_fallback(self):
        X = np.array([[1.0, 3.0, 10.0, 14.0, 22.0]])
        cs_groups = np.array([[0, 0, 1, 1, 2]])

        transformed = CSGaussianRankScaler(min_group_size=1, scale=False).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_gaussian_rank(
            X, cs_groups=cs_groups, min_group_size=1, scale=False
        )

        np.testing.assert_allclose(transformed, expected, rtol=1e-12)
        # Single-asset group resolves to the neutral midrank, and ndtri(0.5) == 0.
        np.testing.assert_allclose(transformed[0, 4], 0.0, atol=1e-12)

    def test_nan_inside_group_reduces_count_and_triggers_fallback(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0, np.nan, 14.0]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        transformed = CSGaussianRankScaler(min_group_size=3, scale=False).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_gaussian_rank(
            X, cs_groups=cs_groups, min_group_size=3, scale=False
        )

        assert np.isnan(transformed[0, 4])
        finite_mask = ~np.isnan(transformed)
        np.testing.assert_allclose(
            transformed[finite_mask], expected[finite_mask], rtol=1e-12
        )

    def test_all_nan_group_falls_back_and_preserves_nan(self):
        X = np.array([[1.0, 2.0, 4.0, np.nan, np.nan, np.nan]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        transformed = CSGaussianRankScaler(min_group_size=2, scale=False).fit_transform(
            X, cs_groups=cs_groups
        )

        assert np.all(np.isnan(transformed[0, 3:]))
        np.testing.assert_allclose(
            transformed[0, :3],
            ndtri(np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])),
            rtol=1e-12,
        )

    def test_nullable_integer_groups_are_accepted(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_groups_int = np.array([[0, 0, 1, -1]])
        cs_groups_nullable = pd.DataFrame([[0, 0, 1, -1]], dtype="Int64")

        scaler = CSGaussianRankScaler(min_group_size=2, scale=False)
        from_int = scaler.fit_transform(X, cs_groups=cs_groups_int)
        from_nullable = scaler.fit_transform(X, cs_groups=cs_groups_nullable)

        np.testing.assert_allclose(from_nullable, from_int, rtol=1e-12)


class TestStatisticalContract:
    """Numerical contracts that protect against silent statistical bugs."""

    def test_grouped_invariant_to_group_relabeling(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 30))
        cs_groups = rng.integers(0, 4, size=X.shape)

        relabel = np.array([3, 1, 0, 2])
        cs_groups_relabeled = relabel[cs_groups]

        scaler = CSGaussianRankScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_groups=cs_groups)
        b = scaler.fit_transform(X, cs_groups=cs_groups_relabeled)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_invariant_to_asset_permutation(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 20))
        cs_groups = rng.integers(0, 3, size=X.shape)
        cs_weights = rng.random(X.shape) + 0.1

        perm = rng.permutation(X.shape[1])
        scaler = CSGaussianRankScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
        b = scaler.fit_transform(
            X[:, perm], cs_weights=cs_weights[:, perm], cs_groups=cs_groups[:, perm]
        )
        np.testing.assert_allclose(a[:, perm], b, rtol=1e-12)

    def test_grouped_cross_row_independence(self):
        X_a = np.array([[1.0, 3.0, 10.0, 14.0]])
        X_b = np.array([[100.0, -50.0, 7.0, 9.0]])
        cs_groups_a = np.array([[0, 0, 1, 1]])
        cs_groups_b = np.array([[1, 1, 0, 0]])

        X_multi = np.vstack([X_a, X_b])
        cs_groups_multi = np.vstack([cs_groups_a, cs_groups_b])

        scaler = CSGaussianRankScaler(min_group_size=2)
        out_multi = scaler.fit_transform(X_multi, cs_groups=cs_groups_multi)
        out_a = scaler.fit_transform(X_a, cs_groups=cs_groups_a)
        out_b = scaler.fit_transform(X_b, cs_groups=cs_groups_b)

        np.testing.assert_allclose(out_multi[0], out_a[0], rtol=1e-12)
        np.testing.assert_allclose(out_multi[1], out_b[0], rtol=1e-12)

    def test_grouped_sparse_group_labels(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0, 14.0, 18.0]])
        cs_groups_dense = np.array([[0, 0, 0, 1, 1, 1]])
        cs_groups_sparse = np.array([[7, 7, 7, 999_999, 999_999, 999_999]])

        scaler = CSGaussianRankScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_groups=cs_groups_dense)
        b = scaler.fit_transform(X, cs_groups=cs_groups_sparse)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_object_dtype_groups_accepted(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0, 14.0, 18.0]])
        cs_groups_int = np.array([[0, 0, 0, 1, 1, 1]])
        cs_groups_obj = np.array([[0, 0, 0, 1, 1, 1]], dtype=object)

        scaler = CSGaussianRankScaler(min_group_size=2)
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

        for t in range(n_obs):
            if not (np.isfinite(X[t]) & (cs_weights[t] > 0)).any():
                X[t, 0] = 1.0
                cs_weights[t, 0] = 1.0
                cs_groups[t, 0] = 0

        actual = CSGaussianRankScaler(min_group_size=3).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        expected = _reference_gaussian_rank(
            X,
            cs_weights=cs_weights,
            cs_groups=cs_groups,
            min_group_size=3,
            scale=True,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_random_inputs_scale_false(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((5, 20))
        X[rng.random(X.shape) < 0.10] = np.nan
        cs_weights = rng.random(X.shape) + 0.05
        cs_groups = rng.integers(-1, 4, size=X.shape)

        actual = CSGaussianRankScaler(min_group_size=3, scale=False).fit_transform(
            X, cs_weights=cs_weights, cs_groups=cs_groups
        )
        expected = _reference_gaussian_rank(
            X,
            cs_weights=cs_weights,
            cs_groups=cs_groups,
            min_group_size=3,
            scale=False,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )

    def test_random_inputs_no_weights(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((4, 20))
        cs_groups = rng.integers(-1, 4, size=X.shape)

        actual = CSGaussianRankScaler(min_group_size=3).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_gaussian_rank(X, cs_groups=cs_groups, min_group_size=3)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )

    def test_random_inputs_no_groups(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 15))
        X[rng.random(X.shape) < 0.10] = np.nan
        cs_weights = rng.random(X.shape)

        actual = CSGaussianRankScaler().fit_transform(X, cs_weights=cs_weights)
        expected = _reference_gaussian_rank(X, cs_weights=cs_weights)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )


class TestValidation:
    """Parameter and input validation."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_group_size": 0}, "min_group_size"),
            ({"min_group_size": 1.5}, "min_group_size"),
            ({"min_group_size": np.nan}, "min_group_size"),
            ({"min_group_size": np.inf}, "min_group_size"),
            ({"min_group_size": -np.inf}, "min_group_size"),
            ({"min_group_size": True}, "min_group_size"),
            ({"atol": -1.0}, "atol"),
            ({"atol": np.nan}, "atol"),
            ({"atol": np.inf}, "atol"),
            ({"atol": -np.inf}, "atol"),
        ],
    )
    def test_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CSGaussianRankScaler(**kwargs).fit_transform(np.ones((1, 4)))

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
            CSGaussianRankScaler().fit_transform(X, cs_weights=cs_weights)

    def test_groups_must_be_integer(self):
        X = np.ones((1, 4))
        cs_groups = np.array([[0.0, 0.0, 1.0, 1.0]])

        with pytest.raises(ValueError, match="integer array"):
            CSGaussianRankScaler().fit_transform(X, cs_groups=cs_groups)

    def test_groups_shape_mismatch(self):
        X = np.ones((1, 4))
        cs_groups = np.ones((1, 3), dtype=int)

        with pytest.raises(ValueError, match="same shape"):
            CSGaussianRankScaler().fit_transform(X, cs_groups=cs_groups)

    def test_groups_must_be_greater_than_minus_one(self):
        X = np.ones((1, 4))
        cs_groups = np.array([[0, -2, 1, 1]])

        with pytest.raises(ValueError, match=">= -1"):
            CSGaussianRankScaler().fit_transform(X, cs_groups=cs_groups)

    def test_empty_estimation_universe_raises(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        cs_weights = np.zeros_like(X)

        with pytest.raises(ValueError, match="estimation asset"):
            CSGaussianRankScaler().fit_transform(X, cs_weights=cs_weights)
