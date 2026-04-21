from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.preprocessing import CSPercentileRankScaler
from skfolio.typing import FloatArray, IntArray


def _midrank_percentile(sample: FloatArray, query: float) -> float:
    """Reference midrank percentile of `query` against `sample`, clipped to [0.5/n, 1 - 0.5/n]."""
    n = sample.size
    num_below = float((sample < query).sum())
    num_equal = float((sample == query).sum())
    raw = (num_below + 0.5 * num_equal) / n
    return float(np.clip(raw, 0.5 / n, 1.0 - 0.5 / n))


def _reference_transform(
    X: FloatArray,
    cs_weights: FloatArray | None = None,
    cs_groups: IntArray | None = None,
    min_group_size: int = 8,
) -> FloatArray:
    """Plain per-row, per-group reference used for property tests."""
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


class TestRegression:
    """Exact-match regression tests against frozen reference values."""

    def test_global_percentile_rank(self):
        X = np.array([[10.0, 20.0, 30.0, 40.0]])
        transformed = CSPercentileRankScaler().fit_transform(X)
        expected = np.array([[0.125, 0.375, 0.625, 0.875]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_ties_use_average_rank_percentiles(self):
        X = np.array([[1.0, 1.0, 2.0, 2.0]])
        transformed = CSPercentileRankScaler().fit_transform(X)
        expected = np.array([[0.25, 0.25, 0.75, 0.75]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_constant_row_maps_to_neutral_percentile(self):
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        transformed = CSPercentileRankScaler().fit_transform(X)
        np.testing.assert_allclose(transformed, 0.5, rtol=1e-12)


class TestProperties:
    """Behavioral properties that must always hold."""

    def test_group_fallback_uses_global_percentile(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_weights = np.array([[1.0, 1.0, 0.0, 0.0]])
        cs_groups = np.array([[0, 0, 1, 1]])

        transformed = CSPercentileRankScaler(min_group_size=2).fit_transform(
            X,
            cs_weights=cs_weights,
            cs_groups=cs_groups,
        )

        expected = np.array([[0.25, 0.75, 0.75, 0.75]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_missing_group_falls_back_to_global_percentile(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_groups = np.array([[0, 0, 1, -1]])

        transformed = CSPercentileRankScaler(min_group_size=2).fit_transform(
            X,
            cs_groups=cs_groups,
        )

        expected = np.array([[0.25, 0.75, 0.625, 0.875]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_nullable_integer_groups_are_accepted(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_groups = pd.DataFrame([[0, 0, 1, -1]], dtype="Int64")

        transformed = CSPercentileRankScaler(min_group_size=2).fit_transform(
            X,
            cs_groups=cs_groups,
        )

        expected = np.array([[0.25, 0.75, 0.625, 0.875]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_zero_weight_assets_still_receive_output(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_weights = np.array([[1.0, 1.0, 0.0, 0.0]])

        transformed = CSPercentileRankScaler().fit_transform(X, cs_weights=cs_weights)

        expected = np.array([[0.25, 0.75, 0.75, 0.75]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-12)

    def test_nan_preservation(self):
        X = np.array([[1.0, np.nan, 3.0, 5.0]])
        transformed = CSPercentileRankScaler().fit_transform(X)

        assert np.isnan(transformed[0, 1])
        np.testing.assert_allclose(
            transformed[0, [0, 2, 3]],
            np.array([1.0 / 6.0, 0.5, 5.0 / 6.0]),
            rtol=1e-12,
        )

    def test_multi_row_independence(self):
        X_single = np.array([[1.0, 2.0, 4.0, 10.0]])
        X_multi = np.array([[1.0, 2.0, 4.0, 10.0], [4.0, 1.0, 3.0, 7.0]])
        scaler = CSPercentileRankScaler()

        np.testing.assert_array_equal(
            scaler.fit_transform(X_single)[0],
            scaler.fit_transform(X_multi)[0],
        )

    def test_output_in_documented_bounds(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 12))
        transformed = CSPercentileRankScaler().fit_transform(X)
        n = X.shape[1]
        np.testing.assert_array_less(transformed, 1.0 - 0.5 / n + 1e-12)
        np.testing.assert_array_less(0.5 / n - 1e-12, transformed)

    def test_monotonic_in_X_per_row(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 20))

        transformed = CSPercentileRankScaler().fit_transform(X)

        for row in range(X.shape[0]):
            order = np.argsort(X[row], kind="stable")
            sorted_percentiles = transformed[row, order]
            assert np.all(np.diff(sorted_percentiles) >= -1e-12)

    def test_single_estimation_member_maps_to_neutral(self):
        X = np.array([[1.0, 5.0, 9.0]])
        cs_weights = np.array([[0.0, 1.0, 0.0]])

        transformed = CSPercentileRankScaler().fit_transform(X, cs_weights=cs_weights)
        np.testing.assert_allclose(transformed, 0.5, rtol=1e-12)

    def test_all_missing_groups_match_global_percentile(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0]])
        cs_groups = np.full(X.shape, -1, dtype=int)

        transformed = CSPercentileRankScaler(min_group_size=2).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = CSPercentileRankScaler().fit_transform(X)

        np.testing.assert_allclose(transformed, expected, rtol=1e-12)


class TestGroupedFallback:
    """Cover the count-based and missing-group fallback paths explicitly."""

    def test_grouped_happy_path_without_fallback(self):
        X = np.array([[1.0, 3.0, 5.0, 10.0, 12.0, 14.0]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        transformed = CSPercentileRankScaler(min_group_size=3).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=3)

        np.testing.assert_allclose(transformed, expected, rtol=1e-12)
        per_group = np.array([[1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0 / 6.0, 0.5, 5.0 / 6.0]])
        np.testing.assert_allclose(transformed, per_group, rtol=1e-12)

    def test_min_group_size_one_disables_fallback(self):
        X = np.array([[1.0, 3.0, 10.0, 14.0, 22.0]])
        cs_groups = np.array([[0, 0, 1, 1, 2]])

        transformed = CSPercentileRankScaler(min_group_size=1).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=1)

        np.testing.assert_allclose(transformed, expected, rtol=1e-12)
        np.testing.assert_allclose(transformed[0, [0, 1]], [0.25, 0.75], rtol=1e-12)
        np.testing.assert_allclose(transformed[0, [2, 3]], [0.25, 0.75], rtol=1e-12)
        np.testing.assert_allclose(transformed[0, 4], 0.5, rtol=1e-12)

    def test_nan_inside_group_reduces_count_and_triggers_fallback(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0, np.nan, 14.0]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        transformed = CSPercentileRankScaler(min_group_size=3).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=3)

        assert np.isnan(transformed[0, 4])
        np.testing.assert_allclose(
            transformed[~np.isnan(transformed)],
            expected[~np.isnan(expected)],
            rtol=1e-12,
        )

    def test_all_nan_group_falls_back_and_preserves_nan(self):
        X = np.array([[1.0, 2.0, 4.0, np.nan, np.nan, np.nan]])
        cs_groups = np.array([[0, 0, 0, 1, 1, 1]])

        transformed = CSPercentileRankScaler(min_group_size=2).fit_transform(
            X, cs_groups=cs_groups
        )
        assert np.all(np.isnan(transformed[0, 3:]))
        np.testing.assert_allclose(
            transformed[0, :3], [1.0 / 6.0, 0.5, 5.0 / 6.0], rtol=1e-12
        )


class TestStatisticalContract:
    """Numerical contracts that protect against silent statistical bugs."""

    def test_grouped_invariant_to_group_relabeling(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 30))
        cs_groups = rng.integers(0, 4, size=X.shape)

        relabel = np.array([3, 1, 0, 2])
        cs_groups_relabeled = relabel[cs_groups]

        scaler = CSPercentileRankScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_groups=cs_groups)
        b = scaler.fit_transform(X, cs_groups=cs_groups_relabeled)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_invariant_to_asset_permutation(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 20))
        cs_groups = rng.integers(0, 3, size=X.shape)
        cs_weights = rng.random(X.shape) + 0.1

        perm = rng.permutation(X.shape[1])
        scaler = CSPercentileRankScaler(min_group_size=2)
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

        scaler = CSPercentileRankScaler(min_group_size=2)
        out_multi = scaler.fit_transform(X_multi, cs_groups=cs_groups_multi)
        out_a = scaler.fit_transform(X_a, cs_groups=cs_groups_a)
        out_b = scaler.fit_transform(X_b, cs_groups=cs_groups_b)

        np.testing.assert_allclose(out_multi[0], out_a[0], rtol=1e-12)
        np.testing.assert_allclose(out_multi[1], out_b[0], rtol=1e-12)

    def test_grouped_sparse_group_labels(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0, 14.0, 18.0]])
        cs_groups_dense = np.array([[0, 0, 0, 1, 1, 1]])
        cs_groups_sparse = np.array([[7, 7, 7, 999_999, 999_999, 999_999]])

        scaler = CSPercentileRankScaler(min_group_size=2)
        a = scaler.fit_transform(X, cs_groups=cs_groups_dense)
        b = scaler.fit_transform(X, cs_groups=cs_groups_sparse)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_grouped_object_dtype_groups_accepted(self):
        X = np.array([[1.0, 2.0, 4.0, 10.0, 14.0, 18.0]])
        cs_groups_int = np.array([[0, 0, 0, 1, 1, 1]])
        cs_groups_obj = np.array([[0, 0, 0, 1, 1, 1]], dtype=object)

        scaler = CSPercentileRankScaler(min_group_size=2)
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

        actual = CSPercentileRankScaler(min_group_size=3).fit_transform(
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

        actual = CSPercentileRankScaler(min_group_size=3).fit_transform(
            X, cs_groups=cs_groups
        )
        expected = _reference_transform(X, cs_groups=cs_groups, min_group_size=3)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )

    def test_random_inputs_no_groups(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 15))
        X[rng.random(X.shape) < 0.10] = np.nan
        cs_weights = rng.random(X.shape)

        actual = CSPercentileRankScaler().fit_transform(X, cs_weights=cs_weights)
        expected = _reference_transform(X, cs_weights=cs_weights)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )


class TestValidation:
    """Parameter and input validation."""

    @pytest.mark.parametrize("min_group_size", [0, 1.5, np.nan, np.inf, -np.inf, True])
    def test_invalid_min_group_size(self, min_group_size):
        with pytest.raises(ValueError, match="min_group_size"):
            CSPercentileRankScaler(min_group_size=min_group_size).fit_transform(
                np.ones((1, 4))
            )

    def test_groups_must_be_integer(self):
        X = np.ones((1, 4))
        cs_groups = np.array([[0.0, 0.0, 1.0, 1.0]])

        with pytest.raises(ValueError, match="integer array"):
            CSPercentileRankScaler().fit_transform(X, cs_groups=cs_groups)

    def test_groups_shape_mismatch(self):
        X = np.ones((1, 4))
        cs_groups = np.ones((1, 3), dtype=int)

        with pytest.raises(ValueError, match="same shape"):
            CSPercentileRankScaler().fit_transform(X, cs_groups=cs_groups)

    def test_groups_must_be_greater_than_minus_one(self):
        X = np.ones((1, 4))
        cs_groups = np.array([[0, -2, 1, 1]])

        with pytest.raises(ValueError, match=">= -1"):
            CSPercentileRankScaler().fit_transform(X, cs_groups=cs_groups)

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
            CSPercentileRankScaler().fit_transform(X, cs_weights=cs_weights)

    def test_empty_estimation_universe_raises(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        cs_weights = np.zeros_like(X)

        with pytest.raises(ValueError, match="estimation asset"):
            CSPercentileRankScaler().fit_transform(X, cs_weights=cs_weights)
