"""Tests for RollingFactor."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from skfolio._constants import _BENCHMARK_WEIGHTS
from skfolio.containers import AssetPanel, FieldCategorical, InactivePolicy
from skfolio.datasets import make_synthetic_characteristics
from skfolio.factor_exposure import RollingFactor
from skfolio.preprocessing import CSStandardScaler, CSWinsorizer
from skfolio.prior import CharacteristicsFactorModel

WINDOWS = {
    "mean": [3, 5],
    "std": [5],
    "var": [4],
    "min": [4],
    "max": [4],
    "median": [3],
    "sum": [2],
    "skew": [6],
    "kurt": [6],
    "sem": [4],
    "count": [3],
    "first": [3],
    "last": [3],
    "nunique": [3],
}


def _raw_factor(**kwargs):
    """RollingFactor without cross-sectional transformations."""
    return RollingFactor(
        scoring_transformer="passthrough", outlier_transformer="passthrough", **kwargs
    )


def _pandas_reference(values, aggregation, window):
    df = pd.DataFrame(values)
    if aggregation == "lag":
        return df.shift(window).to_numpy()
    return getattr(df.rolling(window), aggregation)().to_numpy()


class TestOutput:
    def test_shape_and_factor_names(self, simple_panel):
        factor = RollingFactor(source="returns", windows={"mean": [3, 5], "lag": [1]})
        result = factor.fit_transform(simple_panel)

        assert result.shape == (simple_panel.n_observations, simple_panel.n_assets, 3)
        np.testing.assert_array_equal(
            factor.factor_names_,
            np.array(["returns_mean_3", "returns_mean_5", "returns_lag_1"]),
        )
        assert factor.n_factors_ == 3

    @pytest.mark.parametrize(
        ("aggregation", "window"),
        [(agg, w) for agg, ws in WINDOWS.items() for w in ws]
        + [("lag", 1), ("lag", 3)],
    )
    def test_raw_values_match_pandas(self, simple_panel, aggregation, window):
        result = _raw_factor(
            source="returns", windows={aggregation: [window]}
        ).fit_transform(simple_panel)

        expected = _pandas_reference(simple_panel["returns"], aggregation, window)
        np.testing.assert_allclose(result[:, :, 0], expected, equal_nan=True)

    def test_column_order_follows_windows_order(self, simple_panel):
        factor = _raw_factor(source="returns", windows={"lag": [2], "sum": [3, 2]})
        result = factor.fit_transform(simple_panel)
        returns = simple_panel["returns"]

        np.testing.assert_array_equal(
            factor.factor_names_, ["returns_lag_2", "returns_sum_3", "returns_sum_2"]
        )
        np.testing.assert_allclose(
            result[:, :, 0], _pandas_reference(returns, "lag", 2), equal_nan=True
        )
        np.testing.assert_allclose(
            result[:, :, 1], _pandas_reference(returns, "sum", 3), equal_nan=True
        )
        np.testing.assert_allclose(
            result[:, :, 2], _pandas_reference(returns, "sum", 2), equal_nan=True
        )

    def test_early_observations_are_nan(self, simple_panel):
        result = _raw_factor(
            source="returns", windows={"mean": [5], "lag": [2]}
        ).fit_transform(simple_panel)

        assert np.all(np.isnan(result[:4, :, 0]))
        assert not np.any(np.isnan(result[4:, :, 0]))
        assert np.all(np.isnan(result[:2, :, 1]))
        assert not np.any(np.isnan(result[2:, :, 1]))

    def test_missing_source_value_propagates(self, simple_panel):
        simple_panel["returns"][8, 0] = np.nan
        result = _raw_factor(source="returns", windows={"mean": [3]}).fit_transform(
            simple_panel
        )

        assert np.all(np.isnan(result[8:11, 0, 0]))
        assert not np.isnan(result[7, 0, 0])
        assert not np.isnan(result[11, 0, 0])
        assert not np.any(np.isnan(result[8:11, 1:, 0]))

    def test_late_listing_needs_full_active_window(self, simple_panel):
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = _raw_factor(source="returns", windows={"mean": [3]}).fit_transform(
            simple_panel
        )

        assert np.all(np.isnan(result[:7, 0, 0]))
        assert not np.isnan(result[7, 0, 0])

    def test_inactive_gap_breaks_active_window(self, simple_panel):
        active_mask = simple_panel.active_mask.copy()
        active_mask[5:7, 0] = False
        simple_panel.active_mask = active_mask

        result = _raw_factor(
            source="returns", windows={"max": [3], "lag": [1]}
        ).fit_transform(simple_panel)

        # Rolling max with window 3
        assert not np.isnan(result[4, 0, 0])
        assert np.all(np.isnan(result[5:9, 0, 0]))
        assert not np.isnan(result[9, 0, 0])
        # Lag 1 needs the asset active today and yesterday
        assert not np.isnan(result[4, 0, 1])
        assert np.all(np.isnan(result[5:8, 0, 1]))
        assert not np.isnan(result[8, 0, 1])

    def test_inactive_assets_are_nan(self, simple_panel):
        active_mask = simple_panel.active_mask.copy()
        active_mask[10, 2] = False
        simple_panel.active_mask = active_mask

        result = RollingFactor(source="returns", windows={"lag": [1]}).fit_transform(
            simple_panel
        )

        assert np.isnan(result[10, 2, 0])
        assert not np.isnan(result[10, 3, 0])

    def test_source_other_than_returns(self, simple_panel):
        result = _raw_factor(source="adj_volume", windows={"mean": [4]}).fit_transform(
            simple_panel
        )
        expected = _pandas_reference(simple_panel["adj_volume"], "mean", 4)
        np.testing.assert_allclose(result[:, :, 0], expected, equal_nan=True)


class TestCrossSectionalTransformers:
    def test_default_scoring_standardizes_each_column(self, simple_panel):
        factor = RollingFactor(source="returns", windows={"mean": [3], "std": [4]})
        result = factor.fit_transform(simple_panel)

        assert isinstance(factor.scoring_transformer_, CSStandardScaler)
        assert factor.outlier_transformer_ == "passthrough"
        for j in range(2):
            valid = ~np.isnan(result[:, :, j]).any(axis=1)
            np.testing.assert_allclose(
                np.nanmean(result[valid, :, j], axis=1), 0.0, atol=1e-12
            )

    def test_outlier_transformer_none_defaults_to_winsorizer(self, simple_panel):
        factor = RollingFactor(
            source="returns", windows={"mean": [3]}, outlier_transformer=None
        )
        factor.fit_transform(simple_panel)
        assert isinstance(factor.outlier_transformer_, CSWinsorizer)

    def test_transformers_are_applied_per_column(self, simple_panel):
        raw = _raw_factor(source="returns", windows={"mean": [3], "lag": [1]})
        scored = RollingFactor(source="returns", windows={"mean": [3], "lag": [1]})
        raw_result = raw.fit_transform(simple_panel)
        scored_result = scored.fit_transform(simple_panel)

        for j in range(2):
            expected = CSStandardScaler().fit_transform(
                raw_result[:, :, j], cs_weights=simple_panel["benchmark_weights"]
            )
            np.testing.assert_allclose(scored_result[:, :, j], expected, equal_nan=True)

    def test_transform_by_group(self, simple_panel):
        codes = np.tile(np.array([0, 0, 1, 1, 1], dtype=np.int32), (20, 1))
        simple_panel["industry"] = FieldCategorical(codes, levels=np.array(["A", "B"]))
        raw = _raw_factor(source="returns", windows={"mean": [3]}).fit_transform(
            simple_panel
        )
        result = RollingFactor(
            source="returns", windows={"mean": [3]}, transform_by_group="industry"
        ).fit_transform(simple_panel)

        expected = CSStandardScaler().fit_transform(
            raw[:, :, 0],
            cs_weights=simple_panel["benchmark_weights"],
            cs_groups=simple_panel["industry"],
        )
        np.testing.assert_allclose(result[:, :, 0], expected, equal_nan=True)


class TestOnline:
    def test_partial_fit_matches_fit(self, simple_panel):
        full = RollingFactor(source="returns", windows=WINDOWS).fit_transform(
            simple_panel
        )
        partial = RollingFactor(
            source="returns", windows=WINDOWS
        ).partial_fit_transform(simple_panel)
        np.testing.assert_array_equal(full, partial)

    @pytest.mark.parametrize("bounds", [(7, 13), (1, 2), (4, 5, 6, 19)])
    def test_partial_fit_chunked(self, simple_panel, bounds):
        windows = {**WINDOWS, "lag": [1, 4]}
        full = _raw_factor(source="returns", windows=windows).fit_transform(
            simple_panel
        )

        factor = _raw_factor(source="returns", windows=windows)
        edges = [0, *bounds, simple_panel.n_observations]
        chunks = [
            factor.partial_fit_transform(simple_panel[start:stop])
            for start, stop in itertools.pairwise(edges)
        ]

        np.testing.assert_allclose(np.concatenate(chunks), full, equal_nan=True)

    def test_partial_fit_single_obs_chunks(self, simple_panel):
        full = _raw_factor(source="returns", windows=WINDOWS).fit_transform(
            simple_panel
        )

        factor = _raw_factor(source="returns", windows=WINDOWS)
        chunks = [
            factor.partial_fit_transform(simple_panel[t : t + 1])
            for t in range(simple_panel.n_observations)
        ]

        np.testing.assert_allclose(np.concatenate(chunks), full, equal_nan=True)

    def test_partial_fit_chunked_with_inactive_gap(self, simple_panel):
        active_mask = simple_panel.active_mask.copy()
        active_mask[6:8, 1] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][6:8, 1] = np.nan

        windows = {"mean": [4], "lag": [2]}
        full = _raw_factor(source="returns", windows=windows).fit_transform(
            simple_panel
        )

        factor = _raw_factor(source="returns", windows=windows)
        chunks = [
            factor.partial_fit_transform(simple_panel[:5]),
            factor.partial_fit_transform(simple_panel[5:12]),
            factor.partial_fit_transform(simple_panel[12:]),
        ]

        np.testing.assert_allclose(np.concatenate(chunks), full, equal_nan=True)

    def test_fit_transform_resets_state(self, simple_panel):
        factor = RollingFactor(source="returns", windows={"mean": [5]})
        factor.partial_fit_transform(simple_panel[:10])

        result = factor.fit_transform(simple_panel)
        expected = RollingFactor(source="returns", windows={"mean": [5]}).fit_transform(
            simple_panel
        )
        np.testing.assert_array_equal(result, expected)


class TestValidation:
    def test_missing_source_field_raises(self, simple_panel):
        with pytest.raises(ValueError, match="Required fields are missing"):
            RollingFactor(source="unknown", windows={"mean": [3]}).fit_transform(
                simple_panel
            )

    def test_categorical_source_raises(self, simple_panel):
        codes = np.zeros((20, 5), dtype=np.int32)
        simple_panel["industry"] = FieldCategorical(codes, levels=np.array(["A"]))
        with pytest.raises(ValueError, match="numeric 2D field"):
            RollingFactor(source="industry", windows={"mean": [3]}).fit_transform(
                simple_panel
            )

    @pytest.mark.parametrize(
        ("windows", "match"),
        [
            ({}, "non-empty dict"),
            ([3, 5], "non-empty dict"),
            ({"kurtosis": [3]}, "Unsupported aggregation"),
            ({"quantile": [3]}, "Unsupported aggregation"),
            ({"apply": [3]}, "Unsupported aggregation"),
            ({"mean": 3}, "list of positive integers"),
            ({"mean": "3"}, "list of positive integers"),
            ({"mean": []}, "at least one window size"),
            ({"mean": [0]}, "positive integer"),
            ({"mean": [2.5]}, "positive integer"),
            ({"lag": [-1]}, "positive integer"),
            ({"mean": [3, 3]}, "duplicate window sizes"),
        ],
    )
    def test_invalid_windows_raise(self, simple_panel, windows, match):
        with pytest.raises(ValueError, match=match):
            RollingFactor(source="returns", windows=windows).fit_transform(simple_panel)

    def test_non_finite_source_raises(self, simple_panel):
        simple_panel["returns"][3, 0] = np.inf
        with pytest.raises(ValueError):
            RollingFactor(source="returns", windows={"mean": [3]}).fit_transform(
                simple_panel
            )


class TestCharacteristicsFactorModel:
    def test_multi_factor_names_and_families_are_expanded(self):
        panel = make_synthetic_characteristics(
            n_assets=80,
            n_observations=80,
            late_listing_proba=0.0,
            delisting_proba=0.0,
            random_state=0,
        )
        X = panel.to_dataframe(fields="returns")
        model = CharacteristicsFactorModel(
            factors=[
                (
                    "rolling_returns",
                    RollingFactor(
                        source="returns",
                        windows={"mean": [5], "std": [10], "lag": [1]},
                        family="rolling",
                    ),
                ),
            ],
            n_jobs=1,
        )
        model.fit(X, characteristics=panel)

        factor_names = list(model.factor_model_.factor_names)
        assert factor_names == ["returns_mean_5", "returns_std_10", "returns_lag_1"]
        assert set(model.factor_model_.factor_families) == {"rolling"}


def _gappy_panel(seed=0, n_observations=40, n_assets=4):
    """Panel with listing gaps: inactive observations are NaN in the field."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_observations, n_assets))
    active = rng.random((n_observations, n_assets)) > 0.25
    active[:, 0] = True
    active[4, 1] = True
    active[5:9, 1] = False
    active[9, 1] = True
    values[~active] = np.nan
    panel = AssetPanel(
        fields={"signal": values},
        asset_names=np.array([f"a{i}" for i in range(n_assets)]),
        observations=np.arange(n_observations),
        active_mask=active,
    )
    panel.add_2d_field(
        name=_BENCHMARK_WEIGHTS,
        values=np.where(active, 1.0 / n_assets, 0.0),
        inactive_policy=InactivePolicy.ZERO,
    )
    return panel, values, active


GAP_WINDOWS = {"mean": [3], "std": [4], "lag": [1, 2], "max": [2], "count": [3]}


class TestGapPolicy:
    def test_invalid_policy_raises(self, simple_panel):
        with pytest.raises(ValueError, match="gap_policy"):
            RollingFactor(
                source="returns", windows={"mean": [3]}, gap_policy="ignore"
            ).fit_transform(simple_panel)

    def test_skip_matches_per_asset_compacted_reference(self):
        """With gap_policy="skip" every asset is aggregated over its own sequence of
        active observations."""
        panel, values, active = _gappy_panel()
        factor = _raw_factor(source="signal", windows=GAP_WINDOWS, gap_policy="skip")
        result = factor.fit_transform(panel)

        expected = np.full_like(result, np.nan)
        for j in range(panel.n_assets):
            idx = np.flatnonzero(active[:, j])
            series = pd.Series(values[idx, j])
            for k, (aggregation, window) in enumerate(factor._specs):
                if aggregation == "lag":
                    ref = series.shift(window).to_numpy(copy=True)
                    lookback = window + 1
                else:
                    ref = getattr(series.rolling(window), aggregation)().to_numpy(
                        copy=True
                    )
                    lookback = window
                # NaN until the asset has a full lookback of active observations
                ref[: lookback - 1] = np.nan
                expected[idx, j, k] = ref
        np.testing.assert_allclose(result, expected, equal_nan=True)
        assert np.isnan(result[~active]).all()

    def test_skip_lag_reaches_across_a_gap(self):
        panel, values, _ = _gappy_panel()
        invalidate = _raw_factor(
            source="signal", windows={"lag": [1]}, gap_policy="invalidate"
        ).fit_transform(panel)
        skip = _raw_factor(
            source="signal", windows={"lag": [1]}, gap_policy="skip"
        ).fit_transform(panel)
        # asset 1 is inactive on rows 5 to 8: at row 9 the invalidating lag is
        # NaN, the skipping lag returns the value of row 4
        assert np.isnan(invalidate[9, 1, 0])
        assert skip[9, 1, 0] == values[4, 1]
        assert np.isnan(skip).sum() < np.isnan(invalidate).sum()

    def test_skip_without_gaps_equals_invalidate(self, simple_panel):
        invalidate = _raw_factor(
            source="returns", windows=GAP_WINDOWS, gap_policy="invalidate"
        ).fit_transform(simple_panel)
        skip = _raw_factor(
            source="returns", windows=GAP_WINDOWS, gap_policy="skip"
        ).fit_transform(simple_panel)
        np.testing.assert_allclose(skip, invalidate, equal_nan=True)

    @pytest.mark.parametrize("bounds", [(1, 7, 8, 25), tuple(range(1, 40))])
    def test_skip_chunked_partial_fit_matches_fit(self, bounds):
        panel, _, _ = _gappy_panel()
        full = _raw_factor(
            source="signal", windows=GAP_WINDOWS, gap_policy="skip"
        ).fit_transform(panel)
        factor = _raw_factor(source="signal", windows=GAP_WINDOWS, gap_policy="skip")
        edges = [0, *bounds, panel.n_observations]
        chunks = [
            factor.partial_fit_transform(panel[start:stop])
            for start, stop in itertools.pairwise(edges)
        ]
        np.testing.assert_allclose(np.concatenate(chunks), full, equal_nan=True)

    def test_skip_fit_transform_resets_state(self):
        panel, _, _ = _gappy_panel()
        factor = _raw_factor(source="signal", windows=GAP_WINDOWS, gap_policy="skip")
        factor.partial_fit_transform(panel[:20])
        result = factor.fit_transform(panel)
        expected = _raw_factor(
            source="signal", windows=GAP_WINDOWS, gap_policy="skip"
        ).fit_transform(panel)
        np.testing.assert_allclose(result, expected, equal_nan=True)
