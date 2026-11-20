"""Tests for EWMomentum descriptor."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from skfolio.containers import AssetPanel, Field2D, InactivePolicy
from skfolio.descriptor import EWMomentum


@pytest.fixture(params=[InactivePolicy.MISSING, InactivePolicy.IGNORE])
def momentum_panel(request):
    """Returns with missing observations, late listings and reactivated assets."""
    rng = np.random.default_rng(42)
    shape = (73, 5)
    returns = rng.normal(0, 0.02, shape)
    returns[rng.random(shape) < 0.15] = np.nan
    returns[:, 4] = np.nan
    active_mask = np.ones(shape, dtype=bool)
    active_mask[:13, 1] = False
    active_mask[24:27, 2] = False
    active_mask[50, 2] = False
    active_mask[39:42, 3] = False
    if request.param == InactivePolicy.MISSING:
        returns[~active_mask] = np.nan
    return AssetPanel(
        fields={"returns": Field2D(returns, inactive_policy=request.param)},
        observations=np.arange(shape[0]),
        asset_names=np.array([f"asset_{asset}" for asset in range(shape[1])]),
        active_mask=active_mask,
    )


def _weighted_history(panel, *, half_life, skip, min_periods, exponentiate):
    """Sum full log-return histories using unnormalized exponential weights."""
    returns = panel["returns"]
    result = np.full(returns.shape, np.nan)
    for asset in range(panel.n_assets):
        last_inactive = -1
        history = []
        for t in range(panel.n_observations):
            if not panel.active_mask[t, asset]:
                last_inactive = t
                history.clear()
                continue
            source = t - skip
            if source > last_inactive and np.isfinite(returns[source, asset]):
                history.append(np.log1p(returns[source, asset]))
            if len(history) >= min_periods:
                ages = np.arange(len(history) - 1, -1, -1)
                weights = (1 - 2.0 ** (-1 / half_life)) * 2.0 ** (-ages / half_life)
                momentum = np.dot(history, weights)
                result[t, asset] = np.expm1(momentum) if exponentiate else momentum
    return result


class TestEWMomentum:
    """Tests for EWMomentum descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = EWMomentum(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before skip + min_periods are NaN."""
        skip = 3
        min_periods = 4
        result = EWMomentum(
            half_life=5, skip=skip, min_periods=min_periods
        ).fit_transform(simple_panel)
        # First skip + min_periods - 1 rows should be NaN
        # (skip observations excluded, then min_periods EWMA updates)
        assert np.all(np.isnan(result[: skip + min_periods - 1]))
        # Row at skip + min_periods - 1 should be valid
        assert not np.all(np.isnan(result[skip + min_periods - 1]))

    def test_default_half_life(self):
        """Default half_life is 87 (~1-year rolling window equivalent)."""
        m = EWMomentum()
        assert m.half_life == 87

    def test_default_min_periods(self):
        """Default min_periods is resolved during initialization."""
        m = EWMomentum(half_life=10)
        assert m.min_periods is None

    def test_fractional_half_life_default_min_periods(self, simple_panel):
        """Default min_periods is ceil(half_life), with minimum one."""
        descriptor = EWMomentum(half_life=1.2, skip=0)
        result = descriptor.fit_transform(simple_panel)
        assert descriptor.min_periods_ == 2
        assert np.all(np.isnan(result[0]))
        assert not np.all(np.isnan(result[1]))

        descriptor = EWMomentum(half_life=0.5, skip=0)
        descriptor.fit_transform(simple_panel)
        assert descriptor.min_periods_ == 1

    def test_no_skip_immediate_ewma(self, simple_panel):
        """With skip=0, EWMA starts from first observation."""
        min_periods = 2
        result = EWMomentum(half_life=5, skip=0, min_periods=min_periods).fit_transform(
            simple_panel
        )
        # First min_periods - 1 rows are NaN
        assert np.all(np.isnan(result[: min_periods - 1]))
        # Row at min_periods - 1 should be valid
        assert not np.all(np.isnan(result[min_periods - 1]))

    @pytest.mark.parametrize("half_life", [1e-6, 0.5, 7.3, 63])
    @pytest.mark.parametrize("skip", [0, 1, 3, 21, 80])
    @pytest.mark.parametrize("min_periods", [1, 4])
    @pytest.mark.parametrize("exponentiate", [False, True])
    def test_matches_weighted_history(
        self, momentum_panel, half_life, skip, min_periods, exponentiate
    ):
        """Momentum excludes recent returns and returns from before an inactivity reset."""
        params = dict(
            half_life=half_life,
            skip=skip,
            min_periods=min_periods,
            exponentiate=exponentiate,
        )
        result = EWMomentum(**params).fit_transform(momentum_panel)
        expected = _weighted_history(momentum_panel, **params)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-16)

    @pytest.mark.parametrize("skip", [0, 3])
    @pytest.mark.parametrize("exponentiate", [False, True])
    @pytest.mark.parametrize("daily_return", [-0.01, 0.01])
    def test_zero_initialization_shrinks_short_histories(
        self, simple_panel, skip, exponentiate, daily_return
    ):
        """Equal constant returns give smaller momentum magnitudes for newer assets."""
        simple_panel["returns"][:] = daily_return
        active_mask = simple_panel.active_mask.copy()
        active_mask[:10, 1] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:10, 1] = np.nan

        result = EWMomentum(
            half_life=5, skip=skip, min_periods=5, exponentiate=exponentiate
        ).fit_transform(simple_panel)

        first_output = 10 + skip + 5 - 1
        assert np.isnan(result[:first_output, 1]).all()
        n_observations = (
            np.arange(first_output, simple_panel.n_observations)[:, None]
            - np.array([0, 10])
            - skip
            + 1
        )
        expected = np.log1p(daily_return) * (1 - 2.0 ** (-n_observations / 5))
        if exponentiate:
            expected = np.expm1(expected)
        np.testing.assert_allclose(result[first_output:, :2], expected, rtol=1e-14)
        assert np.all(
            np.abs(result[first_output:, 1]) < np.abs(result[first_output:, 0])
        )

    @pytest.mark.parametrize("skip", [0, 1, 3, 21])
    @pytest.mark.parametrize("exponentiate", [False, True])
    def test_partial_fit_matches_fit_across_resets(
        self, momentum_panel, skip, exponentiate
    ):
        """Calls shorter than skip preserve the excluded observations and resets."""
        params = dict(
            half_life=7.3, skip=skip, min_periods=4, exponentiate=exponentiate
        )
        expected = EWMomentum(**params).fit_transform(momentum_panel)
        descriptor = EWMomentum(**params)
        stops = [
            0,
            1,
            13,
            17,
            24,
            25,
            27,
            39,
            40,
            42,
            51,
            momentum_panel.n_observations,
        ]
        result = np.concatenate(
            [
                descriptor.partial_fit_transform(momentum_panel[start:stop])
                for start, stop in pairwise(stops)
            ]
        )
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(descriptor.momentum_, expected[-1])

    @pytest.mark.parametrize("skip", [0, 1, 4])
    def test_reactivation_discards_history_and_pending_returns(
        self, simple_panel, skip
    ):
        """One inactive row clears the EWMA and returns stored for later inclusion."""
        simple_panel["returns"][:] = 0.01
        active_mask = simple_panel.active_mask.copy()
        active_mask[8, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][8, 0] = np.nan
        simple_panel["returns"][9:, 0] = 0.0
        params = dict(half_life=5, skip=skip, min_periods=2)
        result = EWMomentum(**params).fit_transform(simple_panel)
        fresh = EWMomentum(**params).fit_transform(simple_panel[9:])

        np.testing.assert_array_equal(result[9:, 0], fresh[:, 0])
        first_output = 9 + skip + 2 - 1
        assert np.isnan(result[8:first_output, 0]).all()
        np.testing.assert_array_equal(result[first_output:, 0], 0.0)

    def test_missing_returns_preserve_delay_and_ewma(self, simple_panel):
        """An earlier return can update momentum when the current return is missing."""
        simple_panel["returns"][:, 0] = np.nan
        simple_panel["returns"][:2, 0] = [0.01, 0.03]
        result = EWMomentum(half_life=1, skip=2, min_periods=2).fit_transform(
            simple_panel
        )

        assert np.isnan(result[:3, 0]).all()
        expected = np.dot(np.log1p([0.01, 0.03]), [0.25, 0.5])
        np.testing.assert_allclose(result[3:, 0], expected, rtol=1e-14)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = EWMomentum(half_life=5, skip=2, min_periods=3)

        # First run: partial with a chunk
        descriptor.partial_fit_transform(simple_panel[:10])

        # Second run: fit_transform should reset and produce clean result
        result = descriptor.fit_transform(simple_panel)
        expected = EWMomentum(half_life=5, skip=2, min_periods=3).fit_transform(
            simple_panel
        )
        np.testing.assert_array_equal(result, expected)

    def test_raises_on_infinite_returns(self, simple_panel):
        """Raises ValueError when returns contain infinity."""
        simple_panel["returns"][5, 0] = np.inf
        with pytest.raises(ValueError, match='Field "returns" contains infinite'):
            EWMomentum(half_life=5).fit_transform(simple_panel)

    def test_raises_on_returns_less_than_or_equal_to_minus_one(self, simple_panel):
        """Raises ValueError when log return is undefined."""
        simple_panel["returns"][5, 0] = -1.0
        with pytest.raises(ValueError, match="less than or equal to -1"):
            EWMomentum(half_life=5).fit_transform(simple_panel)

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            EWMomentum(half_life=0).fit_transform(simple_panel)

    def test_raises_on_negative_skip(self, simple_panel):
        """Raises ValueError when skip < 0."""
        with pytest.raises(ValueError, match="skip must be a non-negative integer"):
            EWMomentum(half_life=5, skip=-1).fit_transform(simple_panel)

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            EWMomentum(half_life=5, min_periods=0).fit_transform(simple_panel)

    def test_higher_half_life_smoother(self, simple_panel):
        """Higher half_life produces smoother (less volatile) output."""
        result_fast = EWMomentum(half_life=2, skip=0, min_periods=1).fit_transform(
            simple_panel
        )
        result_slow = EWMomentum(half_life=10, skip=0, min_periods=1).fit_transform(
            simple_panel
        )
        # Variance of slow EWMA should be lower (smoother)
        valid_fast = result_fast[~np.isnan(result_fast)]
        valid_slow = result_slow[~np.isnan(result_slow)]
        assert np.var(valid_slow) < np.var(valid_fast)
