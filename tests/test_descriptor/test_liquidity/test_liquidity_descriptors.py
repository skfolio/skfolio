"""Tests for Liquidity descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    EWAmihudIlliquidity,
    EWShareTurnover,
)
from skfolio.utils.tools import half_life_to_decay_factor


@pytest.fixture(params=[EWShareTurnover, EWAmihudIlliquidity])
def descriptor_cls(request):
    return request.param


def test_early_observations_are_nan(simple_panel, descriptor_cls):
    """Observations before min_periods are NaN."""
    min_periods = 8
    result = descriptor_cls(half_life=5, min_periods=min_periods).fit_transform(
        simple_panel
    )
    assert np.all(np.isnan(result[: min_periods - 1]))
    assert not np.all(np.isnan(result[min_periods - 1]))


def test_default_min_periods(descriptor_cls):
    """Default min_periods is None (resolved to half_life internally)."""
    d = descriptor_cls(half_life=10)
    assert d.min_periods is None


def test_output_non_negative(simple_panel, descriptor_cls):
    """Turnover is non-negative (volume and shares are positive)."""
    result = descriptor_cls(half_life=5, min_periods=1).fit_transform(simple_panel)
    valid = ~np.isnan(result)
    assert np.all(result[valid] >= 0)


def test_partial_fit_matches_fit(simple_panel, descriptor_cls):
    """partial_fit_transform in one shot matches fit_transform."""
    full = descriptor_cls(half_life=5, min_periods=2).fit_transform(simple_panel)
    partial = descriptor_cls(half_life=5, min_periods=2).partial_fit_transform(
        simple_panel
    )
    np.testing.assert_array_equal(full, partial)


def test_partial_fit_chunked(simple_panel, descriptor_cls):
    """Chunked partial_fit_transform matches fit_transform."""
    kwargs = dict(half_life=5, min_periods=3)
    full = descriptor_cls(**kwargs).fit_transform(simple_panel)

    descriptor = descriptor_cls(**kwargs)
    r1 = descriptor.partial_fit_transform(simple_panel[:7])
    r2 = descriptor.partial_fit_transform(simple_panel[7:13])
    r3 = descriptor.partial_fit_transform(simple_panel[13:])

    combined = np.concatenate([r1, r2, r3], axis=0)
    np.testing.assert_array_almost_equal(combined, full)


def test_partial_fit_single_obs_chunks(simple_panel, descriptor_cls):
    """Single-observation chunks produce correct results."""
    kwargs = dict(half_life=5, min_periods=2)
    full = descriptor_cls(**kwargs).fit_transform(simple_panel)

    descriptor = descriptor_cls(**kwargs)
    chunks = []
    for start in range(20):
        chunks.append(descriptor.partial_fit_transform(simple_panel[start : start + 1]))
    combined = np.concatenate(chunks, axis=0)
    np.testing.assert_array_almost_equal(combined, full)


def test_fit_transform_resets_state(simple_panel, descriptor_cls):
    """fit_transform resets state from a previous run."""
    descriptor = descriptor_cls(half_life=5, min_periods=2)
    descriptor.partial_fit_transform(simple_panel[:10])

    result = descriptor.fit_transform(simple_panel)
    expected = descriptor_cls(half_life=5, min_periods=2).fit_transform(simple_panel)
    np.testing.assert_array_equal(result, expected)


def test_raises_on_infinite_adj_volume(simple_panel, descriptor_cls):
    """Raises ValueError when adj_volume contains infinite values."""
    simple_panel["adj_volume"][5, 0] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        descriptor_cls(half_life=5, min_periods=1).fit_transform(simple_panel)


def test_raises_on_negative_adj_volume(simple_panel, descriptor_cls):
    """Raises ValueError when adj_volume is negative."""
    simple_panel["adj_volume"][5, 0] = -1.0
    with pytest.raises(ValueError, match="negative values"):
        descriptor_cls(half_life=5, min_periods=1).fit_transform(simple_panel)


def test_default_min_periods_is_at_least_one(simple_panel, descriptor_cls):
    """Default min_periods is at least one when half_life is below one."""
    result = descriptor_cls(half_life=0.5).fit_transform(simple_panel)
    assert not np.all(np.isnan(result[0]))


def test_raises_on_invalid_half_life(simple_panel, descriptor_cls):
    """Raises ValueError when half_life <= 0."""
    with pytest.raises(ValueError, match="half_life must be a positive number"):
        descriptor_cls(half_life=0).fit_transform(simple_panel)


def test_raises_on_invalid_min_periods(simple_panel, descriptor_cls):
    """Raises ValueError when min_periods < 1."""
    with pytest.raises(ValueError, match="min_periods must be a positive integer"):
        descriptor_cls(half_life=5, min_periods=0).fit_transform(simple_panel)


def test_higher_half_life_smoother(simple_panel, descriptor_cls):
    """Higher half_life produces smoother output."""
    result_fast = descriptor_cls(half_life=2, min_periods=1).fit_transform(simple_panel)
    result_slow = descriptor_cls(half_life=10, min_periods=1).fit_transform(
        simple_panel
    )
    valid_fast = result_fast[~np.isnan(result_fast)]
    valid_slow = result_slow[~np.isnan(result_slow)]
    assert np.var(valid_slow) < np.var(valid_fast)


# ---------------------------------------------------------------------------
# EWShareTurnover
# ---------------------------------------------------------------------------
class TestEWShareTurnover:
    """Tests for EWShareTurnover descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = EWShareTurnover(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert result.shape == simple_panel["adj_volume"].shape

    def test_ewma_formula(self, simple_panel):
        """EWMA matches manual computation."""
        half_life = 5
        decay = half_life_to_decay_factor(half_life)

        result = EWShareTurnover(half_life=half_life, min_periods=1).fit_transform(
            simple_panel
        )

        adj_volume = simple_panel["adj_volume"]
        adj_shares_outstanding = simple_panel["adj_shares_outstanding"]
        raw_turnover = adj_volume / adj_shares_outstanding
        n_obs, n_assets = raw_turnover.shape

        ewma = np.zeros(n_assets)
        for t in range(n_obs):
            ewma = decay * ewma + (1 - decay) * raw_turnover[t]
            np.testing.assert_array_almost_equal(result[t], ewma)

    def test_nan_holds_state(self, simple_panel):
        """NaN volume holds EWMA state and does not update."""
        simple_panel["adj_volume"][5, 0] = np.nan
        result = EWShareTurnover(half_life=5, min_periods=1).fit_transform(simple_panel)
        # Output should still be finite
        assert not np.isnan(result[5, 0])

    def test_nan_does_not_count_toward_min_periods(self, simple_panel):
        """NaN observations do not count as valid observations."""
        simple_panel["adj_volume"][:3, 0] = np.nan
        result = EWShareTurnover(half_life=5, min_periods=2).fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_zero_volume_is_valid_zero_turnover(self, simple_panel):
        """Zero volume is valid and produces zero turnover."""
        simple_panel["adj_volume"][0, 0] = 0.0
        result = EWShareTurnover(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert result[0, 0] == 0.0

    def test_raises_on_infinite_adj_shares_outstanding(self, simple_panel):
        """Raises ValueError when adj_shares_outstanding contains infinite values."""
        simple_panel["adj_shares_outstanding"][5, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            EWShareTurnover(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_non_positive_adj_shares_outstanding(self, simple_panel):
        """Raises ValueError when adj_shares_outstanding is non-positive."""
        simple_panel["adj_shares_outstanding"][5, 0] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            EWShareTurnover(half_life=5, min_periods=1).fit_transform(simple_panel)


# ---------------------------------------------------------------------------
# EWAmihudIlliquidity
# ---------------------------------------------------------------------------
class TestEWAmihudIlliquidity:
    """Tests for EWAmihudIlliquidity descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = EWAmihudIlliquidity(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )
        assert result.shape == simple_panel["returns"].shape

    def test_ewma_formula(self, simple_panel):
        """EWMA matches manual computation."""
        half_life = 5
        decay = half_life_to_decay_factor(half_life)

        result = EWAmihudIlliquidity(half_life=half_life, min_periods=1).fit_transform(
            simple_panel
        )

        returns = simple_panel["returns"]
        adj_close = simple_panel["adj_close"]
        adj_volume = simple_panel["adj_volume"]
        traded_amount = adj_close * adj_volume
        n_obs, n_assets = returns.shape

        ewma = np.zeros(n_assets)
        for t in range(n_obs):
            r_i = returns[t]
            ta_i = traded_amount[t]
            valid = ~np.isnan(r_i) & (ta_i > 0)
            if np.any(valid):
                illiq = np.abs(r_i[valid]) / ta_i[valid]
                ewma[valid] = decay * ewma[valid] + (1 - decay) * illiq
            np.testing.assert_array_almost_equal(result[t], ewma)

    def test_zero_volume_skipped(self, simple_panel):
        """Zero volume is skipped and EWMA state is held."""
        simple_panel["adj_volume"][5, 0] = 0.0
        result = EWAmihudIlliquidity(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )
        # Output should still be finite (state held, not blown up)
        assert not np.isnan(result[5, 0])
        assert np.isfinite(result[5, 0])

    def test_zero_volume_does_not_count_toward_min_periods(self, simple_panel):
        """Zero-volume observations do not count as valid observations."""
        simple_panel["adj_volume"][:3, 0] = 0.0
        result = EWAmihudIlliquidity(half_life=5, min_periods=2).fit_transform(
            simple_panel
        )
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_nan_return_skipped(self, simple_panel):
        """NaN return is skipped and EWMA state is held."""
        simple_panel["returns"][5, 0] = np.nan
        result = EWAmihudIlliquidity(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )
        assert not np.isnan(result[5, 0])

    def test_raises_on_infinite_returns(self, simple_panel):
        """Raises ValueError when returns contains infinite values."""
        simple_panel["returns"][5, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            EWAmihudIlliquidity(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_infinite_adj_close(self, simple_panel):
        """Raises ValueError when adj_close contains infinite values."""
        simple_panel["adj_close"][5, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            EWAmihudIlliquidity(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_non_positive_adj_close(self, simple_panel):
        """Raises ValueError when adj_close is non-positive."""
        simple_panel["adj_close"][5, 0] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            EWAmihudIlliquidity(half_life=5, min_periods=1).fit_transform(simple_panel)
