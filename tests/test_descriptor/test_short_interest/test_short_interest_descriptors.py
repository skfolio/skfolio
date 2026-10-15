"""Tests for Short Interest descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import DaysToCover, ShortInterest
from skfolio.utils.tools import half_life_to_decay_factor


class TestShortInterest:
    """Tests for the ShortInterest descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = ShortInterest().fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = ShortInterest()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_formula(self, simple_panel):
        """Output matches short_interest / adj_shares_outstanding."""
        result = ShortInterest().fit_transform(simple_panel)
        expected = (
            simple_panel["short_interest"] / simple_panel["adj_shares_outstanding"]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_non_negative(self, simple_panel):
        """Result is non-negative when inputs are non-negative."""
        result = ShortInterest().fit_transform(simple_panel)
        assert np.all(result >= 0)

    def test_nan_propagation(self, simple_panel):
        """NaN in either input propagates to output."""
        simple_panel["short_interest"][3, 1] = np.nan
        simple_panel["adj_shares_outstanding"][5, 2] = np.nan
        result = ShortInterest().fit_transform(simple_panel)
        assert np.isnan(result[3, 1])
        assert np.isnan(result[5, 2])

    def test_raise_when_short_interest_negative(self, simple_panel):
        """Short interest must be non-negative."""
        simple_panel["short_interest"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"short_interest.*non-negative"):
            ShortInterest().fit_transform(simple_panel)

    def test_raise_when_short_interest_infinite(self, simple_panel):
        """Infinite short interest is invalid."""
        simple_panel["short_interest"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"short_interest.*non-negative"):
            ShortInterest().fit_transform(simple_panel)

    def test_raise_when_shares_outstanding_zero(self, simple_panel):
        """Adjusted shares outstanding must be strictly positive."""
        simple_panel["adj_shares_outstanding"][0, 0] = 0.0
        with pytest.raises(
            ValueError, match=r"adj_shares_outstanding.*strictly positive"
        ):
            ShortInterest().fit_transform(simple_panel)

    def test_raise_when_shares_outstanding_negative(self, simple_panel):
        """Negative adjusted shares outstanding are invalid."""
        simple_panel["adj_shares_outstanding"][0, 0] = -1.0
        with pytest.raises(
            ValueError, match=r"adj_shares_outstanding.*strictly positive"
        ):
            ShortInterest().fit_transform(simple_panel)

    def test_raise_when_shares_outstanding_infinite(self, simple_panel):
        """Infinite adjusted shares outstanding are invalid."""
        simple_panel["adj_shares_outstanding"][0, 0] = np.inf
        with pytest.raises(
            ValueError, match=r"adj_shares_outstanding.*strictly positive"
        ):
            ShortInterest().fit_transform(simple_panel)


class TestDaysToCover:
    """Tests for the DaysToCover descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = DaysToCover(min_periods=1).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_warmup_nan(self, simple_panel):
        """Observations before min_periods are NaN."""
        result = DaysToCover(half_life=5, min_periods=5).fit_transform(simple_panel)
        assert np.all(np.isnan(result[:4]))
        assert not np.any(np.isnan(result[4]))

    def test_default_min_periods(self, simple_panel):
        """Default min_periods equals half_life."""
        hl = 5
        result = DaysToCover(half_life=hl).fit_transform(simple_panel)
        assert np.all(np.isnan(result[: hl - 1]))
        assert not np.any(np.isnan(result[hl - 1]))

    def test_default_min_periods_is_at_least_one(self, simple_panel):
        """Default min_periods is at least one when half_life is below one."""
        result = DaysToCover(half_life=0.5).fit_transform(simple_panel)
        assert not np.all(np.isnan(result[0]))

    def test_ewma_volume_formula(self, simple_panel):
        """Verify EWMA volume and days-to-cover at a specific timestep."""
        hl = 5.0
        decay = half_life_to_decay_factor(hl)
        result = DaysToCover(half_life=hl, min_periods=1).fit_transform(simple_panel)

        volume = simple_panel["adj_volume"]
        si = simple_panel["short_interest"]

        # Manually compute EWMA volume for asset 0
        ewma_vol = 0.0
        for t in range(volume.shape[0]):
            ewma_vol = decay * ewma_vol + (1 - decay) * volume[t, 0]
        expected_dtc = si[-1, 0] / ewma_vol
        np.testing.assert_almost_equal(result[-1, 0], expected_dtc)

    def test_non_negative(self, simple_panel):
        """Result is non-negative when inputs are non-negative."""
        result = DaysToCover(min_periods=1).fit_transform(simple_panel)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        full = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)
        partial = DaysToCover(half_life=5, min_periods=1).partial_fit_transform(
            simple_panel
        )
        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        full = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

        descriptor = DaysToCover(half_life=5, min_periods=1)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_single_obs(self, simple_panel):
        """Single-observation chunks produce correct results."""
        full = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

        descriptor = DaysToCover(half_life=5, min_periods=1)
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = DaysToCover(half_life=5, min_periods=1)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)

    def test_nan_volume_holds_state(self, simple_panel):
        """NaN volume holds EWMA state and does not update."""
        simple_panel["adj_volume"][8, 0] = np.nan
        result = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert not np.isnan(result[8, 0])

    def test_nan_volume_does_not_count_toward_min_periods(self, simple_panel):
        """NaN volume observations do not count as valid observations."""
        simple_panel["adj_volume"][:3, 0] = np.nan
        result = DaysToCover(half_life=5, min_periods=2).fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_zero_volume_holds_state(self, simple_panel):
        """Zero volume holds EWMA state and does not update."""
        simple_panel["adj_volume"][8, 0] = 0.0
        result = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert not np.isnan(result[8, 0])
        assert np.isfinite(result[8, 0])

    def test_zero_volume_does_not_count_toward_min_periods(self, simple_panel):
        """Zero-volume observations do not count as valid observations."""
        simple_panel["adj_volume"][:3, 0] = 0.0
        result = DaysToCover(half_life=5, min_periods=2).fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_nan_short_interest_updates_volume_state(self, simple_panel):
        """NaN short interest does not prevent volume state updates."""
        simple_panel["short_interest"][0, 0] = np.nan
        result = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert not np.isnan(result[1, 0])

    def test_raises_on_negative_short_interest(self, simple_panel):
        """Raises ValueError when short interest is negative."""
        simple_panel["short_interest"][5, 0] = -1.0
        with pytest.raises(ValueError, match="negative values"):
            DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_infinite_short_interest(self, simple_panel):
        """Raises ValueError when short interest contains infinite values."""
        simple_panel["short_interest"][5, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_negative_adj_volume(self, simple_panel):
        """Raises ValueError when adjusted volume is negative."""
        simple_panel["adj_volume"][5, 0] = -1.0
        with pytest.raises(ValueError, match="negative values"):
            DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_infinite_adj_volume(self, simple_panel):
        """Raises ValueError when adjusted volume contains infinite values."""
        simple_panel["adj_volume"][5, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            DaysToCover(half_life=0).fit_transform(simple_panel)

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            DaysToCover(min_periods=0).fit_transform(simple_panel)

    def test_higher_short_interest_higher_dtc(self, simple_panel):
        """More shares short -> higher days to cover, all else equal."""
        result_low = DaysToCover(half_life=5, min_periods=1).fit_transform(simple_panel)

        simple_panel["short_interest"] = simple_panel["short_interest"] * 3
        result_high = DaysToCover(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )

        valid = ~np.isnan(result_low) & ~np.isnan(result_high)
        assert np.all(result_high[valid] > result_low[valid])

    def test_different_half_lives_differ(self, simple_panel):
        """Different half-lives produce different outputs."""
        r3 = DaysToCover(half_life=3, min_periods=1).fit_transform(simple_panel)
        r10 = DaysToCover(half_life=10, min_periods=1).fit_transform(simple_panel)
        assert not np.allclose(r3[-1], r10[-1])
