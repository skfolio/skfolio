"""Tests for LogMarketCap descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import LogMarketCap


class TestLogMarketCap:
    """Unit tests for LogMarketCap descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = LogMarketCap().fit_transform(simple_panel)
        assert result.shape == simple_panel["market_cap"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = LogMarketCap()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_numpy_log(self, simple_panel):
        """Output equals np.log(market_cap)."""
        result = LogMarketCap().fit_transform(simple_panel)
        expected = np.log(simple_panel["market_cap"])
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in market_cap propagates to output."""
        simple_panel["market_cap"][0, 0] = np.nan
        result = LogMarketCap().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        # Other values unaffected
        assert not np.isnan(result[0, 1])

    def test_raise_when_market_cap_zero(self, simple_panel):
        """Non-missing market cap must be strictly positive."""
        simple_panel["market_cap"][0, 0] = 0.0
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            LogMarketCap().fit_transform(simple_panel)

    def test_raise_when_market_cap_negative(self, simple_panel):
        """Negative market cap is invalid."""
        simple_panel["market_cap"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            LogMarketCap().fit_transform(simple_panel)

    def test_raise_when_market_cap_infinite(self, simple_panel):
        """Infinite market cap is invalid."""
        simple_panel["market_cap"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            LogMarketCap().fit_transform(simple_panel)
