"""Tests for make_synthetic_characteristics."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.containers import AssetPanel, FieldCategorical
from skfolio.datasets import make_synthetic_characteristics

_EXPECTED_FIELDS = {
    "returns",
    "adj_close",
    "adj_volume",
    "adj_shares_outstanding",
    "market_cap",
    "ebitda_ttm",
    "enterprise_value",
    "net_income_ttm",
    "sales_ttm",
    "dividends_ttm",
    "net_buybacks_ttm",
    "book_equity",
    "operating_cash_flow_ttm",
    "total_debt",
    "total_assets",
    "cost_of_revenue_ttm",
    "capex_ttm",
    "short_interest",
    "eps_ntm",
    "dps_ntm",
    "eps_ntm_std",
    "industry",
}


@pytest.fixture(scope="module")
def panel() -> AssetPanel:
    return make_synthetic_characteristics(
        n_assets=60, n_observations=300, n_industries=6, random_state=0
    )


def test_shape_and_fields(panel):
    assert isinstance(panel, AssetPanel)
    assert panel.n_assets == 60
    assert panel.n_observations == 300
    assert set(panel.keys()) == _EXPECTED_FIELDS
    assert isinstance(panel.get_field("industry"), FieldCategorical)


def test_reproducible():
    a = make_synthetic_characteristics(n_assets=20, n_observations=80, random_state=7)
    b = make_synthetic_characteristics(n_assets=20, n_observations=80, random_state=7)
    c = make_synthetic_characteristics(n_assets=20, n_observations=80, random_state=8)
    np.testing.assert_array_equal(a["returns"], b["returns"])
    assert not np.array_equal(a["returns"], c["returns"])


def test_bearish_signal_descriptors_predict_lower_returns():
    panel = make_synthetic_characteristics(
        n_assets=200,
        n_observations=500,
        late_listing_proba=0.0,
        delisting_proba=0.0,
        missing_ratio=0.0,
        random_state=0,
    )
    short_interest = panel["short_interest"] / (panel["adj_shares_outstanding"] * 1e6)
    analyst_dispersion = panel["eps_ntm_std"] / panel["adj_close"]

    assert np.nanmean(np.nanstd(short_interest, axis=0)) > 0.0
    assert np.nanmean(np.nanstd(analyst_dispersion, axis=0)) > 0.0

    proxy_mask = np.isfinite(short_interest) & np.isfinite(analyst_dispersion)
    assert (
        np.corrcoef(short_interest[proxy_mask], analyst_dispersion[proxy_mask])[0, 1]
        > 0.25
    )

    next_returns = panel["returns"][1:]
    next_returns -= np.nanmean(next_returns, axis=1, keepdims=True)
    short_interest = np.log(short_interest[:-1])
    short_interest -= np.nanmean(short_interest, axis=1, keepdims=True)
    short_mask = np.isfinite(short_interest) & np.isfinite(next_returns)
    assert np.corrcoef(short_interest[short_mask], next_returns[short_mask])[0, 1] < 0.0

    analyst_dispersion = np.log(analyst_dispersion[:-1])
    analyst_dispersion -= np.nanmean(analyst_dispersion, axis=1, keepdims=True)
    dispersion_mask = np.isfinite(analyst_dispersion) & np.isfinite(next_returns)
    assert (
        np.corrcoef(analyst_dispersion[dispersion_mask], next_returns[dispersion_mask])[
            0, 1
        ]
        < 0.0
    )


def test_active_mask_invariants(panel):
    active = panel.active_mask
    # Every observation has at least one active asset.
    assert active.any(axis=1).all()
    # Float fields are NaN exactly where inactive and finite on protected active cells.
    inactive = ~active
    for name in ("returns", "adj_close", "market_cap"):
        values = panel[name]
        assert np.isnan(values[inactive]).all()
        assert np.isfinite(values[active]).all()


def test_n_industries_controls_levels():
    panel = make_synthetic_characteristics(
        n_assets=40, n_observations=50, n_industries=4, random_state=1
    )
    labels = panel.decode_categorical_field("industry")
    present = set(np.unique(labels[panel.active_mask]))
    assert present <= {
        "Real Estate",
        "Software",
        "Banks",
        "Energy",
    }
    assert 1 <= len(present) <= 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_industries": 0},
        {"n_industries": 99},
        {"systematic_variance_ratio": 0.0},
        {"systematic_variance_ratio": 1.0},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_characteristics(n_assets=10, n_observations=30, **kwargs)
