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
        "Software & Services",
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
