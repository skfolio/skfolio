"""Tests for Profitability descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    AssetTurnover,
    CashFlowToAssets,
    GrossMargin,
    GrossProfitability,
    ReturnOnAssets,
    ReturnOnEquity,
    SalesToEnterpriseValue,
)


# Formula inputs are explicit so expected results do not use descriptor internals.
@pytest.fixture(
    params=[
        pytest.param(
            (ReturnOnAssets, "net_income_ttm", "total_assets", None), id="roa"
        ),
        pytest.param((AssetTurnover, "sales_ttm", "total_assets", None), id="turnover"),
        pytest.param(
            (CashFlowToAssets, "operating_cash_flow_ttm", "total_assets", None),
            id="cashflow",
        ),
        pytest.param(
            (SalesToEnterpriseValue, "sales_ttm", "enterprise_value", None),
            id="sales-ev",
        ),
        pytest.param((ReturnOnEquity, "net_income_ttm", "book_equity", None), id="roe"),
        pytest.param(
            (GrossProfitability, "sales_ttm", "total_assets", "cost_of_revenue_ttm"),
            id="gross-profitability",
        ),
        pytest.param(
            (GrossMargin, "sales_ttm", "sales_ttm", "cost_of_revenue_ttm"),
            id="gross-margin",
        ),
    ]
)
def ratio_case(request):
    return request.param


def test_shape_formula_and_stateless_update(simple_panel, ratio_case):
    descriptor_cls, numerator, denominator, subtraction = ratio_case
    descriptor = descriptor_cls()
    result = descriptor.fit_transform(simple_panel)
    assert result.shape == simple_panel[numerator].shape
    expected_numerator = simple_panel[numerator]
    if subtraction is not None:
        expected_numerator = expected_numerator - simple_panel[subtraction]
    np.testing.assert_array_equal(
        result, expected_numerator / simple_panel[denominator]
    )
    np.testing.assert_allclose(
        descriptor.partial_fit_transform(simple_panel),
        result,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "descriptor_cls,fields",
    [
        (ReturnOnAssets, ("net_income_ttm", "total_assets")),
        (AssetTurnover, ("sales_ttm", "total_assets")),
        (CashFlowToAssets, ("operating_cash_flow_ttm", "total_assets")),
        (SalesToEnterpriseValue, ("sales_ttm", "enterprise_value")),
        (ReturnOnEquity, ("net_income_ttm",)),
        (GrossProfitability, ("sales_ttm", "cost_of_revenue_ttm", "total_assets")),
        (GrossMargin, ("sales_ttm", "cost_of_revenue_ttm")),
    ],
)
def test_nan_propagation(simple_panel, descriptor_cls, fields):
    for i, field in enumerate(fields):
        simple_panel[field][3 + i, i] = np.nan
    result = descriptor_cls().fit_transform(simple_panel)
    for i in range(len(fields)):
        assert np.isnan(result[3 + i, i])


@pytest.mark.parametrize("negative", [False, True], ids=["zero", "negative"])
def test_nonpositive_denominator_is_nan(simple_panel, ratio_case, negative):
    descriptor_cls, _, denominator, _ = ratio_case
    value = -1e8 if descriptor_cls is ReturnOnEquity else -1.0
    simple_panel[denominator][5, 0] = value if negative else 0.0
    result = descriptor_cls().fit_transform(simple_panel)
    assert np.isnan(result[5, 0])


@pytest.mark.parametrize(
    "descriptor_cls,denominator",
    [
        (ReturnOnAssets, "total_assets"),
        (AssetTurnover, "total_assets"),
        (CashFlowToAssets, "total_assets"),
        (SalesToEnterpriseValue, "enterprise_value"),
        (GrossProfitability, "total_assets"),
    ],
)
def test_infinite_denominator_raises(simple_panel, descriptor_cls, denominator):
    simple_panel[denominator][5, 0] = np.inf
    with pytest.raises(ValueError, match=rf"{denominator}.*finite"):
        descriptor_cls().fit_transform(simple_panel)


def test_positive_equity_unaffected(simple_panel):
    simple_panel["book_equity"][5, 0] = -1e8
    result = ReturnOnEquity().fit_transform(simple_panel)
    assert not np.isnan(result[4, 0])
    assert not np.isnan(result[6, 0])
