"""Public exports for profitability descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.factor_model.descriptor._profitability._asset_turnover import (
    AssetTurnover,
)
from skfolio.factor_model.descriptor._profitability._cash_flow_to_assets import (
    CashFlowToAssets,
)
from skfolio.factor_model.descriptor._profitability._gross_margin import GrossMargin
from skfolio.factor_model.descriptor._profitability._gross_profitability import (
    GrossProfitability,
)
from skfolio.factor_model.descriptor._profitability._return_on_assets import (
    ReturnOnAssets,
)
from skfolio.factor_model.descriptor._profitability._return_on_equity import (
    ReturnOnEquity,
)
from skfolio.factor_model.descriptor._profitability._sales_to_enterprise_value import (
    SalesToEnterpriseValue,
)

__all__ = [
    "AssetTurnover",
    "CashFlowToAssets",
    "GrossMargin",
    "GrossProfitability",
    "ReturnOnAssets",
    "ReturnOnEquity",
    "SalesToEnterpriseValue",
]
