"""Descriptors that map `AssetPanel` columns to factor characteristics."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.base import BaseAssetPanelTransformer
from skfolio.descriptor._base import BaseDescriptor
from skfolio.descriptor._dividend_yield import (
    DividendToPrice,
    ForwardDividendToPrice,
    ShareholderYield,
)
from skfolio.descriptor._downside_risk import EWDownsideBeta
from skfolio.descriptor._earnings_quality import (
    AccrualsCashFlow,
    AnalystDispersionToPrice,
)
from skfolio.descriptor._earnings_yield import (
    EarningsToPrice,
    EbitdaToEnterpriseValue,
    ForwardEarningsToPrice,
)
from skfolio.descriptor._growth import (
    AssetsGrowthRate,
    CapexToAssetsChangeInIntensity,
    ChangeInIntensity,
    ChangeToScale,
    EarningsChangeToPrice,
    GrowthRate,
    IssuanceGrowthRate,
    SalesGrowthRate,
)
from skfolio.descriptor._leverage import (
    BookLeverage,
    DebtToAssets,
    MarketLeverage,
)
from skfolio.descriptor._liquidity import (
    EWAmihudIlliquidity,
    EWShareTurnover,
)
from skfolio.descriptor._lottery_demand import MaxReturn
from skfolio.descriptor._momentum import EWMomentum, RollingMomentum
from skfolio.descriptor._passthrough import Passthrough
from skfolio.descriptor._profitability import (
    AssetTurnover,
    CashFlowToAssets,
    GrossMargin,
    GrossProfitability,
    ReturnOnAssets,
    ReturnOnEquity,
    SalesToEnterpriseValue,
)
from skfolio.descriptor._reversal import Reversal
from skfolio.descriptor._sensitivity import (
    EWMacroSensitivity,
    EWMarketBeta,
)
from skfolio.descriptor._short_interest import (
    DaysToCover,
    ShortInterest,
)
from skfolio.descriptor._size import LogMarketCap
from skfolio.descriptor._value import (
    BookToPrice,
    CashFlowToPrice,
    SalesToPrice,
)
from skfolio.descriptor._volatility import (
    EWDownsideVolatility,
    EWResidualDownsideVolatility,
    EWResidualVolatility,
    EWVolatility,
)

__all__ = [
    "AccrualsCashFlow",
    "AnalystDispersionToPrice",
    "AssetTurnover",
    "AssetsGrowthRate",
    "BaseAssetPanelTransformer",
    "BaseDescriptor",
    "BookLeverage",
    "BookToPrice",
    "CapexToAssetsChangeInIntensity",
    "CashFlowToAssets",
    "CashFlowToPrice",
    "ChangeInIntensity",
    "ChangeToScale",
    "DaysToCover",
    "DebtToAssets",
    "DividendToPrice",
    "EWAmihudIlliquidity",
    "EWDownsideBeta",
    "EWDownsideVolatility",
    "EWMacroSensitivity",
    "EWMarketBeta",
    "EWMomentum",
    "EWResidualDownsideVolatility",
    "EWResidualVolatility",
    "EWShareTurnover",
    "EWVolatility",
    "EarningsChangeToPrice",
    "EarningsToPrice",
    "EbitdaToEnterpriseValue",
    "ForwardDividendToPrice",
    "ForwardEarningsToPrice",
    "GrossMargin",
    "GrossProfitability",
    "GrowthRate",
    "IssuanceGrowthRate",
    "LogMarketCap",
    "MarketLeverage",
    "MaxReturn",
    "Passthrough",
    "ReturnOnAssets",
    "ReturnOnEquity",
    "Reversal",
    "RollingMomentum",
    "SalesGrowthRate",
    "SalesToEnterpriseValue",
    "SalesToPrice",
    "ShareholderYield",
    "ShortInterest",
]
