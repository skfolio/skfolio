"""Descriptors that map `AssetPanel` columns to factor characteristics."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.factor_model._base import BaseAssetPanelTransformer
from skfolio.factor_model.descriptor._base import BaseDescriptor
from skfolio.factor_model.descriptor._dividend_yield import (
    DividendToPrice,
    ForwardDividendToPrice,
    ShareholderYield,
)
from skfolio.factor_model.descriptor._downside_risk import EWDownsideBeta
from skfolio.factor_model.descriptor._earnings_quality import (
    AccrualsCashFlow,
    AnalystDispersionToPrice,
)
from skfolio.factor_model.descriptor._earnings_yield import (
    EarningsToPrice,
    EbitdaToEnterpriseValue,
    ForwardEarningsToPrice,
)
from skfolio.factor_model.descriptor._growth import (
    AssetsGrowthRate,
    CapexToAssetsChangeInIntensity,
    ChangeInIntensity,
    ChangeToScale,
    EarningsChangeToPrice,
    GrowthRate,
    IssuanceGrowthRate,
    SalesGrowthRate,
)
from skfolio.factor_model.descriptor._leverage import (
    BookLeverage,
    DebtToAssets,
    MarketLeverage,
)
from skfolio.factor_model.descriptor._liquidity import (
    EWAmihudIlliquidity,
    EWShareTurnover,
)
from skfolio.factor_model.descriptor._lottery_demand import MaxReturn
from skfolio.factor_model.descriptor._momentum import EWMomentum, RollingMomentum
from skfolio.factor_model.descriptor._passthrough import Passthrough
from skfolio.factor_model.descriptor._profitability import (
    AssetTurnover,
    CashFlowToAssets,
    GrossMargin,
    GrossProfitability,
    ReturnOnAssets,
    ReturnOnEquity,
    SalesToEnterpriseValue,
)
from skfolio.factor_model.descriptor._reversal import Reversal
from skfolio.factor_model.descriptor._sensitivity import (
    EWMacroSensitivity,
    EWMarketBeta,
)
from skfolio.factor_model.descriptor._short_interest import (
    DaysToCover,
    ShortInterest,
)
from skfolio.factor_model.descriptor._size import LogMarketCap
from skfolio.factor_model.descriptor._value import (
    BookToPrice,
    CashFlowToPrice,
    SalesToPrice,
)
from skfolio.factor_model.descriptor._volatility import (
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
