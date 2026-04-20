"""Cross-sectional transformers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.preprocessing._transformer._cross_sectional._base import BaseCSTransformer
from skfolio.preprocessing._transformer._cross_sectional._cs_gaussian_rank_scaler import (
    CSGaussianRankScaler,
)
from skfolio.preprocessing._transformer._cross_sectional._cs_percentile_rank_scaler import (
    CSPercentileRankScaler,
)
from skfolio.preprocessing._transformer._cross_sectional._cs_standard_scaler import (
    CSStandardScaler,
)
from skfolio.preprocessing._transformer._cross_sectional._cs_tanh_shrinker import (
    CSTanhShrinker,
)
from skfolio.preprocessing._transformer._cross_sectional._cs_winsorizer import (
    CSWinsorizer,
)

__all__ = [
    "BaseCSTransformer",
    "CSGaussianRankScaler",
    "CSPercentileRankScaler",
    "CSStandardScaler",
    "CSTanhShrinker",
    "CSWinsorizer",
]
