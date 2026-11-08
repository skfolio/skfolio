"""Distance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.distance._base import BaseDistance
from skfolio.distance._covariance_distance import CovarianceDistance
from skfolio.distance._distance_correlation import DistanceCorrelation
from skfolio.distance._kendall_distance import KendallDistance
from skfolio.distance._mutual_information import MutualInformation
from skfolio.distance._pearson_distance import PearsonDistance
from skfolio.distance._spearman_distance import SpearmanDistance
from skfolio.utils.stats import NBinsMethod

__all__ = [
    "BaseDistance",
    "CovarianceDistance",
    "DistanceCorrelation",
    "KendallDistance",
    "MutualInformation",
    "NBinsMethod",
    "PearsonDistance",
    "SpearmanDistance",
]
