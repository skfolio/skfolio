"""Base Prior estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb

from skfolio.prior._model import ReturnDistribution
from skfolio.typing import ArrayLike

_all__ = ["BasePrior"]


class BasePrior(skb.BaseEstimator, ABC):
    """Base class for all prior estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    return_distribution_: ReturnDistribution

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: ArrayLike, y=None, **fit_params):
        pass
