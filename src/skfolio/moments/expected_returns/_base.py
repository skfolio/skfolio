"""Base Expected returns estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb

from skfolio.typing import ArrayLike, FloatArray


class BaseMu(skb.BaseEstimator, ABC):
    """Base class for all expected returns estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    mu_: FloatArray

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: ArrayLike, y=None):
        pass
