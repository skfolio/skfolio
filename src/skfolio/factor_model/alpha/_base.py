"""Abstract base class for factor-model alpha estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb

from skfolio.containers import AssetPanel
from skfolio.typing import FloatArray, ObjArray

__all__ = ["BaseAlpha"]


class BaseAlpha(skb.BaseEstimator, ABC):
    """Base class for all Alpha estimators in skfolio."""

    alpha_: FloatArray
    n_assets_: int
    asset_names_: ObjArray

    @abstractmethod
    def fit(self, X: AssetPanel, y=None, **fit_params) -> BaseAlpha:
        pass
