"""Base Prior estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb

from skfolio.prior._model import ReturnDistribution
from skfolio.typing import ArrayLike

__all__ = ["BasePrior"]


class BasePrior(skb.BaseEstimator, ABC):
    """Base class for all prior estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their `__init__` as explicit keyword
    arguments (no `*args` or `**kwargs`).
    """

    return_distribution_: ReturnDistribution

    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def fit(self, X: ArrayLike, y=None, **fit_params):
        """Fit the prior estimator and set `return_distribution_`.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using `sklearn.set_config(enable_metadata_routing=True)`.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : BasePrior
            Fitted estimator.
        """
        ...
