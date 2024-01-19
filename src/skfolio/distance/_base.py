"""Base Distance Estimators"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb


class BaseDistance(skb.BaseEstimator, ABC):
    """Base class for all distance estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.
    """

    codependence_: np.ndarray
    distance_: np.ndarray

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None) -> "BaseDistance":
        """Fit the Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BaseDistance
            Fitted estimator.
        """
        pass
