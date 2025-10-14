"""Exponentially Weighted Expected Returns (Mu) Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy.typing as npt
import pandas as pd
import sklearn.utils.validation as skv

from skfolio.moments.expected_returns._base import BaseMu


class EWMu(BaseMu):
    r"""Exponentially Weighted Expected Returns (Mu) estimator.

    Estimates the expected returns with the exponentially weighted mean (EWM).

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    alpha : float, default=0.2
        Exponential smoothing factor. The default value is `0.2`.

        :math:`0 < \alpha \leq 1`.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
       Estimated expected returns of the assets.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    def __init__(self, window_size: int | None = None, alpha: float = 0.2):
        self.window_size = window_size
        self.alpha = alpha

    def fit(self, X: npt.ArrayLike, y=None) -> "EWMu":
        """Fit the EWMu estimator model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EWMu
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        self.mu_ = pd.DataFrame(X).ewm(alpha=self.alpha).mean().iloc[-1, :].to_numpy()
        return self
