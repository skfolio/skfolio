"""Exponentially Weighted Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import warnings

import numpy.typing as npt
import pandas as pd
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance


class EWCovariance(BaseCovariance):
    r"""Exponentially Weighted Covariance estimator.

    Estimator of the covariance using the historical exponentially weighted returns.

    For factor model applications requiring bias-adjusted volatility forecasts see
    :class:`BiasAdjustedEWCovariance`.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    decay_factor : float, default=0.97
        EWMA decay factor (:math:`\lambda`) for covariance estimation.

        Higher values produce **more stable** (robust) estimates; lower values are
        **more responsive** (adaptive) but noisier:

        * :math:`\lambda \to 1`: Very stable and slow to adapt (robust to noise)
        * :math:`\lambda \to 0`: Very responsive and fast to adapt (sensitive to noise)

        **Relationship to half-life:**

        The half-life is the number of observations for the weight to decay to 50%.
        :math:`\text{half-life} = -\ln(2) / \ln(\lambda)`

        **For example**:

        * :math:`\lambda = 0.97`: 23-day half-life
        * :math:`\lambda = 0.94`: 11-day half-life
        * :math:`\lambda = 0.90`: 6-day half-life
        * :math:`\lambda = 0.80`: 3-day half-life

        **Note:** For portfolio optimization, more stable values (≥ 0.94) are generally
        preferred to avoid excessive turnover from estimation noise.

        Must satisfy :math:`0 < \lambda < 1`.

    alpha : float, optional
        .. deprecated:: 0.x.0
            `alpha` is deprecated and will be removed in version 2.O.
            Use `decay_factor` instead. Note: `alpha = 1 - decay_factor`.
            The default value will change from `alpha=0.2` (decay_factor=0.8)
            to `decay_factor=0.97` for improved stability.

    nearest : bool, default=True
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition than can be
        computed. The variance is left unchanged.
        A covariance matrix that is not positive definite often occurs in high
        dimensional problems. It can be due to multicollinearity, floating-point
        inaccuracies, or when the number of observations is smaller than the number of
        assets. For more details, see :func:`~skfolio.utils.stats.cov_nearest`.
        The default is `True`.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm is used to find the
        nearest PD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and uses the clipping method as the
        Higham algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iterations of the Higham (2002) algorithm.
        The default value is `100`.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    def __init__(
        self,
        window_size: int | None = None,
        decay_factor: float | None = None,
        alpha: float | None = None,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ) -> None:
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.alpha = alpha

    def fit(self, X: npt.ArrayLike, y=None) -> EWCovariance:
        """Fit the Exponentially Weighted Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : EWCovariance
          Fitted estimator.
        """
        X = skv.validate_data(self, X)

        # Handle backward compatibility for alpha parameter
        if self.alpha is not None and self.decay_factor is not None:
            raise ValueError(
                "Cannot specify both 'alpha' and 'decay_factor'. "
                "Use 'decay_factor' (alpha is deprecated)."
            )

        if self.alpha is not None:
            warnings.warn(
                "The 'alpha' parameter is deprecated and will be removed in version 2.0. "
                "Use 'decay_factor' instead, where decay_factor = 1 - alpha. "
                "Note: The default will change from alpha=0.2 (decay_factor=0.8) "
                "to decay_factor=0.97 for improved stability in portfolio optimization.",
                FutureWarning,
                stacklevel=2,
            )

        if self.alpha is not None:
            decay_factor = 1.0 - self.alpha
        elif self.decay_factor is not None:
            decay_factor = self.decay_factor
        else:
            decay_factor = 0.97  # TODO set in __init__ after depreciation

        if self.window_size is not None:
            X = X[-int(self.window_size) :]
        n_observations = X.shape[0]

        covariance = (
            pd.DataFrame(X)
            .ewm(alpha=1.0 - decay_factor)
            .cov()
            .loc[(n_observations - 1, slice(None)), :]
            .to_numpy()
        )
        self._set_covariance(covariance)
        return self
