"""Gerber Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.utils.stats import corr_to_cov


class GerberCovariance(BaseCovariance):
    """Gerber Covariance estimator.

    Robust co-movement measure which ignores fluctuations below a certain threshold
    while simultaneously limiting the effects of extreme movements.
    The Gerber statistic extends Kendall's Tau by counting the proportion of
    simultaneous co-movements in series when their amplitudes exceed data-dependent
    thresholds.

    Three variant has been published:

        * Gerber et al. (2015): tend to produce matrices that are non-PSD.
        * Gerber et al. (2019): alteration of the denominator of the above statistic.
        * Gerber et al. (2022): final alteration to ensure PSD matrix.

    The last two variants are implemented.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    threshold : float, default=0.5
        Gerber threshold. The default value is `0.5`.

    psd_variant : bool, default=True
        If this is set to True, the Gerber et al. (2022) variant is used to ensure a
        positive semi-definite matrix.
        Otherwise, the Gerber et al. (2019) variant is used.
        The default is `True`.

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
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "The gerber statistic: A robust co-movement measure for portfolio
        optimization".
        The Journal of Portfolio Management.
        Gerber, S., B. Javid, H. Markowitz, P. Sargen, and D. Starer (2022).

    .. [2]  "The gerber statistic: A robust measure of correlation".
        Gerber, S., B. Javid, H. Markowitz, P. Sargen, and D. Starer (2019).

    .. [3]  "Enhancing multi-asset portfolio construction under modern portfolio theory
        with a robust co-movement measure".
        Social Science Research network Working Paper Series.
        Gerber, S., H. Markowitz, and P. Pujara (2015).

    .. [4]  "Deconstructing the Gerber Statistic".
        Flint & Polakow, 2023.
    """

    def __init__(
        self,
        window_size: int | None = None,
        threshold: float = 0.5,
        psd_variant: bool = True,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.threshold = threshold
        self.psd_variant = psd_variant

    def fit(self, X: npt.ArrayLike, y=None) -> "GerberCovariance":
        """Fit the Gerber covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
          Price returns of the assets.

        y : Ignored
           Not used, present for API consistency by convention.

        Returns
        -------
        self : GerberCovariance
           Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        if not (1 > self.threshold > 0):
            raise ValueError("The threshold must be between 0 and 1")
        n_observations = X.shape[0]
        std = X.std(axis=0).reshape((-1, 1))
        u = X >= std.T * self.threshold
        d = X <= -std.T * self.threshold
        n = np.invert(u) & np.invert(d)  # np.invert preferred that ~ for type hint
        n = n.astype(int)
        u = u.astype(int)
        d = d.astype(int)
        concordant = u.T @ u + d.T @ d
        discordant = u.T @ d + d.T @ u
        h = concordant - discordant
        if self.psd_variant:
            corr = h / (n_observations - n.T @ n)
        else:
            h_sqrt = np.sqrt(np.diag(h)).reshape((-1, 1))
            corr = h / (h_sqrt @ h_sqrt.T)
        covariance = corr_to_cov(corr, std.reshape(-1))
        self._set_covariance(covariance)
        return self
