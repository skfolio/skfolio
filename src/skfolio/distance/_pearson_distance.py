"""Pearson distance estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn.utils.validation as skv

from skfolio.distance._base import BaseDistance
from skfolio.distance._utils import _corr_to_distance
from skfolio.typing import ArrayLike


class PearsonDistance(BaseDistance):
    r"""Pearson Distance estimator.

    The codependence is computed from the Pearson correlation to which is applied a
    power and/or absolute transformation.
    This codependence is then used to compute the distance matrix.
    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`

    Parameters
    ----------
    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        López de Prado, Journal of Portfolio Management (2016)
    """

    def __init__(self, absolute: bool = False, power: float = 1) -> None:
        self.absolute = absolute
        self.power = power

    def fit(self, X: ArrayLike, y: None = None) -> PearsonDistance:
        """Fit the Pearson Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : PearsonDistance
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        corr = np.corrcoef(X.T)
        self.codependence_, self.distance_ = _corr_to_distance(
            corr, absolute=self.absolute, power=self.power
        )
        return self
