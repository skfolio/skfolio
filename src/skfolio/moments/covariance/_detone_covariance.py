"""Covariance Detoning Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._empirical_covariance import EmpiricalCovariance
from skfolio.utils.stats import corr_to_cov, cov_to_corr
from skfolio.utils.tools import check_estimator


class DetoneCovariance(BaseCovariance):
    """Covariance Detoning estimator.

    Financial covariance matrices usually incorporate a market component corresponding
    to the first eigenvectors [1]_.
    For some applications like clustering, removing the market component (loud tone)
    allow a greater portion of the covariance to be explained by components that affect
    specific subsets of the securities.

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` to estimate the covariance
        matrix prior detoning.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    n_markets : int, default=1
        Number of eigenvectors related to the market.
        The default value is `1`.

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

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "Machine Learning for Asset Managers".
        Elements in Quantitative Finance.
        Lòpez de Prado (2020).
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        n_markets: float = 1,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.covariance_estimator = covariance_estimator
        self.n_markets = n_markets

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            covariance_estimator=self.covariance_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "DetoneCovariance":
        """Fit the Covariance Detoning estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
          Price returns of the assets.

        y : Ignored
           Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : DetoneCovariance
           Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        # noinspection PyArgumentList
        self.covariance_estimator_.fit(X, y, **routed_params.covariance_estimator.fit)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        _ = skv.validate_data(self, X)
        corr, std = cov_to_corr(self.covariance_estimator_.covariance_)
        e_val, e_vec = np.linalg.eigh(corr)
        indices = e_val.argsort()[::-1]
        e_val, e_vec = e_val[indices], e_vec[:, indices]
        # market eigenvalues and eigenvectors
        market_e_val, market_e_vec = e_val[: self.n_markets], e_vec[:, : self.n_markets]
        # market correlation
        market_corr = market_e_vec @ np.diag(market_e_val) @ market_e_vec.T
        # Removing the market correlation
        corr -= market_corr
        corr, _ = cov_to_corr(corr)
        covariance = corr_to_cov(corr, std)
        self._set_covariance(covariance)
        return self
