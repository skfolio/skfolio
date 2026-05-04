"""Bootstrap Uncertainty Set estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.stats as st
import sklearn.utils.metadata_routing as skm

from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.typing import ArrayLike
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
)
from skfolio.uncertainty_set._model import UncertaintySet
from skfolio.utils.bootstrap import stationary_bootstrap
from skfolio.utils.tools import check_estimator


class BootstrapMuUncertaintySet(BaseMuUncertaintySet):
    r"""Bootstrap Mu Uncertainty set.

    Compute the expected returns ellipsoidal uncertainty set using stationary bootstrap:

    .. math::

        U_{\mu}
        =
        \left\{
            \mu :
            (\mu - \hat{\mu})^\top S^{-1}(\mu - \hat{\mu})
            \le \kappa^2
        \right\}.

    The radius of the ellipsoid :math:`\kappa` (confidence region) is computed using:

    .. math:: \kappa^2 = \chi^2_{n_{\text{assets}}}(\beta)

    with :math:`\chi^2_{n_{\text{assets}}}(\beta)` the inverse cumulative distribution
    function of the chi-squared distribution with :math:`n_{\text{assets}}` degrees of
    freedom at the :math:`\beta` confidence level.

    The shape matrix :math:`S` of the ellipsoid is computed using stationary bootstrap,
    with the option to retain only its diagonal. The estimator stores the square-root
    factor as the linear geometry map:math:`L = S^{1/2}`.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        The :ref:`prior estimator <prior>` used to estimate the assets return
        distribution. The default (`None`) is to use
        :class:`~skfolio.prior.EmpiricalPrior`.

    confidence_level : float , default=0.95
        Confidence level :math:`\beta` of the inverse cumulative distribution function
        of the chi-squared distribution. The default value is `0.95`.

    diagonal : bool, default=True
        If `True`, only the diagonal of the ellipsoid shape matrix is retained.

    n_bootstrap_samples : int, default=1000
        Number of bootstrap samples to generate. The default value is `1000`.

    block_size : float, optional
        Bootstrap block size. The default (`None`) is to estimate the optimal block size
        using Politis & White algorithm for all individual assets.

    seed : int, optional
        Random seed used to initialize the pseudo-random number generator.

    Attributes
    ----------
    uncertainty_set_ : UncertaintySet
        Mu Uncertainty set :class:`~skfolio.uncertainty_set.UncertaintySet`.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    References
    ----------
    .. [1] "Portfolio Optimization: Theory and Application", Chapter 14,
        Daniel P. Palomar (2025)

    .. [2] "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [3] "Automatic Block-Length Selection for the Dependent Bootstrap",
        Politis & White (2004).

    .. [4] "Correction to Automatic Block-Length Selection for the Dependent
        Bootstrap",
        Patton, Politis & White (2009).
    """

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        seed: int | None = None,
    ):
        super().__init__(prior_estimator=prior_estimator)
        self.confidence_level = confidence_level
        self.diagonal = diagonal
        self.n_bootstrap_samples = n_bootstrap_samples
        self.block_size = block_size
        self.seed = seed

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params
    ) -> BootstrapMuUncertaintySet:
        """Fit the Bootstrap Mu Uncertainty set estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors.
            The default is `None`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using `sklearn.set_config(enable_metadata_routing=True)`.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : BootstrapMuUncertaintySet
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        # fitting estimators
        self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
        mu = self.prior_estimator_.return_distribution_.mu
        returns = self.prior_estimator_.return_distribution_.returns
        n_assets = returns.shape[1]
        radius = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=n_assets))
        samples = stationary_bootstrap(
            returns=returns,
            block_size=self.block_size,
            n_bootstrap_samples=self.n_bootstrap_samples,
            seed=self.seed,
        )
        mus = np.mean(samples, axis=1)
        deviations = mus - mu
        if self.diagonal:
            geometry = np.diag(np.sqrt(np.var(deviations, axis=0, ddof=1)))
        else:
            geometry = sla.sqrtm(np.cov(deviations, rowvar=False)).real

        self.uncertainty_set_ = UncertaintySet(radius=radius, geometry=geometry, norm=2)
        return self


class BootstrapCovarianceUncertaintySet(BaseCovarianceUncertaintySet):
    r"""Bootstrap Covariance Uncertainty set.

    Compute the covariance ellipsoidal uncertainty set using stationary bootstrap:

    .. math::

        U_{\Sigma}
        =
        \left\{
            \Sigma :
            d^\top S^{-1} d \le \kappa^2,
            \Sigma \succeq 0
        \right\},
        \quad
        d =
        \operatorname{vec}(\Sigma) - \operatorname{vec}(\hat{\Sigma}).

    The radius of the ellipsoid :math:`\kappa` (confidence region) is computed using:

    .. math:: \kappa^2 = \chi^2_{n_{\text{assets}}^2}(\beta)

    with :math:`\chi^2_{n_{\text{assets}}^2}(\beta)` the inverse cumulative
    distribution function of the chi-squared distribution with :math:`n_{\text{assets}}^2`
    degrees of freedom at the :math:`\beta` confidence level.

    The shape matrix :math:`S` of the ellipsoid is the covariance matrix of the
    bootstrapped vectorized covariance estimator. If `diagonal` is `True`, only the
    diagonal of :math:`S` is retained and the linear geometry map :math:`L` is built
    directly from it. Otherwise, the estimator stores a full square-root factor
    :math:`L = S^{1/2}`.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        The :ref:`prior estimator <prior>` used to estimate the assets return
        distribution. The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    confidence_level : float , default=0.95
        Confidence level :math:`\beta` of the inverse cumulative distribution function
        of the chi-squared distribution. The default value is `0.95`.

    diagonal : bool, default=True
        If `True`, only the diagonal of the ellipsoid shape matrix in vectorized
        covariance space is retained.

    n_bootstrap_samples : int, default=1000
        Number of bootstrap samples to generate. The default value is `1000`.

    block_size : float, optional
        Bootstrap block size. The default (`None`) is to estimate the optimal block size
        using Politis & White algorithm for all individual assets.

    seed : int, optional
        Random seed used to initialize the pseudo-random number generator.

    Attributes
    ----------
    uncertainty_set_ : UncertaintySet
        Covariance Uncertainty set :class:`~skfolio.uncertainty_set.UncertaintySet`.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    References
    ----------
    .. [1] "Portfolio Optimization: Theory and Application", Chapter 14,
        Daniel P. Palomar (2025)

    .. [2] "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [3] "Automatic Block-Length Selection for the Dependent Bootstrap",
        Politis & White (2004).

    .. [4] "Correction to Automatic Block-Length Selection for the Dependent
        Bootstrap",
        Patton, Politis & White (2009).
    """

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        seed: int | None = None,
    ):
        super().__init__(prior_estimator=prior_estimator)
        self.confidence_level = confidence_level
        self.diagonal = diagonal
        self.n_bootstrap_samples = n_bootstrap_samples
        self.block_size = block_size
        self.seed = seed

    def fit(
        self, X: ArrayLike, y=None, **fit_params
    ) -> BootstrapCovarianceUncertaintySet:
        """Fit the Bootstrap Covariance Uncertainty set estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors.
            The default is `None`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using `sklearn.set_config(enable_metadata_routing=True)`.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : BootstrapCovarianceUncertaintySet
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        # fitting estimators
        self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
        returns = self.prior_estimator_.return_distribution_.returns
        n_assets = returns.shape[1]
        radius = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=n_assets**2))

        samples = stationary_bootstrap(
            returns=returns,
            block_size=self.block_size,
            n_bootstrap_samples=self.n_bootstrap_samples,
            seed=self.seed,
        )
        deviations = np.empty((self.n_bootstrap_samples, n_assets**2))
        for i, sample in enumerate(samples):
            deviations[i] = np.cov(sample.T).ravel(order="F")

        if self.diagonal:
            geometry = np.diag(np.sqrt(np.var(deviations, axis=0, ddof=1)))
        else:
            geometry = sla.sqrtm(np.cov(deviations, rowvar=False)).real

        self.uncertainty_set_ = UncertaintySet(radius=radius, geometry=geometry, norm=2)
        return self
