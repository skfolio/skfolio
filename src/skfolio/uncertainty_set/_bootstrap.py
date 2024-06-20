"""Bootstrap Uncertainty Set estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import scipy.stats as st

from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
    UncertaintySet,
)
from skfolio.utils.bootstrap import stationary_bootstrap
from skfolio.utils.tools import check_estimator


class BootstrapMuUncertaintySet(BaseMuUncertaintySet):
    r"""Bootstrap Mu Uncertainty set.

    Compute the expected returns ellipsoidal uncertainty set using circular bootstrap:

    .. math:: U_{\mu}=\left\{\mu\,|\left(\mu-\hat{\mu}\right)S^{-1}\left(\mu-\hat{\mu}\right)^{T}\leq\kappa^{2}\right\}

    The size of the ellipsoid  :math:`\kappa` (confidence region), is computed using:

    .. math:: \kappa^2 = \chi^2_{n\_assets} (\beta)

    with :math:`\chi^2_{n\_assets}(\beta)` the inverse cumulative distribution function
    of the chi-squared distribution with `n_assets` degrees of freedom at the
    :math:`\beta` confidence level.

    The Shape of the ellipsoid :math:`S` is computed using stationary bootstrap with
    the option to force the non-diagonal elements of the covariance matrix to zero.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        The :ref:`prior estimator <prior>` used to estimate the assets covariance
        matrix. The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    confidence_level : float , default=0.95
        Confidence level :math:`\beta` of the inverse cumulative distribution function
        of the chi-squared distribution. The default value is `0.95`.

    diagonal : bool, default=True
        If this is set to True, the non-diagonal elements of the covariance matrix are
        set to zero.

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
    .. [1]  "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [2]  "Automatic Block-Length Selection for the Dependent Bootstrap",
        Politis & White (2004).

    .. [3]  "Correction to Automatic Block-Length Selection for the Dependent
        Bootstrap",
        Patton, Politis & White (2009).
    """

    prior_estimator_: BasePrior

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        seed: int | None = None,
    ):
        self.prior_estimator = prior_estimator
        self.confidence_level = confidence_level
        self.diagonal = diagonal
        self.n_bootstrap_samples = n_bootstrap_samples
        self.block_size = block_size
        self.seed = seed

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> "BootstrapMuUncertaintySet":
        """Fit the Bootstrap Mu Uncertainty set estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors.
            The default is `None`.

        Returns
        -------
        self : BootstrapMuUncertaintySet
            Fitted estimator.
        """
        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        # fitting estimators
        self.prior_estimator_.fit(X, y)
        mu = self.prior_estimator_.prior_model_.mu
        returns = self.prior_estimator_.prior_model_.returns
        n_assets = returns.shape[1]
        k = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=n_assets))
        samples = stationary_bootstrap(
            returns=returns,
            block_size=self.block_size,
            n_bootstrap_samples=self.n_bootstrap_samples,
            seed=self.seed,
        )
        mus = np.mean(samples, axis=1)
        covs = np.zeros((self.n_bootstrap_samples, n_assets, n_assets))
        for i in range(self.n_bootstrap_samples):
            covs[i] = np.cov(samples[i].T)

        sigma = np.cov((mus - mu).T)
        if self.diagonal:
            sigma = np.diag(np.diag(sigma))

        self.uncertainty_set_ = UncertaintySet(k=k, sigma=sigma)
        return self


class BootstrapCovarianceUncertaintySet(BaseCovarianceUncertaintySet):
    r"""Bootstrap Covariance Uncertainty set.

    Compute the covariance ellipsoidal uncertainty set using circular bootstrap:

    .. math:: U_{\Sigma}=\left\{\Sigma\,|\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)S^{-1}\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)^{T}\leq k^{2}\,,\,\Sigma\succeq 0\right\}

    The size of the ellipsoid :math:`\kappa` (confidence region), is computed using:

    .. math:: \kappa^2 = \chi^2_{n\_assets^2} (\beta)

    with :math:`\chi^2_{n\_assets^2}(\beta)` the inverse cumulative distribution
    function of the chi-squared distribution with `n_assets` degrees of freedom at the
    :math:`\beta` confidence level.

    The Shape of the ellipsoid :math:`S` is computed using stationary bootstrap with the
    option to force the non-diagonal elements of the covariance matrix to zero.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        The :ref:`prior estimator <prior>` used to estimate the assets covariance
        matrix. The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    confidence_level : float , default=0.95
        Confidence level :math:`\beta` of the inverse cumulative distribution function
        of the chi-squared distribution. The default value is `0.95`.

    diagonal : bool, default=True
        If this is set to True, the non-diagonal elements of the covariance matrix are
        set to zero.

    n_bootstrap_samples : int, default=1000
        Number of bootstrap samples to generate. The default value is `1000`.

    block_size : float, optional
        Bootstrap block size. The default (`None`) is to estimate the optimal block size
        using Politis & White algorithm for all individual assets.

    seed : int, optional
        Random seed used to initialize the pseudo-random number generator

    Attributes
    ----------
    uncertainty_set_ : UncertaintySet
        Covariance Uncertainty set :class:`~skfolio.uncertainty_set.UncertaintySet`.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    References
    ----------
    .. [1]  "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [2]  "Automatic Block-Length Selection for the Dependent Bootstrap",
        Politis & White (2004).

    .. [3]  "Correction to Automatic Block-Length Selection for the Dependent
        Bootstrap",
        Patton, Politis & White (2009).
    """

    prior_estimator_: BasePrior

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        seed: int | None = None,
    ):
        self.prior_estimator = prior_estimator
        self.confidence_level = confidence_level
        self.diagonal = diagonal
        self.n_bootstrap_samples = n_bootstrap_samples
        self.block_size = block_size
        self.seed = seed

    def fit(self, X: npt.ArrayLike, y=None) -> "BootstrapCovarianceUncertaintySet":
        """Fit the Bootstrap Covariance Uncertainty set estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors.
            The default is `None`.

        Returns
        -------
        self : EmpiricalCovarianceUncertaintySet
            Fitted estimator.
        """

        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        # fitting estimators
        self.prior_estimator_.fit(X, y)
        covariance = self.prior_estimator_.prior_model_.covariance
        returns = self.prior_estimator_.prior_model_.returns
        n_assets = returns.shape[1]
        k = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=n_assets**2))

        samples = stationary_bootstrap(
            returns=returns,
            block_size=self.block_size,
            n_bootstrap_samples=self.n_bootstrap_samples,
            seed=self.seed,
        )
        covs = np.zeros((self.n_bootstrap_samples, n_assets, n_assets))
        for i in range(self.n_bootstrap_samples):
            covs[i] = np.cov(samples[i].T)

        sigma = np.cov(
            (covs - covariance)
            .reshape((self.n_bootstrap_samples, n_assets**2), order="F")
            .T
        )
        if self.diagonal:
            sigma = np.diag(np.diag(sigma))

        self.uncertainty_set_ = UncertaintySet(k=k, sigma=sigma)
        return self
