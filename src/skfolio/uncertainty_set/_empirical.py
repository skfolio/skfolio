"""Empirical Uncertainty Set estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import scipy.stats as st
import sklearn.utils.metadata_routing as skm

from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
    UncertaintySet,
)
from skfolio.utils.stats import commutation_matrix
from skfolio.utils.tools import check_estimator


class EmpiricalMuUncertaintySet(BaseMuUncertaintySet):
    r"""Empirical Mu Uncertainty Set.

    Compute the expected returns ellipsoidal uncertainty set [1]_:

    .. math:: U_{\mu}=\left\{\mu\,|\left(\mu-\hat{\mu}\right)S^{-1}\left(\mu-\hat{\mu}\right)^{T}\leq\kappa^{2}\right\}

    Under the assumption that :math:`\Sigma` is given, the distribution of the sample
    estimator :math:`\hat{\mu}` based on an i.i.d. sample
    :math:`R_{t}\sim N(\mu, \Sigma), t=1,...,T` is given by
    :math:`\hat{\mu}\sim N(\mu, \frac{1}{T}\Sigma)`.

    The size of the ellipsoid  :math:`\kappa` (confidence region), is computed using:

    .. math:: \kappa^2 = \chi^2_{n\_assets} (\beta)

    with :math:`\chi^2_{n\_assets}(\beta)` the inverse cumulative distribution function
    of the chi-squared distribution with `n_assets` degrees of freedom at the
    :math:`\beta` confidence level.

    The shape of the ellipsoid :math:`S` is computed using:

    .. math:: S = \frac{1}{T}\Sigma

    with the option to force the non-diagonal elements of the covariance matrix to zero.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        The :ref:`prior estimator <prior>` used to estimate the assets covariance
        matrix. The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    confidence_level : float, default=0.95
        Confidence level :math:`\beta` of the inverse cumulative distribution function
        of the chi-squared distribution. The default value is `0.95`.

    diagonal : bool, default=True
        If this is set to True, the non-diagonal elements of the covariance matrix are
        set to zero.

    n_eff : float, optional
        Effective number of observations used for the mean estimator. If `None`,
        the number of observations in `X` is used. This is useful when the expected
        returns are estimated using a different window length or a weighted estimator
        (e.g. EWMA), in which case `n_eff` should be set to the corresponding effective
        sample size.

    Attributes
    ----------
    uncertainty_set_ : UncertaintySet
        Mu Uncertainty set :class:`~skfolio.uncertainty_set.UncertaintySet`.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    n_eff_ : float
        Effective number of observations actually used to build the uncertainty set.

    References
    ----------
    .. [1]  "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [2] "Portfolio Optimization: Theory and Application", Chapter 14,
        Daniel P. Palomar (2025)
    """

    n_eff_: float

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_eff: float | None = None,
    ):
        super().__init__(prior_estimator=prior_estimator)
        self.confidence_level = confidence_level
        self.diagonal = diagonal
        self.n_eff = n_eff

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
    ) -> "EmpiricalMuUncertaintySet":
        """Fit the Empirical Mu Uncertainty set estimator.

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
        self : EmpiricalMuUncertaintySet
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

        return_distribution = self.prior_estimator_.return_distribution_
        n_observations, n_assets = return_distribution.returns.shape
        self.n_eff_ = n_observations if self.n_eff is None else self.n_eff

        k = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=n_assets))

        sigma = return_distribution.covariance / self.n_eff_
        if self.diagonal:
            sigma = np.diag(np.diag(sigma))

        self.uncertainty_set_ = UncertaintySet(k=k, sigma=sigma)
        return self


class EmpiricalCovarianceUncertaintySet(BaseCovarianceUncertaintySet):
    r"""Empirical Covariance Uncertainty set.

    Compute the covariance ellipsoidal uncertainty set [1]_:

    .. math:: U_{\Sigma}=\left\{\Sigma\,|\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)S^{-1}\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)^{T}\leq k^{2}\,,\,\Sigma\succeq 0\right\}

    We consider the Wishart distribution for the covariance matrix:

    .. math:: \hat{\Sigma}\sim W(\frac{1}{T-1}\Sigma, T-1)

    The size of the ellipsoid  :math:`\kappa` (confidence region), is computed using:

    .. math:: \kappa^2 = \chi^2_{n\_assets^2} (\beta)

    with :math:`\chi^2_{n\_assets^2}(\beta)` the inverse cumulative distribution
    function of the chi-squared distribution with `n_assets^2` degrees of freedom at
    the :math:`\beta` confidence level.

    The shape of the ellipsoid :math:`S` is based on a closed form solution of the
    covariance matrix of the Wishart distributed random variable by using the vector
    notation :math:`vec(x)`:

    .. math:: \text{Cov}[\text{vec}(\hat{\Sigma})]=\frac{1}{T-1}(I_{n^2} + K_{nn})(\Sigma \otimes \Sigma)

    with :math:`K_{nn}` denoting a commutation matrix and :math:`\otimes` representing
    the Kronecker product.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        The :ref:`prior estimator <prior>` used to estimate the assets covariance
        matrix. The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    confidence_level : float, default=0.95
        Confidence level :math:`\beta` of the inverse cumulative distribution function
        of the chi-squared distribution. The default value is `0.95`.

    diagonal : bool, default=True
        If this is set to True, the non-diagonal elements of the covariance matrix are
        set to zero.

    n_eff : float, optional
        Effective number of observations used for the covariance estimator. If `None`,
        the number of observations in `X` is used. This is useful when the covariance
        matrix is estimated using a different window length or a weighted estimator
        (e.g. EWMA), in which case `n_eff` should be set to the corresponding effective
        sample size.

    Attributes
    ----------
    uncertainty_set_ : UncertaintySet
        Covariance Uncertainty set :class:`~skfolio.uncertainty_set.UncertaintySet`.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    n_eff_ : float
        Effective number of observations actually used to build the uncertainty set.

    References
    ----------
    .. [1]  "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [2] "Portfolio Optimization: Theory and Application", Chapter 14,
        Daniel P. Palomar (2025)
    """

    n_eff_: float

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_eff: float | None = None,
    ):
        super().__init__(prior_estimator=prior_estimator)
        self.confidence_level = confidence_level
        self.diagonal = diagonal
        self.n_eff = n_eff

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
    ) -> "EmpiricalCovarianceUncertaintySet":
        """Fit the Empirical Covariance Uncertainty set estimator.

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
        self : EmpiricalCovarianceUncertaintySet
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

        return_distribution = self.prior_estimator_.return_distribution_
        n_observations, n_assets = return_distribution.returns.shape
        self.n_eff_ = n_observations if self.n_eff is None else self.n_eff

        k = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=n_assets**2))

        sigma = return_distribution.covariance / self.n_eff_
        if self.diagonal:
            sigma = np.diag(np.diag(sigma))

        sigma = np.diag(
            np.diag(
                self.n_eff_
                * (np.identity(n_assets**2) + commutation_matrix(sigma))
                @ np.kron(sigma, sigma)
            )
        )

        self.uncertainty_set_ = UncertaintySet(k=k, sigma=sigma)
        return self
