"""Naive estimators."""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.optimization._base import BaseOptimization
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.stats import rand_weights_dirichlet
from skfolio.utils.tools import check_estimator


class InverseVolatility(BaseOptimization):
    """Inverse Volatility estimator.

    Each asset weight is computed using the inverse of its volatility and rescaled to
    have a sum of weights equal to one. The assets volatilities are derived from the
    prior estimator's covariance matrix.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.ReturnDistribution`
        containing the estimation of assets expected returns, covariance matrix,
        returns and Cholesky decomposition of the covariance.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets)
        Weights of the assets.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.
    """

    prior_estimator_: BasePrior

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        portfolio_params: dict | None = None,
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.prior_estimator = prior_estimator

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
    ) -> "InverseVolatility":
        """Fit the Inverse Volatility estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : InverseVolatility
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # fitting prior estimator
        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
        covariance = self.prior_estimator_.return_distribution_.covariance
        w = 1 / np.sqrt(np.diag(covariance))
        self.weights_ = w / sum(w)
        return self


class EqualWeighted(BaseOptimization):
    """Equally Weighted estimator.

    Each asset weight is equal to `1/n_assets`.

    Parameters
    ----------
    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets)
        Weights of the assets.
    """

    def __init__(self, portfolio_params: dict | None = None):
        super().__init__(portfolio_params=portfolio_params)

    def fit(self, X: npt.ArrayLike, y=None) -> "EqualWeighted":
        """Fit the Equal Weighted estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EqualWeighted
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        n_assets = X.shape[1]
        self.weights_ = np.ones(n_assets) / n_assets
        return self


class Random(BaseOptimization):
    """Random weight estimator.

    The assets weight are drawn from a Dirichlet distribution and sum to one.

    Parameters
    ----------
    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets)
        Weights of the assets.
    """

    def __init__(self, portfolio_params: dict | None = None):
        super().__init__(portfolio_params=portfolio_params)

    def fit(self, X: npt.ArrayLike, y=None):
        """Fit the Random Weighted estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EqualWeighted
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        n_assets = X.shape[1]
        self.weights_ = rand_weights_dirichlet(n=n_assets)
        return self
