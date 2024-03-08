"""Factor Model estimator"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.linear_model as skl
import sklearn.multioutput as skm

from skfolio.prior._base import BasePrior, PriorModel
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.utils.stats import cov_nearest
from skfolio.utils.tools import check_estimator


class BaseLoadingMatrix(skb.BaseEstimator, ABC):
    """Base class for all Loading Matrix estimators.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    loading_matrix_: np.ndarray
    intercepts_: np.ndarray

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike):
        pass


class LoadingMatrixRegression(BaseLoadingMatrix):
    """Loading Matrix Regression estimator.

    Estimate the loading matrix by fitting one linear regressor per asset.

    Parameters
    ----------
    linear_regressor : BaseEstimator, optional
       Linear regressor used to fit the factors on each asset separately.
       The default (`None`) is to use `LassoCV(fit_intercept=False)`.

    n_jobs : int, optional
        The number of jobs to run in parallel.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        The value `-1` means using all processors.
        The default (`None`) means 1 unless in a `joblib.parallel_backend` context.

    Attributes
    ----------
    loading_matrix_ : ndarray of shape (n_assets, n_factors)
        The loading matrix.

    intercepts_: ndarray of shape (n_assets,)
        The intercepts.

    multi_output_regressor_: MultiOutputRegressor
        Fitted `sklearn.multioutput.MultiOutputRegressor`
    """

    multi_output_regressor_: skm.MultiOutputRegressor

    def __init__(
        self,
        linear_regressor: skb.BaseEstimator | None = None,
        n_jobs: int | None = None,
    ):
        self.linear_regressor = linear_regressor
        self.n_jobs = n_jobs

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike):
        """Fit the Loading Matrix Regression Estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors)
            Price returns of the factors.

        Returns
        -------
        self : LoadingMatrixRegression
            Fitted estimator.
        """
        _linear_regressor = check_estimator(
            self.linear_regressor,
            default=skl.LassoCV(fit_intercept=False),
            check_type=skb.BaseEstimator,
        )

        self.multi_output_regressor_ = skm.MultiOutputRegressor(
            _linear_regressor, n_jobs=self.n_jobs
        )
        self.multi_output_regressor_.fit(X=y, y=X)
        # noinspection PyUnresolvedReferences
        n_assets = X.shape[1]
        self.loading_matrix_ = np.array(
            [self.multi_output_regressor_.estimators_[i].coef_ for i in range(n_assets)]
        )
        self.intercepts_ = np.array(
            [
                self.multi_output_regressor_.estimators_[i].intercept_
                for i in range(n_assets)
            ]
        )


class FactorModel(BasePrior):
    """Factor Model estimator.

    The purpose of Factor Models is to impose a structure on financial variables and
    their covariance matrix by explaining them through a small number of common factors.
    This can help overcome estimation error by reducing the number of parameters,
    i.e. the dimensionality of the estimation problem, making portfolio optimization
    more robust against noise in the data. Factor Models also provide a decomposition of
    financial risk into systematic and security-specific components.

    Parameters
    ----------
    loading_matrix_estimator : LoadingMatrixEstimator, optional
        Estimator of the loading matrix (betas) of the factors.
        The default (`None`) is to use :class:`LoadingMatrixRegression` which fit the
        factors using `LassoCV` on each asset separately.

    factor_prior_estimator : BasePrior, optional
        The factors :ref:`prior estimator <prior>`.
        It is used to estimate the :class:`~skfolio.prior.PriorModel` containing the
        factors expected returns and covariance matrix.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    residual_variance : bool, default=True
        If this is set to True, the diagonal term of the residuals covariance
        (residuals variance) is added to the factor model covariance.

    higham : bool, default=False
        If this is set to True, we use the Higham & Nick (2002) algorithm to find the
        nearest covariance matrix that is positive semi-definite. It is more accurate
        but slower that the default clipping method. For more information
        see :func:`~skfolio.utils.stats.cov_nearest`.

    max_iteration : int, default=100
        Only used when `higham` is set to True. Maximum number of iterations of the
        Higham & Nick (2002) algorithm.

    Attributes
    ----------
    prior_model_ : PriorModel
        The :class:`~skfolio.prior.PriorModel`.

    factor_prior_estimator_ : BasePrior
        Fitted `factor_prior_estimator`.

    loading_matrix_estimator_ : BaseLoadingMatrix
        Fitted `loading_matrix_estimator`.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    factor_prior_estimator_: BasePrior
    loading_matrix_estimator_: BaseLoadingMatrix

    def __init__(
        self,
        loading_matrix_estimator: BaseLoadingMatrix | None = None,
        factor_prior_estimator: BasePrior | None = None,
        residual_variance: bool = True,
        higham: bool = False,
        max_iteration: int = 100,
    ):
        self.loading_matrix_estimator = loading_matrix_estimator
        self.factor_prior_estimator = factor_prior_estimator
        self.residual_variance = residual_variance
        self.higham = higham
        self.max_iteration = max_iteration

    # noinspection PyMethodOverriding, PyPep8Naming
    def fit(self, X: npt.ArrayLike, y: Any):
        """Fit the Factor Model estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors)
            Factors' returns.

        Returns
        -------
        self : FactorModel
            Fitted estimator.
        """
        self.factor_prior_estimator_ = check_estimator(
            self.factor_prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        self.loading_matrix_estimator_ = check_estimator(
            self.loading_matrix_estimator,
            default=LoadingMatrixRegression(),
            check_type=BaseLoadingMatrix,
        )

        # Fitting prior estimator
        self.factor_prior_estimator_.fit(y)
        factor_mu = self.factor_prior_estimator_.prior_model_.mu
        factor_covariance = self.factor_prior_estimator_.prior_model_.covariance

        # Fitting loading matrix estimator
        self.loading_matrix_estimator_.fit(X, y)
        loading_matrix = self.loading_matrix_estimator_.loading_matrix_
        intercepts = self.loading_matrix_estimator_.intercepts_

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X, y = self._validate_data(X, y, multi_output=True)
        n_assets = X.shape[1]
        n_factors = y.shape[1]

        if loading_matrix.shape != (n_assets, n_factors):
            raise ValueError(
                "`loading_matrix_estimator.loading_matrix_` must ba a 2D array of"
                f" shape {(n_assets, n_factors)}, got"
                f" {loading_matrix.shape} instead."
            )

        if intercepts.shape != (n_assets,):
            raise ValueError(
                "`loading_matrix_estimator.intercepts_` must ba a 1D array of "
                f"shape {(n_assets,)}, got {intercepts.shape} instead."
            )

        mu = loading_matrix @ factor_mu + intercepts
        covariance = loading_matrix @ factor_covariance @ loading_matrix.T
        returns = y @ loading_matrix.T + intercepts
        cholesky = loading_matrix @ np.linalg.cholesky(factor_covariance)

        if self.residual_variance:
            err = X - returns
            err_cov = np.diag(np.var(err, ddof=1, axis=0))
            covariance += err_cov
            cholesky = np.hstack((cholesky, np.sqrt(err_cov)))

        covariance = cov_nearest(
            covariance, higham=self.higham, higham_max_iteration=self.max_iteration
        )

        self.prior_model_ = PriorModel(
            mu=mu, covariance=covariance, returns=returns, cholesky=cholesky
        )
        return self
