"""Time-series factor model estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import sklearn.base as skb
import sklearn.linear_model as skl
import sklearn.multioutput as skmo
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

import skfolio.measures as sm
from skfolio.prior._base import BasePrior
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.prior._model import FactorModel, ReturnDistribution
from skfolio.typing import ArrayLike, FloatArray, StrArray
from skfolio.utils.stats import cov_nearest
from skfolio.utils.tools import check_estimator, get_feature_names


class BaseLoadingMatrix(skb.BaseEstimator, ABC):
    """Base class for all Loading Matrix estimators.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    loading_matrix_: FloatArray
    intercepts_: FloatArray

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, **fit_params):
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
        The asset-by-factor loading (exposure) matrix.

    intercepts_: ndarray of shape (n_assets,)
        The intercepts.

    multi_output_regressor_: MultiOutputRegressor
        Fitted `sklearn.multioutput.MultiOutputRegressor`
    """

    multi_output_regressor_: skmo.MultiOutputRegressor

    def __init__(
        self,
        linear_regressor: skb.BaseEstimator | None = None,
        n_jobs: int | None = None,
    ):
        self.linear_regressor = linear_regressor
        self.n_jobs = n_jobs

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            linear_regressor=self.linear_regressor,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: ArrayLike, y: ArrayLike, **fit_params):
        """Fit the Loading Matrix Regression Estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors)
            Price returns of the factors.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : LoadingMatrixRegression
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        _linear_regressor = check_estimator(
            self.linear_regressor,
            default=skl.LassoCV(fit_intercept=False),
            check_type=skb.BaseEstimator,
        )

        self.multi_output_regressor_ = skmo.MultiOutputRegressor(
            _linear_regressor, n_jobs=self.n_jobs
        )
        self.multi_output_regressor_.fit(
            X=y, y=X, **routed_params.factor_prior_estimator.fit
        )
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


class TimeSeriesFactorModel(BasePrior):
    r"""Time-series factor model estimator.

    The purpose of factor models is to impose a structure on financial
    variables and their covariance matrix by explaining them through a small
    number of common factors. This reduces the number of free parameters in
    the estimation problem, making portfolio optimization more robust against
    noise. Factor models also provide a decomposition of risk into systematic
    and idiosyncratic components.

    This estimator implements a time-series regression approach: for each
    asset :math:`i`, the return is regressed on a common set of factor return
    series:

    .. math::

        r_i(t) = \alpha_i + B_i \, f(t) + \epsilon_i(t)

    where :math:`B_i` is the factor loadings (exposures), :math:`f(t)` is the
    vector of factor returns, :math:`\alpha_i` is the intercept, and
    :math:`\epsilon_i(t)` is the idiosyncratic return (residual).

    The expected return vector is:

    .. math::

        \mu = B \, \mathbb{E}[f] + \alpha

    and the covariance matrix is:

    .. math::

        \Sigma = B \, F \, B^\top + D

    where :math:`F` is the factor covariance matrix and :math:`D` is the
    diagonal matrix of idiosyncratic variances.

    .. note::

        This formulation assumes that the factors are tradable assets or portfolios
        (e.g. long-short equity factors or ETF returns), so that the
        factor sample mean is a valid estimate of the factor risk premium.
        When factors are non-tradable variables (e.g. macroeconomic series),
        sometimes called a *macroeconomic factor model* in the literature,
        the sample mean no longer equals the risk premium and a two-pass
        procedure such as Fama-MacBeth (1973) is required to estimate the
        cross-sectional price of risk :math:`\lambda`. That procedure also
        requires a large estimation universe in order to reliably identify the
        factor risk premia.

    Parameters
    ----------
    loading_matrix_estimator : LoadingMatrixEstimator, optional
        Estimator of the loading matrix (betas) of the factors.
        The default (`None`) is to use :class:`LoadingMatrixRegression`
        which fits the factors using `LassoCV` on each asset separately.

    factor_prior_estimator : BasePrior, optional
        Estimator of the factor return distribution. It is used to estimate
        the :class:`~skfolio.prior.ReturnDistribution` containing the factor
        expected returns and covariance matrix.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    factor_families : array-like of shape (n_factors,), optional
        Family label for each factor. When provided, the labels are stored in the
        :class:`~skfolio.prior.FactorModel` and can be used by downstream diagnostics,
        plots and optimization constraints referencing factor families. The default
        (`None`) means that no family labels are attached to the factors.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm is used to find
        the nearest positive semi-definite covariance matrix. It is more
        accurate but slower than the default clipping method. For more
        information see :func:`~skfolio.utils.stats.cov_nearest`.

    max_iteration : int, default=100
        Only used when `higham` is set to True. Maximum number of iterations
        of the Higham (2002) algorithm.

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` containing the
        asset distribution and moments estimation based on the factor model.

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
    n_features_in_: int
    feature_names_in_: StrArray

    def __init__(
        self,
        loading_matrix_estimator: BaseLoadingMatrix | None = None,
        factor_prior_estimator: BasePrior | None = None,
        factor_families: ArrayLike | None = None,
        higham: bool = False,
        max_iteration: int = 100,
    ):
        self.loading_matrix_estimator = loading_matrix_estimator
        self.factor_prior_estimator = factor_prior_estimator
        self.factor_families = factor_families
        self.higham = higham
        self.max_iteration = max_iteration

    def get_metadata_routing(self):
        # route to factor_prior_estimator.fit
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                factor_prior_estimator=self.factor_prior_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            # route to loading_matrix_estimator.fit
            .add(
                loading_matrix_estimator=self.loading_matrix_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    def fit(
        self,
        X: ArrayLike,
        y: Any,
        factors: ArrayLike | None = None,
        **fit_params,
    ) -> TimeSeriesFactorModel:
        """Fit the Time-series factor model estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors)
            Factors' returns.

        factors : array-like of shape (n_observations, n_factors), optional
            Factors' returns. If provided, it will override `y`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : TimeSeriesFactorModel
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

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

        factor_returns = y
        if factors is not None:
            factor_returns = factors

        observations = X.index
        factor_names = get_feature_names(factor_returns)

        # Fitting prior estimator
        self.factor_prior_estimator_.fit(
            X=factor_returns, **routed_params.factor_prior_estimator.fit
        )
        factor_return_dist = self.factor_prior_estimator_.return_distribution_

        # Fitting loading matrix estimator
        self.loading_matrix_estimator_.fit(
            X, factor_returns, **routed_params.loading_matrix_estimator.fit
        )
        loading_matrix = self.loading_matrix_estimator_.loading_matrix_
        intercepts = self.loading_matrix_estimator_.intercepts_

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X, factor_returns = skv.validate_data(
            self, X, factor_returns, multi_output=True
        )
        n_assets = X.shape[1]
        n_factors = factor_returns.shape[1]
        factor_families = None

        if self.factor_families is not None:
            factor_families = np.asarray(self.factor_families, dtype=object)
            if factor_families.ndim != 1:
                raise ValueError(
                    "`factor_families` must be a 1D array of shape "
                    f"({n_factors},), got {factor_families.ndim}D array."
                )
            if factor_families.shape[0] != n_factors:
                raise ValueError(
                    f"`factor_families` must have length {n_factors}, got "
                    f"{factor_families.shape[0]}."
                )

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

        mu = loading_matrix @ factor_return_dist.mu + intercepts
        covariance = loading_matrix @ factor_return_dist.covariance @ loading_matrix.T
        returns = factor_return_dist.returns @ loading_matrix.T + intercepts

        factor_returns_pred = factor_returns @ loading_matrix.T + intercepts
        idio_returns = X - factor_returns_pred
        idio_var = sm.variance(idio_returns)
        covariance[np.diag_indices_from(covariance)] += idio_var

        covariance = cov_nearest(
            covariance, higham=self.higham, higham_max_iteration=self.max_iteration
        )

        self.return_distribution_ = ReturnDistribution(
            mu=mu,
            covariance=covariance,
            returns=returns,
            sample_weight=factor_return_dist.sample_weight,
            factor_model=FactorModel(
                observations=observations,
                asset_names=self.feature_names_in_,
                factor_names=factor_names,
                factor_families=factor_families,
                loading_matrix=loading_matrix,
                exposures=None,
                factor_covariance=factor_return_dist.covariance,
                factor_mu=factor_return_dist.mu,
                factor_returns=factor_returns,
                idio_covariance=idio_var,
                idio_variances=None,
                idio_mu=None,
                idio_returns=idio_returns,
            ),
        )
        return self
