"""Black & Litterman Prior Model estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# PyPortfolioOpt, Copyright (c) 2018 Robert Andrew Martin, Licensed under MIT Licence.

import numpy as np
import numpy.typing as npt

from skfolio.moments import EquilibriumMu
from skfolio.prior._base import BasePrior, PriorModel
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import check_estimator, input_to_array


class BlackLitterman(BasePrior):
    """Black & Litterman Prior Model estimator.

    The Black & Litterman model [1]_ takes a Bayesian approach by using a prior estimate
    of the assets expected returns and covariance matrix, which are updated using the
    analyst views to get a posterior estimate.

    Parameters
    ----------
    views : array-like of floats of shape (n_views,)
        The analyst views about the assets expected returns.
        The views must match the following patterns:

            * Absolute view: "asset_i = a"
            * Relative view: "asset_i - asset_j = b"

        With "asset_i" and "asset_j" the assets names and "a" and "b" the analyst views
        about the assets expected returns expressed in the same frequency as the
        returns `X`.

        Examples:

            * "SPX = 0.00015" --> the SPX will have a daily expected return of 0.015%
            * "SX5E - TLT = 0.00039" --> the SX5E will outperform the TLT by a daily expected return of 0.039%
            * "SX5E - SPX = -0.0002" --> the SX5E will underperform the SPX by a daily expected return of 0.02%
            * "Equity = 0.00010" --> the sum of Equity assets will have a daily expected return of 0.01%
            * "Europe - US = 0.0004" --> the sum of European assets will outperform the sum of US assets by a daily expected return of 0.04%

    groups : dict[str, list[str]] or array-like of strings of shape (n_groups, n_assets), optional
        The assets groups to be referenced in `views`.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset groups) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.

        Examples:

            * groups = {"SX5E": ["Equity", "Europe"], "SPX": ["Equity", "US"], "TLT": ["Bond", "US"]}
            * groups = [["Equity", "Equity", "Bond"], ["Europe", "US", "US"]]

    prior_estimator : BasePrior, optional
        The assets' :ref:`prior model estimator <prior>`. It is used to estimate
        the :class:`~skfolio.prior.PriorModel` containing the estimation of the assets
        expected returns, covariance matrix, returns and Cholesky decomposition.
        The default (`None`) is to use `EmpiricalPrior(mu_estimator=EquilibriumMu())`.

    tau : float, default=0.05
        Tau controls the degree of uncertainty given to the analyst views. A low value
        means high uncertainty and will put less weight on the analyst views compared to
        the prior returns. The default value is `0.05`.
        Other common values used in the literature are `1.0` or the inverse of the
        number of observations.

    view_confidences : array-like of floats of shape (n_views,), optional
        Instead of using a diagonal uncertainty matrix (Omega) proportional to the prior
        covariance matrix, you can provide the vector of view confidences (between 0
        and 1) as describe by the Idzorek's method [2]_.

    risk_free_rate : float, default=0.0
        The risk-free rate.

    Attributes
    ----------
    prior_model_ : PriorModel
        The :class:`~skfolio.prior.PriorModel`.

    groups_ : ndarray of shape(n_groups, n_assets)
        Assets names and groups converted to an 2D array.

    views_ : ndarray of shape (n_views,)
        The analyst views converted to a ndarray of floats.

    picking_matrix_ : ndarray of shape (n_views, n_assets)
        Picking matrix computed from the views and assets names/groups.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    References
    ----------
    .. [1]  "Combining investor views with market equilibrium",
        The Journal of Fixed Income,
        Fischer Black and Robert Litterman, 1991.

    .. [2]  "A step-by-step guide to the Black-Litterman model : Incorporating
        user-specified confidence",
        Forecasting Expected Returns in the Financial Markets,
        Idzorek T, 2007.
    """

    groups_: np.ndarray
    views_: np.ndarray
    picking_matrix_: np.ndarray
    prior_estimator_: BasePrior

    def __init__(
        self,
        views: npt.ArrayLike,
        groups: dict[str, list[str]] | npt.ArrayLike | None = None,
        prior_estimator: BasePrior | None = None,
        tau: float = 0.05,
        view_confidences: npt.ArrayLike | None = None,
        risk_free_rate: float = 0,
    ):
        self.views = views
        self.groups = groups
        self.prior_estimator = prior_estimator
        self.tau = tau
        self.view_confidences = view_confidences
        self.risk_free_rate = risk_free_rate

    def fit(self, X: npt.ArrayLike, y=None) -> "BlackLitterman":
        """Fit the Black & Litterman estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BlackLitterman
            Fitted estimator.
        """
        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(mu_estimator=EquilibriumMu()),
            check_type=BasePrior,
        )
        # fitting prior estimator
        self.prior_estimator_.fit(X)

        prior_mu = self.prior_estimator_.prior_model_.mu
        prior_covariance = self.prior_estimator_.prior_model_.covariance
        prior_returns = self.prior_estimator_.prior_model_.returns

        # we validate after all models have been fitted to keep features names
        # information.
        self._validate_data(X)

        n_assets = prior_returns.shape[1]
        views = np.asarray(self.views)
        if views.ndim != 1:
            raise ValueError(f"`views` must be a 1D array, got a {views.ndim}D array.")
        if self.groups is None:
            if not hasattr(self, "feature_names_in_"):
                raise ValueError(
                    "You must provide either `groups`"
                    " or `X` as a DataFrame with asset names in columns"
                )
            self.groups_ = np.asarray([self.feature_names_in_])
        else:
            self.groups_ = input_to_array(
                items=self.groups,
                n_assets=n_assets,
                fill_value="",
                dim=2,
                assets_names=(
                    self.feature_names_in_
                    if hasattr(self, "feature_names_in_")
                    else None
                ),
                name="groups",
            )
        self.picking_matrix_, self.views_ = equations_to_matrix(
            groups=self.groups_,
            equations=views,
            sum_to_one=True,
            raise_if_group_missing=True,
            names=("groups", "views"),
        )

        if self.view_confidences is None:
            omega = np.diag(
                np.diag(
                    self.tau
                    * self.picking_matrix_
                    @ prior_covariance
                    @ self.picking_matrix_.T
                )
            )
        else:
            # Idzorek's method using Jay Walters closed form solution
            view_confidences = np.asarray(self.view_confidences)
            if np.any(view_confidences < 0) or np.any(view_confidences > 1):
                raise ValueError(
                    "all values of view_confidences must be between 0 and 1"
                )
            view_confidences[view_confidences == 0] = 1e-16
            alphas = 1 / view_confidences - 1
            omega = np.diag(
                np.diag(
                    self.tau
                    * alphas[:, np.newaxis]
                    * self.picking_matrix_
                    @ prior_covariance
                    @ self.picking_matrix_.T
                )
            )

        # solving linear system instead of matrix inversion
        _v = self.tau * prior_covariance @ self.picking_matrix_.T
        _a = self.picking_matrix_ @ _v + omega
        _b = self.views_ - self.picking_matrix_ @ prior_mu
        posterior_mu = prior_mu + _v @ np.linalg.solve(_a, _b) + self.risk_free_rate
        posterior_covariance = (
            prior_covariance
            + self.tau * prior_covariance
            - _v @ np.linalg.solve(_a, _v.T)
        )
        self.prior_model_ = PriorModel(
            mu=posterior_mu, covariance=posterior_covariance, returns=prior_returns
        )
        return self
