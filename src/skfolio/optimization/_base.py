"""Base Optimization estimator."""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
from sklearn.utils.validation import check_is_fitted

from skfolio.measures import RatioMeasure
from skfolio.population import Population
from skfolio.portfolio import Portfolio

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.


class BaseOptimization(skb.BaseEstimator, ABC):
    """Base class for all portfolio optimizations in skfolio.

    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets)
        Weights of the assets.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their `__init__` as explicit keyword
    arguments (no `*args` or `**kwargs`).
    """

    weights_: np.ndarray

    @abstractmethod
    def __init__(self, portfolio_params: dict | None = None):
        self.portfolio_params = portfolio_params

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        pass

    def predict(self, X: npt.ArrayLike) -> Portfolio | Population:
        """Predict the `Portfolio` or `Population` of `Portfolio` on `X` based on the
        fitted weights.

        Optimization estimators can return a 1D or a 2D array of `weights`.
        For a 1D array, the prediction returns a `Portfolio`.
        For a 2D array, the prediction returns a `Population` of `Portfolio`.

        If `name` is not provided in the portfolio arguments, we use the first
        500 characters of the estimator name.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        Returns
        -------
        prediction : Portfolio | Population
            `Portfolio` or `Population` of `Portfolio` estimated on `X` based on the
            fitted `weights`.
        """
        check_is_fitted(self, "weights_")

        if self.portfolio_params is None:
            ptf_kwargs = {}
        else:
            ptf_kwargs = self.portfolio_params.copy()

        # Set the default portfolio parameters equal to the optimization parameters
        for param in [
            "transaction_costs",
            "management_fees",
            "previous_weights",
            "risk_free_rate",
        ]:
            if param not in ptf_kwargs and hasattr(self, param):
                ptf_kwargs[param] = getattr(self, param)

        # If 'name' is not provided in the portfolio arguments, we use the first
        # 500 characters of the optimization estimator's name
        name = ptf_kwargs.pop("name", type(self).__name__)

        # Optimization estimators can return a 1D or a 2D array of weights.
        # For a 1D array we return a portfolio.
        # For a 2D array we return a population of portfolios.
        if self.weights_.ndim == 2:
            n_portfolios = self.weights_.shape[0]
            return Population(
                [
                    Portfolio(
                        X=X,
                        weights=self.weights_[i],
                        name=f"ptf{i} - {name}",
                        **ptf_kwargs,
                    )
                    for i in range(n_portfolios)
                ]
            )
        return Portfolio(X=X, weights=self.weights_, name=name, **ptf_kwargs)

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> float:
        """Prediction score.
        If the prediction is a single `Portfolio`, the score is the Sharpe Ratio.
        If the prediction is a `Population` of `Portfolio`, the score is the mean of all
        the portfolios Sharpe Ratios in the population.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            The Sharpe Ratio of the portfolio if the prediction is a single `Portfolio`
            or the mean of all the portfolios Sharpe Ratios if the prediction is a
            `Population` of `Portfolio`.
        """
        result = self.predict(X)
        if isinstance(result, Population):
            return result.measures_mean(RatioMeasure.SHARPE_RATIO)
        return result.sharpe_ratio

    def fit_predict(self, X):
        """Perform `fit` on `X` and returns the predicted `Portfolio` or
        `Population` of `Portfolio` on `X` based on the fitted `weights`.
        For factor models, use `fit(X, y)` then `predict(X)` separately.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        Returns
        -------
        prediction : Portfolio | Population
            `Portfolio` or `Population` of `Portfolio` estimated on `X` based on the
            fitted `weights`.
        """
        return self.fit(X).predict(X)
