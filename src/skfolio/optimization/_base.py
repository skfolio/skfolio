"""Base classes and utilities for portfolio optimization estimators.

This module defines the abstract `BaseOptimization` estimator that all
optimization algorithms in skfolio should inherit from.
"""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import wraps
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn as sk
import sklearn.base as skb
from sklearn.utils.validation import check_is_fitted

import skfolio.typing as skt
from skfolio._constants import _ParamKey
from skfolio.measures import RatioMeasure
from skfolio.population import Population
from skfolio.portfolio import FailedPortfolio, Portfolio
from skfolio.prior import ReturnDistribution
from skfolio.utils.tools import input_to_array


class BaseOptimization(skb.BaseEstimator, ABC):
    """Base class for all portfolio optimizations in skfolio.

    Parameters
    ----------
    portfolio_params : dict, optional
        Portfolio parameters forwarded to the resulting `Portfolio` in `predict`.
        If not provided and if available on the estimator, the following
        attributes are propagated to the portfolio by default: `name`,
        `transaction_costs`, `management_fees`, `previous_weights` and `risk_free_rate`.

    fallback : BaseOptimization | "previous_weights" | list[BaseOptimization | "previous_weights"], optional
        Fallback estimator or a list of estimators to try, in order, when the primary
        optimization raises during `fit`. Alternatively, use `"previous_weights"`
        (alone or in a list) to fall back to the estimator's `previous_weights`.
        When a fallback succeeds, its fitted `weights_` are copied back to the primary
        estimator so that `fit` still returns the original instance. For traceability,
        `fallback_` stores the successful estimator (or the string `"previous_weights"`)
         and `fallback_chain_` stores each attempt with the associated outcome.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets,), optional
        Previous asset weights. Some estimators use this to compute costs or turnover.
        Additionally, when `fallback="previous_weights"`, failures will fall back to
        these weights if provided.

    raise_on_failure : bool, default=True
        Controls error handling when fitting fails.
        If True, any failure during `fit` is raised immediately, no `weights_` are
        set and subsequent calls to `predict` will raise a `NotFittedError`.
        If False, errors are not raised; instead, a warning is emitted, `weights_`
        is set to `None` and subsequent calls to `predict` will return a
        `FailedPortfolio`. When fallbacks are specified, this behavior applies only
        after all fallbacks have been exhausted.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets)
        Weights of the assets.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    fallback_ : BaseOptimization | "previous_weights" | None
        The fallback estimator instance, or the string `"previous_weights"`, that
        produced the final result. `None` if no fallback was used.

    fallback_chain_ : list[tuple[str, str]] | None
        Sequence describing the optimization fallback attempts. Each element is a
        pair `(estimator_repr, outcome)` where `estimator_repr` is the string
        representation of the primary estimator or a fallback (e.g. `"EqualWeighted()"`,
        `"previous_weights"`), and `outcome` is `"success"` if that step produced
        a valid solution, otherwise the stringified error message. For successful
        fits without any fallback, this is `None`.

    error_ : str | list[str] | None
        Captured error message(s) when `fit` fails. For multi-portfolio outputs
        (`weights_` is 2D), this is a list aligned with portfolios.

    Notes
    -----
    All estimators should specify all parameters as explicit keyword arguments in
    `__init__` (no `*args` or `**kwargs`), following scikit-learn conventions.
    """

    weights_: np.ndarray
    n_features_in_: int
    feature_names_in_: np.ndarray
    fallback_: BaseOptimization | Literal["previous_weights"] | None
    fallback_chain_: list[tuple[str, str]] | None
    error_: str | list[str] | None

    @abstractmethod
    def __init__(
        self,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        previous_weights: skt.MultiInput | None = None,
        raise_on_failure: bool = True,
    ):
        self.portfolio_params = portfolio_params
        self.fallback = fallback
        self.previous_weights = previous_weights
        self.raise_on_failure = raise_on_failure

    # Automatically wrap all subclasses' fit to add fallback behavior
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        original_fit = cls.__dict__.get("fit")
        if original_fit is None or getattr(original_fit, "_fallback_wrapped", False):
            return

        @wraps(original_fit)
        def _wrapped_fit(
            self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
        ):
            self.fallback_ = None
            self.fallback_chain_ = None
            self.error_ = None

            try:
                original_fit(self, X, y, **fit_params)
            except Exception as primary_error:
                try:
                    self._run_fallback_chain(
                        X=X, y=y, primary_error=primary_error, **fit_params
                    )
                except Exception as last_error:
                    self.error_ = str(last_error)
                    if self.raise_on_failure:
                        raise
                    warnings.warn(
                        (
                            f"{self.__class__.__name__}.fit failed: {last_error}. "
                            "Because raise_on_failure=False, weights_ is set to None. "
                            "Inspect 'error_' and 'fallback_chain_' for details."
                        ),
                        stacklevel=2,
                    )
                    self.weights_ = None
            return self

        _wrapped_fit._fallback_wrapped = True
        cls.fit = _wrapped_fit

    def _run_fallback_chain(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None,
        primary_error: Exception,
        **fit_params,
    ) -> None:
        """Execute the configured fallback chain after a primary `fit` failure.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Training data passed to `fit`.

        y : array-like or None
            Optional target data.

        primary_error : Exception
            The exception raised by the primary estimator.

        **fit_params : dict
            Additional keyword arguments forwarded to each fallback's `fit`.

        Raises
        ------
        Exception
            Re-raises the last encountered error if all fallbacks fail.
        """
        fallback = self.fallback

        if fallback is None:
            raise primary_error

        # Log the primary error in fallback_chain_ only when fallbacks are provided
        self.fallback_chain_ = [(str(self), str(primary_error))]

        n_assets = X.shape[1]

        if not isinstance(fallback, list | tuple):
            fallback = [fallback]

        if len(fallback) == 0:
            raise primary_error

        last_error: Exception = primary_error
        for fb in fallback:
            try:
                fb = _validate_fallback(fb)
                if fb == _ParamKey.PREVIOUS_WEIGHTS.value:
                    self._fallback_to_previous_weights_or_raise(n_assets=n_assets)
                    return

                fb_est = sk.clone(fb)

                # previous_weights are propagated to the fallbacks
                if self.previous_weights is not None:
                    if fb_est.previous_weights is not None:
                        warnings.warn(
                            (
                                "previous_weights are automatically propagated to "
                                "fallback estimators. To silence this warning, leave "
                                "the fallback's previous_weights as None."
                            ),
                            stacklevel=2,
                        )
                    fb_est.set_params(previous_weights=self.previous_weights)

                fb_est.fit(X, y, **fit_params)

                # Success: copy learned artifacts back to self
                for name in ("weights_", "n_features_in_", "feature_names_in_"):
                    setattr(self, name, getattr(fb_est, name))

                self.fallback_ = fb_est
                self.fallback_chain_.append((str(fb_est), "success"))
                return
            except Exception as err:  # try next fallback
                last_error = err
                self.fallback_chain_.append((str(fb), str(err)))
                continue

        # All fallbacks failed
        if last_error is not None:
            # Defer raising to the caller which decides based on raise_on_failure
            raise last_error
        raise RuntimeError(
            "All fallback estimators failed; inspect 'fallback_chain_' for details."
        )

    def _fallback_to_previous_weights_or_raise(self, n_assets: int) -> None:
        """Fallback to `previous_weights` or raise if unavailable/invalid.

        Parameters
        ----------
        n_assets : int
            Number of assets used to validate the shape of `previous_weights`.

        Raises
        ------
        RuntimeError
            If `previous_weights` is `None` when the fallback is requested.
        """
        try:
            if self.previous_weights is None:
                raise RuntimeError(
                    "Fallback 'previous_weights' requested, but 'previous_weights' is None. "
                    "Provide valid previous weights or remove this fallback."
                )
            self.weights_ = self._clean_previous_weights(n_assets=n_assets)
            self.fallback_ = _ParamKey.PREVIOUS_WEIGHTS.value
            self.fallback_chain_.append((_ParamKey.PREVIOUS_WEIGHTS.value, "success"))

        except Exception as error:
            self.fallback_chain_.append((_ParamKey.PREVIOUS_WEIGHTS.value, str(error)))
            raise

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        pass

    def predict(self, X: npt.ArrayLike | ReturnDistribution) -> Portfolio | Population:
        """Predict the `Portfolio` or a `Population` of portfolios on `X`.

        Optimization estimators can return a 1D or a 2D array of `weights`.
        For a 1D array, the prediction is a single `Portfolio`.
        For a 2D array, the prediction is a `Population` of `Portfolio`.

        If `name` is not provided in the portfolio parameters, the estimator
        class name is used.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets) | ReturnDistribution
            Asset returns or a `ReturnDistribution` carrying returns and optional
            sample weights.

        Returns
        -------
        Portfolio | Population
            The predicted `Portfolio` or `Population` based on the fitted `weights`.
        """
        check_is_fitted(self, "weights_")

        if self.portfolio_params is None:
            ptf_kwargs = {}
        else:
            ptf_kwargs = self.portfolio_params.copy()

        # Set X and sample_weight
        if isinstance(X, ReturnDistribution):
            ptf_kwargs["sample_weight"] = X.sample_weight
            if hasattr(self, "feature_names_in_"):
                ptf_kwargs["X"] = pd.DataFrame(
                    X.returns, columns=self.feature_names_in_
                )
            else:
                ptf_kwargs["X"] = X.returns
        else:
            ptf_kwargs["X"] = X

        # Set the default portfolio parameters equal to the optimization parameters
        for param in [
            _ParamKey.TRANSACTION_COSTS.value,
            _ParamKey.MANAGEMENT_FEES.value,
            _ParamKey.PREVIOUS_WEIGHTS.value,
            _ParamKey.RISK_FREE_RATE.value,
        ]:
            if param not in ptf_kwargs and hasattr(self, param):
                ptf_kwargs[param] = getattr(self, param)

        # If 'name' is not provided in the portfolio arguments, we use the first
        # 500 characters of the optimization estimator's name
        name = ptf_kwargs.pop("name", type(self).__name__)

        # Add fallback chain
        ptf_kwargs["fallback_chain"] = self.fallback_chain_

        # If weights are None and raise_on_failure is False, we return a FailedPortfolio
        if self.weights_ is None:
            return FailedPortfolio(
                name=name, optimization_error=self.error_, **ptf_kwargs
            )

        # Optimization estimators can return a 1D or a 2D array of weights.
        # For a 1D array we return a portfolio.
        if self.weights_.ndim == 1:
            return Portfolio(weights=self.weights_, name=name, **ptf_kwargs)

        # For a 2D array we return a population of portfolios.
        n_portfolios = self.weights_.shape[0]
        population = Population([])
        for i in range(n_portfolios):
            ptf_name = f"ptf{i} - {name}"
            if np.isnan(self.weights_[i]).all():
                error = self.error_[i] if isinstance(self.error_, list) else None
                population.append(
                    FailedPortfolio(
                        name=ptf_name,
                        optimization_error=error,
                        **ptf_kwargs,
                    )
                )
            else:
                population.append(
                    Portfolio(weights=self.weights_[i], name=ptf_name, **ptf_kwargs)
                )
        return population

    def score(
        self, X: npt.ArrayLike | ReturnDistribution, y: npt.ArrayLike = None
    ) -> float:
        """Prediction score using the Sharpe Ratio.
        If the prediction is a single `Portfolio`, the score is its Sharpe Ratio.
        If the prediction is a `Population`, the score is the mean Sharpe Ratio
        across portfolios.

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

        If fitting fails and `raise_on_failure=False`, this returns a
        `FailedPortfolio`.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        Returns
        -------
        Portfolio | Population
            The predicted `Portfolio` or `Population` based on the fitted `weights`.
        """
        return self.fit(X).predict(X)

    @property
    def needs_previous_weights(self) -> bool:
        """Whether `previous_weights` must be propagated between folds/rebalances.

        Used by `cross_val_predict` to decide whether to run sequentially and pass
        the weights from the previous rebalancing to the next. This is `True` when
        transaction costs, a maximum turnover, or a fallback depending on
        `previous_weights` are present.
        """
        if _has_transaction_cost(
            getattr(self, _ParamKey.TRANSACTION_COSTS.value, None)
        ):
            return True

        if getattr(self, "max_turnover", None) is not None:
            return True

        fallback = self.fallback
        if fallback is not None:
            if not isinstance(fallback, list | tuple):
                fallback = [fallback]

            for fb in fallback:
                fb = _validate_fallback(fb)
                if fb == _ParamKey.PREVIOUS_WEIGHTS.value or fb.needs_previous_weights:
                    return True

        return False

    def _clean_input(
        self,
        value: float | dict | npt.ArrayLike | None,
        n_assets: int,
        fill_value: Any,
        name: str,
    ) -> float | np.ndarray:
        """Convert input to a cleaned float or 1D ndarray.

        Parameters
        ----------
        value : float | dict | array-like | None
            Input value to clean.

        n_assets : int
            Number of assets. Used to verify the shape of the converted array.

        fill_value : Any
            When `value` is a dictionary, keys not present in the asset names are
            filled with `fill_value` in the converted array.

        name : str
            Name used for error messages.

        Returns
        -------
        float | ndarray of shape (n_assets,)
            The cleaned scalar or 1D array.
        """
        if value is None:
            return fill_value
        if np.isscalar(value):
            return float(value)
        return input_to_array(
            items=value,
            n_assets=n_assets,
            fill_value=fill_value,
            dim=1,
            assets_names=(
                self.feature_names_in_ if hasattr(self, "feature_names_in_") else None
            ),
            name=name,
        )

    def _clean_previous_weights(self, n_assets: int) -> np.ndarray:
        """Return validated previous weights as a 1D array of length `n_assets`.

        Converts `previous_weights` to a numpy array using `_clean_input`, accepting
        scalars, mappings keyed by asset name, or array-like inputs. Scalars are
        broadcast to all assets. Missing assets in mappings are filled with zeros.

        Parameters
        ----------
        n_assets : int
            Number of assets; used to validate shape and for broadcasting.

        Returns
        -------
        ndarray of shape (n_assets,)
            Cleaned previous weights.
        """
        previous_weights = self._clean_input(
            self.previous_weights,
            n_assets=n_assets,
            fill_value=0,
            name=_ParamKey.PREVIOUS_WEIGHTS.value,
        )
        if np.isscalar(previous_weights):
            previous_weights = np.full(n_assets, float(previous_weights))
        return previous_weights


def _has_transaction_cost(x: Any) -> bool:
    """Return True if any non-zero transaction cost is present in `x`.

    Accepts scalars, arrays, nested mappings, or structures convertible to arrays.
    Zero or empty values are treated as no cost.
    """
    if x is None:
        return False

    if isinstance(x, Mapping):
        # Empty dict -> no costs; otherwise recurse
        return any(_has_transaction_cost(v) for v in x.values())

    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        # If coercion fails, assume non-zero to be conservative
        return True

    if arr.size == 0:
        return False

    return not np.allclose(arr, 0.0, atol=1e-15, rtol=1e-18, equal_nan=False)


def _validate_fallback(
    fallback: Literal["previous_weights"] | BaseOptimization,
) -> Literal["previous_weights"] | BaseOptimization:
    """Validate the fallback specification.

    Parameters
    ----------
    fallback : BaseOptimization | "previous_weights"
        The configured fallback.

    Returns
    -------
    BaseOptimization | "previous_weights"
        The validated fallback, unchanged.

    Raises
    ------
    ValueError
        If `fallback` is a string different from `"previous_weights"`.
    TypeError
        If `fallback` is not a string and not an instance of `BaseOptimization`.
    """
    if isinstance(fallback, str):
        if fallback != _ParamKey.PREVIOUS_WEIGHTS.value:
            raise ValueError(
                f"Unsupported string fallback: {fallback!r}. Only 'previous_weights' is allowed."
            )
        return _ParamKey.PREVIOUS_WEIGHTS.value
    if not isinstance(fallback, BaseOptimization):
        raise TypeError(
            f"Fallback estimators must inherit from BaseOptimization (got {type(fallback).__name__})."
        )
    return fallback
