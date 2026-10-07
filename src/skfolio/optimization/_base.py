"""Base classes and utilities for portfolio optimization estimators.

This module defines the abstract `BaseOptimization` estimator that all
optimization algorithms in skfolio should inherit from.
"""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import numbers
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.base as skb
import sklearn.utils.metadata_routing as skm
from sklearn.utils.validation import check_is_fitted, validate_data

import skfolio.typing as skt
from skfolio._constants import (
    _MANAGEMENT_FEES,
    _PREVIOUS_WEIGHTS,
    _RISK_FREE_RATE,
    _TRANSACTION_COSTS,
)
from skfolio.exceptions import OptimizationError
from skfolio.measures import RatioMeasure
from skfolio.population import Population
from skfolio.portfolio import FailedPortfolio, Portfolio
from skfolio.prior import ReturnDistribution
from skfolio.typing import ArrayLike, FloatArray, StrArray
from skfolio.utils.tools import _filter_supported_params, input_to_array


class BaseOptimization(skb.BaseEstimator, ABC):
    """Base class for all portfolio optimizations in skfolio.

    Parameters
    ----------
    portfolio_params : dict, optional
        Portfolio parameters forwarded to the resulting `Portfolio` in `predict`.
        Unless set in this dictionary, `transaction_costs`, `management_fees`,
        `previous_weights` and `risk_free_rate` are forwarded from the optimizer when
        available, and `name` defaults to the optimizer class name.
        For example, `portfolio_params={"weight_drift": True}` evaluates the predicted
        portfolios with drifted weights instead of the target weights on every
        observation.

    fallback : BaseOptimization | "previous_weights" | list[BaseOptimization | "previous_weights"], optional
        Fallback estimator or a list of estimators to try, in order, when the primary
        optimization raises during `fit`. Alternatively, use `"previous_weights"` (alone
        or in a list) to fall back to the estimator's `previous_weights`. When a
        fallback succeeds, its fitted `weights_` are copied back to the primary
        estimator so that `fit` still returns the original instance. For traceability,
        `fallback_` stores the successful estimator (or the string `"previous_weights"`)
        and `fallback_chain_` stores each attempt with the associated outcome. With
        `partial_fit`, only None or `"previous_weights"` is supported because fallback
        estimators have not accumulated the primary model's online history. See
        :ref:`optimization_fallbacks`.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets,), optional
        Previous asset weights. Some portfolio optimizers use this to compute costs or
        turnover. Additionally, when `fallback="previous_weights"`, failures will fall
        back to these weights if provided.

    raise_on_failure : bool, default=True
        Controls error handling when fitting fails and no fallback succeeds. If True,
        the estimator raises the final error. If False, the estimator emits a warning
        and sets `weights_` to None, so subsequent calls to `predict` return a
        :class:`~skfolio.portfolio.FailedPortfolio`. During `fit`, `raise_on_failure`
        applies to any fitting error, including errors raised by the prior estimator.
        During `partial_fit`, `raise_on_failure` applies only to optimization failures
        after learning completes. Input validation failures and errors from the prior or
        other learning estimators are always raised. See
        :ref:`optimization_failure_handling` for batch recovery and
        :ref:`online_failure_handling` for online continuation and restart rules.

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

    error_ : str | list[str | None] | None
        For a single portfolio, this is the recorded error message, or None after a
        successful allocation or fallback. For multiple portfolios, it is a list with
        one entry per row of `weights_`, containing an error message for each failed
        portfolio and None for each successful portfolio.

    Notes
    -----
    All estimators should specify all parameters as explicit keyword arguments in
    `__init__` (no `*args` or `**kwargs`), following scikit-learn conventions.
    """

    weights_: FloatArray | None
    n_features_in_: int
    feature_names_in_: StrArray
    fallback_: BaseOptimization | Literal["previous_weights"] | None
    fallback_chain_: list[tuple[str, str]] | None
    error_: str | list[str | None] | None
    _fit_in_progress: bool

    def __init__(
        self,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        previous_weights: skt.MultiInput | None = None,
        raise_on_failure: bool = True,
    ) -> None:
        self.portfolio_params = portfolio_params
        self.fallback = fallback
        self.previous_weights = previous_weights
        self.raise_on_failure = raise_on_failure

    # Automatically wrap all subclasses' fit to add fallback behavior
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        original_fit = cls.__dict__.get("fit")
        if original_fit is None or getattr(original_fit, "_fallback_wrapped", False):
            return

        @wraps(original_fit)
        def _wrapped_fit(
            self: BaseOptimization,
            X: ArrayLike,
            y: ArrayLike | None = None,
            **fit_params: Any,
        ) -> BaseOptimization:
            """Run `original_fit` and try the fallback chain if it fails."""
            # Both subclass and parent fit methods are wrapped. Calls to
            # super().fit() skip the parent's fallback handling so the outermost
            # wrapper runs the fallback chain once, using the original inputs.
            if getattr(self, "_fit_in_progress", False):
                original_fit(self, X, y, **fit_params)
                return self

            # Batch fallback must not reuse the investable mask from a previous fit.
            if hasattr(self, "investable_mask_"):
                del self.investable_mask_

            self._fit_in_progress = True
            try:
                with self._handle_optimization_errors(
                    X, y, method="fit", fit_params=fit_params
                ):
                    original_fit(self, X, y, **fit_params)
            finally:
                del self._fit_in_progress
            return self

        _wrapped_fit._fallback_wrapped = True  # ty: ignore[unresolved-attribute]
        cls.fit = _wrapped_fit  # ty: ignore[invalid-assignment]

    @contextmanager
    def _handle_optimization_errors(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        *,
        method: str,
        enabled: bool = True,
        fit_params: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """Handle fitting failures according to `fallback` and `raise_on_failure`.

        During `fit`, this handler catches any `Exception` from fitting, including
        errors raised by the prior estimator.

        During `partial_fit`, update the prior and other estimators before entering this
        handler. Errors from those updates are always raised, even when they are
        `OptimizationError`. The handler then catches only `OptimizationError` from
        computing and assigning portfolio weights.

        If a fallback succeeds, `weights_` contains its allocation. If all attempts fail
        and `raise_on_failure=False`, `weights_` is set to None. Assign primary weights
        within the `with` statement so that this assignment is skipped after a failure
        and cannot overwrite the fallback allocation.

        After a suppressed optimization error or a successful online fallback, the prior
        and other estimators retain their updates. The next `partial_fit` must receive
        only new observations.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Returns passed to the fitting method.

        y : array-like or None
            Optional target data.

        method : str
            Either "fit" or "partial_fit".

        enabled : bool, default=True
            If False, do not reset diagnostics or handle errors here. During `fit`, the
            wrapper around the fitting method handles fallback once using the original
            inputs.

        fit_params : dict, optional
            Parameters routed to fallback estimators during `fit`. Passed as a
            dictionary so metadata names cannot conflict with this handler's parameters.

        Yields
        ------
        None
            Runs the fitting or weight computation operation.
        """
        if not enabled:
            yield
            return

        self.error_ = None
        self.fallback_ = None
        self.fallback_chain_ = None
        error_type = Exception if method == "fit" else OptimizationError
        try:
            yield
        except error_type as error:
            try:
                self._run_fallback_chain(
                    X, y, primary_error=error, **(fit_params or {})
                )
            except Exception as last_error:
                self.error_ = str(last_error)
                if self.raise_on_failure:
                    raise
                message = (
                    f"{self.__class__.__name__}.{method} failed: {last_error}. "
                    "Because raise_on_failure=False, weights_ is set to None. "
                    "Inspect 'error_' and 'fallback_chain_' for details."
                )
                if method == "partial_fit":
                    message += (
                        " The batch was consumed. Supply only new observations "
                        "on the next partial_fit."
                    )
                # Skip contextlib.__exit__ and point to the batch caller or the
                # with statement inside the internal _fit for online updates.
                warnings.warn(message, stacklevel=4 if method == "fit" else 3)
                self.weights_ = None
            else:
                self.error_ = None

    def _validate_partial_fit_fallback(self) -> None:
        """Restrict online fallbacks to reusing the supplied previous weights."""
        if self.fallback is None or self.fallback == _PREVIOUS_WEIGHTS:
            return
        raise ValueError(
            "`partial_fit` only supports fallback=None or fallback='previous_weights'."
        )

    def _run_fallback_chain(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        primary_error: Exception,
        **fit_params: Any,
    ) -> None:
        """Execute the configured fallback chain after an optimization failure.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Training data passed to `fit`.

        y : array-like or None
            Optional target data.

        primary_error : Exception
            The exception raised by the primary estimator.

        **fit_params : dict
            Additional keyword arguments routed separately to each fallback's `fit`.

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

        if not isinstance(fallback, list | tuple):
            fallback = [fallback]

        last_error: Exception = primary_error
        for fb in fallback:
            try:
                fb = _validate_fallback(fb)
                if fb == _PREVIOUS_WEIGHTS:
                    self._fallback_to_previous_weights(X)
                    self.fallback_chain_.append((_PREVIOUS_WEIGHTS, "success"))
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

                params = _fallback_fit_params(
                    fb_est, fit_params, owner=self.__class__.__name__
                )
                fb_est.fit(X, y, **params)

                # A fallback with raise_on_failure=False can return without weights.
                if fb_est.weights_ is None:
                    raise RuntimeError(
                        fb_est.error_ or "Fallback estimator returned no weights."
                    )

                # Success: copy learned artifacts back to self
                self.weights_ = fb_est.weights_
                self.n_features_in_ = fb_est.n_features_in_
                if hasattr(fb_est, "feature_names_in_"):
                    self.feature_names_in_ = fb_est.feature_names_in_
                elif hasattr(self, "feature_names_in_"):
                    del self.feature_names_in_

                self.fallback_ = fb_est
                self.fallback_chain_.append((str(fb_est), "success"))
                return
            except Exception as err:  # try next fallback
                last_error = err
                self.fallback_chain_.append((str(fb), str(err)))

        # All fallbacks failed. The caller decides based on raise_on_failure.
        raise last_error

    def _fallback_to_previous_weights(self, X: ArrayLike) -> None:
        """Reuse previous holdings aligned to the current asset schema.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Current returns, used to align holdings to the full asset schema.

        Raises
        ------
        RuntimeError
            If `previous_weights` is `None` when the fallback is requested.

        ValueError
            If supplied holdings have an invalid shape or contain non-finite values.
        """
        if self.previous_weights is None:
            raise RuntimeError(
                "Fallback 'previous_weights' requested, but 'previous_weights' is None. "
                "Provide valid previous weights or remove this fallback."
            )
        # Align holdings to the current X, even if fit failed before setting asset names.
        validate_data(self, X, reset=True, skip_check_array=True)
        weights = self._clean_previous_weights(
            n_assets=self.n_features_in_,
            apply_investable_mask=False,
        )
        if not np.isfinite(weights).all():
            raise ValueError("previous_weights must be finite.")
        investable_mask = getattr(self, "investable_mask_", None)
        if investable_mask is not None:
            weights = np.where(investable_mask, weights, 0)
        self.weights_ = weights
        self.fallback_ = _PREVIOUS_WEIGHTS

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BaseOptimization:
        """Fit the optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        Returns
        -------
        self : BaseOptimization
            Fitted estimator.
        """
        ...

    def predict(self, X: ArrayLike | ReturnDistribution) -> Portfolio | Population:
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
            ptf_kwargs: dict[str, Any] = {}
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
            _TRANSACTION_COSTS,
            _MANAGEMENT_FEES,
            _PREVIOUS_WEIGHTS,
            _RISK_FREE_RATE,
        ]:
            if param not in ptf_kwargs and hasattr(self, param):
                ptf_kwargs[param] = getattr(self, param)

        # If 'name' is not provided in the portfolio arguments, we use the first
        name = ptf_kwargs.pop("name", type(self).__name__)

        # Forward the fallback attempts to the predicted portfolio.
        ptf_kwargs["fallback_chain"] = getattr(self, "fallback_chain_", None)

        # If weights are None and raise_on_failure is False, we return a FailedPortfolio
        if self.weights_ is None:
            return FailedPortfolio(
                name=name,
                optimization_error=self.error_,  # ty: ignore[invalid-argument-type]
                **ptf_kwargs,
            )

        if not isinstance(X, ReturnDistribution):
            _ = validate_data(self, X, reset=False, skip_check_array=True)

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

    def score(self, X: ArrayLike | ReturnDistribution, y: None = None) -> float:
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
            or the mean of all the portfolio Sharpe Ratios if the prediction is a
            `Population` of `Portfolio`.
        """
        result = self.predict(X)
        if isinstance(result, Population):
            return result.measures_mean(RatioMeasure.SHARPE_RATIO)
        return result.sharpe_ratio

    def fit_predict(self, X: ArrayLike) -> Portfolio | Population:
        """Perform `fit` on `X` and returns the predicted `Portfolio` or
        `Population` of `Portfolio` on `X` based on the fitted `weights`.
        For factor models, use `fit(X, factors=...)` then `predict(X)` separately.

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

        Used by `cross_val_predict` and `online_predict` to decide whether to run
        sequentially and pass the weights from the previous rebalancing to the next.
        This is `True` when `portfolio_params` sets `weight_drift=True`, or when
        transaction costs, a maximum turnover, or a fallback depending on
        `previous_weights` are present.
        """
        if (getattr(self, "portfolio_params", None) or {}).get("weight_drift", False):
            return True

        if _has_transaction_cost(getattr(self, _TRANSACTION_COSTS, None)):
            return True

        if getattr(self, "max_turnover", None) is not None:
            return True

        fallback = self.fallback
        if fallback is not None:
            if not isinstance(fallback, list | tuple):
                fallback = [fallback]

            for fb in fallback:
                fb = _validate_fallback(fb)
                if fb == _PREVIOUS_WEIGHTS or fb.needs_previous_weights:
                    return True

        return False

    def _prepare_investable_distribution(
        self, return_distribution: ReturnDistribution, slim: bool = False
    ) -> ReturnDistribution:
        """Prepare the return distribution used by the optimizer.

        The input `return_distribution` is defined on the full asset universe. This
        method stores its `investable_mask` in `investable_mask_`, then returns the
        distribution restricted to assets that can be used in the optimization problem.
        Downstream helpers use `investable_mask_` to map user inputs and optimized
        weights between the full universe and the investable subset.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            Full-universe return distribution. Non-investable assets may be represented
            by NaNs in `mu`, `covariance` or both.

        slim : bool, default=False
            If True, drop heavy diagnostic fields from the nested factor model when the
            investable subset is built.

        Returns
        -------
        ReturnDistribution
            Return distribution restricted to investable assets.
        """
        self.investable_mask_ = return_distribution.investable_mask
        return return_distribution.investable_subset(slim=slim)

    def _expand_weights_to_full_universe(self, weights: FloatArray) -> FloatArray:
        """Expand investable-subset weights to the full asset universe.

        Optimization is performed on the investable subset prepared by
        `_prepare_investable_distribution`. This method maps the optimized weights back
        to the full universe, filling non-investable positions with zero so that
        `weights_` stays aligned with the original assets passed to `fit`.

        If `investable_mask_` is missing or None, all assets are investable and
        `weights` is returned unchanged.

        Parameters
        ----------
        weights : ndarray of shape (n_investable_assets,) or (..., n_investable_assets)
            Optimized weights on the investable subset.

        Returns
        -------
        ndarray of shape (n_assets,) or (..., n_assets)
            Weights aligned with the full asset universe.
        """
        investable_mask = getattr(self, "investable_mask_", None)

        if investable_mask is None:
            return weights

        n_full_universe = len(investable_mask)

        if weights.ndim == 1:
            full_weights = np.zeros(n_full_universe, dtype=weights.dtype)
            full_weights[investable_mask] = weights
        else:
            full_weights = np.zeros(
                (*weights.shape[:-1], n_full_universe), dtype=weights.dtype
            )
            full_weights[..., investable_mask] = weights
        return full_weights

    def _clean_input(
        self,
        value: float | dict | ArrayLike | None,
        n_assets: int,
        fill_value: float,
        name: str,
        *,
        apply_investable_mask: bool = True,
    ) -> float | FloatArray:
        """Convert input to a cleaned float or 1D ndarray.

        By default, dictionary keys are resolved against the full asset names and arrays
        are restricted to `investable_mask_` when it is available.

        Parameters
        ----------
        value : float | dict | array-like | None
            Input value to clean.

        n_assets : int
            Expected number of assets in the output array. Use the full asset count when
            `apply_investable_mask=False`.

        fill_value : float
            When `value` is a dictionary, keys not present in the asset names are filled
            with `fill_value` in the converted array.

        name : str
            Name used for error messages.

        apply_investable_mask : bool, default=True
            Whether to restrict arrays and resolved dictionaries to investable assets.
            Set to False to keep all input assets.

        Returns
        -------
        float | ndarray of shape (n_assets,)
            The cleaned scalar or 1D array.
        """
        if value is None:
            return fill_value
        if isinstance(value, numbers.Real):
            return float(value)
        return input_to_array(
            items=value,
            n_assets=n_assets,
            fill_value=fill_value,
            dim=1,
            assets_names=getattr(self, "feature_names_in_", None),
            investable_mask=(
                getattr(self, "investable_mask_", None)
                if apply_investable_mask
                else None
            ),
            name=name,
        )

    def _clean_previous_weights(
        self, n_assets: int, *, apply_investable_mask: bool = True
    ) -> FloatArray:
        """Return validated previous weights as a 1D array of length `n_assets`.

        Converts `previous_weights` to a numpy array using `_clean_input`, accepting
        scalars, mappings keyed by asset name, or array-like inputs. Scalars are
        broadcast to all assets. Missing assets in mappings are filled with zeros.

        Parameters
        ----------
        n_assets : int
            Number of assets used to validate shape and broadcast scalars. Use the full
            asset count when `apply_investable_mask=False`.

        apply_investable_mask : bool, default=True
            Whether to restrict arrays and resolved dictionaries to investable assets.
            Set to False to keep all input assets.

        Returns
        -------
        ndarray of shape (n_assets,)
            Cleaned previous weights.
        """
        previous_weights = self._clean_input(
            self.previous_weights,
            n_assets=n_assets,
            fill_value=0,
            name=_PREVIOUS_WEIGHTS,
            apply_investable_mask=apply_investable_mask,
        )
        if not isinstance(previous_weights, np.ndarray):
            previous_weights = np.full(n_assets, previous_weights, dtype=float)
        return previous_weights


def _check_finite_weights(weights: FloatArray) -> None:
    """Raise if computed allocation weights are non-finite.

    Parameters
    ----------
    weights : ndarray
        Computed allocation weights.

    Raises
    ------
    OptimizationError
        If any weight is non-finite.
    """
    if not np.isfinite(weights).all():
        raise OptimizationError("Allocation produced non-finite weights.")


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
        if fallback != _PREVIOUS_WEIGHTS:
            raise ValueError(
                f"Unsupported string fallback: {fallback!r}. Only 'previous_weights' is allowed."
            )
        return _PREVIOUS_WEIGHTS
    if not isinstance(fallback, BaseOptimization):
        raise TypeError(
            f"Fallback estimators must inherit from BaseOptimization (got {type(fallback).__name__})."
        )
    return fallback


def _fallback_fit_params(
    fallback: BaseOptimization, fit_params: dict[str, Any], owner: str
) -> dict[str, Any]:
    """Select the fit parameters forwarded to a fallback estimator.

    With metadata routing enabled, or when the fallback routes metadata to
    sub-estimators, the parameters follow its metadata requests. Otherwise, the
    fallback receives the parameters accepted by its `fit` signature.

    Parameters
    ----------
    fallback : BaseOptimization
        The fallback estimator.

    fit_params : dict
        Fit parameters passed to the primary estimator.

    owner : str
        Name of the primary estimator, used in routing error messages.

    Returns
    -------
    dict
        Fit parameters for the fallback's `fit`.
    """
    if not fit_params:
        return {}
    routing = skm.get_routing_for_object(fallback)
    if (
        isinstance(routing, skm.MetadataRouter)
        or sk.get_config()["enable_metadata_routing"]
    ):
        router = skm.MetadataRouter(owner=owner).add(
            fallback=routing,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router.route_params(caller="fit", params=fit_params).fallback.fit
    return _filter_supported_params(fallback, "fit", **fit_params)


def _has_transaction_cost(x: object) -> bool:
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
    except (TypeError, ValueError, OverflowError):
        # If coercion fails, assume non-zero to be conservative
        return True

    if arr.size == 0:
        return False

    return not np.allclose(arr, 0.0, atol=1e-15, rtol=1e-18, equal_nan=False)
