"""Empirical Prior estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments import BaseCovariance, BaseMu, EmpiricalCovariance, EmpiricalMu
from skfolio.prior._base import BasePrior, ReturnDistribution
from skfolio.utils._array_buffer import _ArrayBuffer
from skfolio.utils.tools import _call_estimator, check_estimator

_FITTED_ATTR = "return_distribution_"


# TODO for next release: allow NANs and align NANs policy inside ReturnDistribution dataclass.
class EmpiricalPrior(BasePrior):
    """Empirical Prior estimator.

    The Empirical Prior estimates the :class:`~skfolio.prior.ReturnDistribution` by
    fitting a `mu_estimator` and a `covariance_estimator` separately.

    Parameters
    ----------
    mu_estimator : BaseMu, optional
        The assets :ref:`expected returns estimator <mu_estimator>`.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalMu`.

    covariance_estimator : BaseCovariance , optional
        The assets :ref:`covariance matrix estimator <covariance_estimator>`.
        The default (`None`) is to use  :class:`~skfolio.moments.EmpiricalCovariance`.

    is_log_normal : bool, default=False
        If this is set to True, the moments are estimated on the logarithmic returns
        as opposed to the linear returns. Then the moments estimations of the
        logarithmic returns are projected to the investment horizon and transformed
        to obtain the moments estimation of the linear returns at the investment
        horizon. If True, `investment_horizon` must be provided. The input `X` must be
        **linear returns**. They will be converted into logarithmic returns only for the
        moments estimation.

        .. seealso::

            :ref:`data preparation <data_preparation>`


    investment_horizon : float, optional
        The investment horizon used for the moments estimation of the linear returns
        when `is_log_normal` is `True`.

    max_history : int, optional
        Maximum number of observations to keep in `return_distribution_.returns`.
        This is useful for controlling memory usage during incremental learning
        with :meth:`partial_fit`.

        * If `None` (default), all returns are accumulated.
        * If an integer, only the last `max_history` observations are kept
          (rolling window).

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` to be used by the optimization
        estimators, containing the asset returns distribution and moments estimation.

    mu_estimator_ : BaseMu
        Fitted `mu_estimator`.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    References
    ----------
    .. [1]  "Linear vs. Compounded Returns - Common Pitfalls in Portfolio Management".
        GARP Risk Professional.
        Attilio Meucci (2010).
    """

    mu_estimator_: BaseMu
    covariance_estimator_: BaseCovariance
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        mu_estimator: BaseMu | None = None,
        covariance_estimator: BaseCovariance | None = None,
        is_log_normal: bool = False,
        investment_horizon: float | None = None,
        max_history: int | None = None,
    ):
        self.mu_estimator = mu_estimator
        self.covariance_estimator = covariance_estimator
        self.is_log_normal = is_log_normal
        self.investment_horizon = investment_horizon
        self.max_history = max_history

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> EmpiricalPrior:
        """Fit the Empirical Prior estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : EmpiricalPrior
            Fitted estimator.
        """
        self._reset()
        return self._fit(X, y, method="fit", **fit_params)

    def partial_fit(self, X: npt.ArrayLike, y=None, **fit_params) -> EmpiricalPrior:
        """Incrementally fit the Empirical Prior estimator.

        This method allows for streaming/online updates to the prior estimate.
        Each call updates the internal state with new observations.

        Both `mu_estimator` and `covariance_estimator` must implement
        `partial_fit` for this method to work.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : EmpiricalPrior
            Fitted estimator.
        """
        return self._fit(X, y, method="partial_fit", **fit_params)

    def get_metadata_routing(self):
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add(
                mu_estimator=self.mu_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
            .add(
                covariance_estimator=self.covariance_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        )
        return router

    def _fit(self, X: npt.ArrayLike, y, method: str, **fit_params) -> EmpiricalPrior:
        """Core fitting logic shared by fit and partial_fit.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        method : str
            Either "fit" or "partial_fit". Determines which method to call
            on sub-estimators and how to handle returns accumulation.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

        Returns
        -------
        self : EmpiricalPrior
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, method, **fit_params)

        first_call = not hasattr(self, _FITTED_ATTR)

        if first_call:
            self._validate_params()
            self._initialize()

        if self.is_log_normal:
            X_fit = np.log(1 + X)
            y_fit = np.log(1 + y) if y is not None else None
        else:
            X_fit = X
            y_fit = y

        # Fit or partial_fit the mu estimator
        _call_estimator(
            self.mu_estimator_,
            method,
            X_fit,
            y_fit,
            routed_params=routed_params.mu_estimator,
        )
        # Fit or partial_fit the cov estimator
        _call_estimator(
            self.covariance_estimator_,
            method,
            X_fit,
            y_fit,
            routed_params=routed_params.covariance_estimator,
        )

        mu = self.mu_estimator_.mu_
        covariance = self.covariance_estimator_.covariance_

        # Transform log moments to linear if needed
        if self.is_log_normal:
            mu *= self.investment_horizon
            covariance *= self.investment_horizon

            # Convert to linear returns distribution
            mu = np.exp(mu + 0.5 * np.diag(covariance))
            covariance = np.outer(mu, mu) * (np.exp(covariance) - 1)
            mu -= 1

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = skv.validate_data(self, X, ensure_all_finite=True, reset=first_call)

        # Accumulate returns with amortized O(1) appends
        if first_call:
            self._returns_buffer = _ArrayBuffer()
        self._returns_buffer.append(X)

        if self.max_history is not None:
            self._returns_buffer.truncate_to_last(self.max_history)

        self.return_distribution_ = ReturnDistribution(
            mu=mu,
            covariance=covariance,
            returns=self._returns_buffer.array,
        )
        return self

    def _validate_params(self) -> None:
        """Validate parameters."""
        if self.is_log_normal:
            if self.investment_horizon is None:
                raise ValueError(
                    "`investment_horizon` must be provided when "
                    "`is_log_normal` is `True`"
                )
        else:
            if self.investment_horizon is not None:
                raise ValueError(
                    "`investment_horizon` must be `None` when "
                    "`is_log_normal` is `False`"
                )

        if self.max_history is not None:
            if isinstance(self.max_history, bool) or not isinstance(
                self.max_history, numbers.Integral
            ):
                raise ValueError(
                    f"`max_history` must be a positive integer or None, "
                    f"got {self.max_history}"
                )
            if self.max_history < 1:
                raise ValueError(
                    f"`max_history` must be a positive integer or None, "
                    f"got {self.max_history}"
                )

    def _initialize(self):
        self.mu_estimator_ = check_estimator(
            self.mu_estimator,
            default=EmpiricalMu(),
            check_type=BaseMu,
        )
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
