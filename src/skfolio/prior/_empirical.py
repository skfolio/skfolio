"""Empirical Prior estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments import BaseCovariance, BaseMu, EmpiricalCovariance, EmpiricalMu
from skfolio.prior._base import BasePrior, ReturnDistribution
from skfolio.utils.tools import check_estimator


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
    ):
        self.mu_estimator = mu_estimator
        self.covariance_estimator = covariance_estimator
        self.is_log_normal = is_log_normal
        self.investment_horizon = investment_horizon

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add(
                mu_estimator=self.mu_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                covariance_estimator=self.covariance_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "EmpiricalPrior":
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
        routed_params = skm.process_routing(self, "fit", **fit_params)

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
        # fitting estimators
        if not self.is_log_normal:
            if self.investment_horizon is not None:
                raise ValueError(
                    "`investment_horizon` must be `None` when "
                    "`is_log_normal` is `False`"
                )
            # Expected returns
            # noinspection PyArgumentList
            self.mu_estimator_.fit(X, y, **routed_params.mu_estimator.fit)
            mu = self.mu_estimator_.mu_

            # Covariance
            # noinspection PyArgumentList
            self.covariance_estimator_.fit(
                X, y, **routed_params.covariance_estimator.fit
            )
            covariance = self.covariance_estimator_.covariance_
        else:
            if self.investment_horizon is None:
                raise ValueError(
                    "`investment_horizon` must be provided when "
                    "`is_log_normal` is `True`"
                )
            # Convert linear returns to log returns
            X_log = np.log(1 + X)
            y_log = np.log(1 + y) if y is not None else None

            # Estimates the moments on the log returns
            # Expected returns
            # noinspection PyArgumentList
            self.mu_estimator_.fit(X_log, y_log, **routed_params.mu_estimator.fit)
            mu = self.mu_estimator_.mu_

            # Covariance
            # noinspection PyArgumentList
            self.covariance_estimator_.fit(
                X_log, y_log, **routed_params.covariance_estimator.fit
            )
            covariance = self.covariance_estimator_.covariance_

            # Using the property of aggregation across time we scale this distribution
            # to the investment horizon by the “square-root rule”.
            mu *= self.investment_horizon
            covariance *= self.investment_horizon

            # We convert it into a distribution of linear returns over the investment
            # horizon
            mu = np.exp(mu + 0.5 * np.diag(covariance))
            covariance = np.outer(mu, mu) * (np.exp(covariance) - 1)
            mu -= 1

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = skv.validate_data(self, X)
        self.return_distribution_ = ReturnDistribution(
            mu=mu,
            covariance=covariance,
            returns=X,
        )
        return self
