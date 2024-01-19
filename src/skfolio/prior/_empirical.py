"""Empirical Prior Model estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np
import numpy.typing as npt

from skfolio.moments import BaseCovariance, BaseMu, EmpiricalCovariance, EmpiricalMu
from skfolio.prior._base import BasePrior, PriorModel
from skfolio.utils.tools import check_estimator


class EmpiricalPrior(BasePrior):
    """Empirical Prior estimator.

    The Empirical Prior estimates the :class:`~skfolio.prior.PriorModel` by fitting a
    `mu_estimator` and a `covariance_estimator` separately.

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
    prior_model_ : PriorModel
        The assets :class:`~skfolio.prior.PriorModel`.

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

    def fit(self, X: npt.ArrayLike, y=None) -> "EmpiricalPrior":
        """Fit the Empirical Prior estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EmpiricalPrior
            Fitted estimator.
        """
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
            self.mu_estimator_.fit(X)
            mu = self.mu_estimator_.mu_

            # Covariance
            self.covariance_estimator_.fit(X)
            covariance = self.covariance_estimator_.covariance_
        else:
            if self.investment_horizon is None:
                raise ValueError(
                    "`investment_horizon` must be provided when "
                    "`is_log_normal` is `True`"
                )
            # Convert linear returns to log returns
            X_log = np.log(1 + X)

            # Estimates the moments on the log returns
            # Expected returns
            self.mu_estimator_.fit(X_log)
            mu = self.mu_estimator_.mu_

            # Covariance
            self.covariance_estimator_.fit(X_log)
            covariance = self.covariance_estimator_.covariance_

            # Using the property of aggregation across time we scale this distribution
            # to the investment horizon by the “square-root rule”.
            mu *= self.investment_horizon
            covariance *= self.investment_horizon

            # We convert it into a distribution of linear returns over the investment
            # horizon
            mu = np.exp(mu + 0.5 * np.diag(covariance))
            covariance = np.outer(mu, mu) * (np.exp(covariance) - 1)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = self._validate_data(X)
        self.prior_model_ = PriorModel(
            mu=mu,
            covariance=covariance,
            returns=X,
        )
        return self
