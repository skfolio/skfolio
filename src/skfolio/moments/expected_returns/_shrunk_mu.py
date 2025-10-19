"""Shrinkage Expected Returns (Mu) Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from enum import auto

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance import BaseCovariance, EmpiricalCovariance
from skfolio.moments.expected_returns._base import BaseMu
from skfolio.utils.tools import AutoEnum, check_estimator


class ShrunkMuMethods(AutoEnum):
    """Shrinkage methods for the ShrunkMu estimator.

    Parameters
    ----------
    JAMES_STEIN : str
        James-Stein method

    BAYES_STEIN : str
        Bayes-Stein method

    BODNAR_OKHRIN : str
        Bodnar Okhrin Parolya method
    """

    JAMES_STEIN = auto()
    BAYES_STEIN = auto()
    BODNAR_OKHRIN = auto()


class ShrunkMu(BaseMu):
    r"""Shrinkage Expected Returns (Mu) estimator.

    Estimates the expected returns using shrinkage.

    The sample mean estimator is unbiased but has high variance.
    Stein (1955) proved that it's possible to find an estimator with reduced total
    error using shrinkage by trading a small bias against high variance.

    The estimator shrinks the sample mean toward a target vector:

        .. math:: \hat{\mu} = \alpha\bar{\mu}+\beta \mu_{target}

    with :math:`\bar{\mu}` the sample mean, :math:`\mu_{target}` the target vector
    and :math:`\alpha` and :math:`\beta` two constants to determine.

    There are two choices for the target vector :math:`\mu_{target}` :

        * Grand Mean: constant vector of the mean of the sample mean
        * Volatility-Weighted Grand Mean: volatility-weighted sample mean

    And three methods for :math:`\alpha` and :math:`\beta` :

        * James-Stein
        * Bayes-Stein
        * Bodnar Okhrin Parolya

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` used to estimate the
        covariance in the shrinkage formulae.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    vol_weighted_target : bool, default=False
        If this is set to True, the target vector :math:`\mu_{target}` is the
        Volatility-Weighted Grand Mean otherwise it is the Grand Mean.
        The default is `False`.

    method : ShrunkMuMethods, default=ShrunkMuMethods.JAMES_STEIN
        Shrinkage method :class:`ShrunkMuMethods`.

        Possible values are:

            * JAMES_STEIN
            * BAYES_STEIN
            * BODNAR_OKHRIN

        The default value is `ShrunkMuMethods.JAMES_STEIN`.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
        Estimated expected returns of the assets.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    mu_target_ : ndarray of shape (n_assets,)
        Target vector :math:`\mu_{target}`.

    alpha_ : float
        Alpha value :math:`\alpha`.

    beta_ : float
        Beta value :math:`\beta`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Risk and Asset Allocation",
        Attilio Meucci (2005)

    .. [2] "Bayes-stein estimation for portfolio analysis",
        Philippe Jorion (1986)

    .. [3] "Optimal shrinkage estimator for high-dimensional mean vector"
        Bodnar, Okhrin and Parolya (2019)
    """

    covariance_estimator_: BaseCovariance
    mu_target_: np.ndarray
    alpha_: float
    beta_: float

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        vol_weighted_target: bool = False,
        method: ShrunkMuMethods = ShrunkMuMethods.JAMES_STEIN,
    ):
        self.covariance_estimator = covariance_estimator
        self.vol_weighted_target = vol_weighted_target
        self.method = method

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            covariance_estimator=self.covariance_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "ShrunkMu":
        """Fit the ShrunkMu estimator model.

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
        self : ShrunkMu
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        if not isinstance(self.method, ShrunkMuMethods):
            raise ValueError(
                "`method` must be of type ShrunkMuMethods, got"
                f" {type(self.method).__name__}"
            )
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        # noinspection PyArgumentList
        self.covariance_estimator_.fit(X, y, **routed_params.covariance_estimator.fit)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = skv.validate_data(self, X)
        n_observations, n_assets = X.shape

        covariance = self.covariance_estimator_.covariance_

        sample_mu = np.mean(X, axis=0)
        cov_inv = None

        # Calculate target vector
        if self.vol_weighted_target:
            cov_inv = np.linalg.inv(covariance)
            self.mu_target_ = np.sum(cov_inv, axis=1) @ sample_mu / np.sum(cov_inv)
        else:
            self.mu_target_ = np.mean(sample_mu)
        self.mu_target_ *= np.ones(n_assets)

        # Calculate Estimators
        match self.method:
            case ShrunkMuMethods.JAMES_STEIN:
                eigenvalues = np.linalg.eigvals(covariance)
                self.beta_ = (
                    (np.sum(eigenvalues) - 2 * np.max(eigenvalues))
                    / np.sum((sample_mu - self.mu_target_) ** 2)
                    / n_observations
                )
                self.alpha_ = 1 - self.beta_
            case ShrunkMuMethods.BAYES_STEIN:
                if cov_inv is None:
                    cov_inv = np.linalg.inv(covariance)
                self.beta_ = (n_assets + 2) / (
                    n_observations
                    * (sample_mu - self.mu_target_).T
                    @ cov_inv
                    @ (sample_mu - self.mu_target_)
                    + (n_assets + 2)
                )
                self.alpha_ = 1 - self.beta_
            case ShrunkMuMethods.BODNAR_OKHRIN:
                if cov_inv is None:
                    cov_inv = np.linalg.inv(covariance)
                u = sample_mu.T @ cov_inv @ sample_mu
                v = sample_mu.T @ cov_inv @ self.mu_target_
                w = self.mu_target_.T @ cov_inv @ self.mu_target_
                self.alpha_ = (
                    (u - n_assets / (n_observations - n_assets)) * w - v**2
                ) / (u * w - v**2)
                self.beta_ = (1 - self.alpha_) * v / u
            case _:
                raise ValueError(f"method {self.method} is not valid")

        self.mu_ = self.alpha_ * sample_mu + self.beta_ * self.mu_target_
        return self
