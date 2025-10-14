"""Equilibrium Expected Returns (Mu) Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance import BaseCovariance, EmpiricalCovariance
from skfolio.moments.expected_returns._base import BaseMu
from skfolio.utils.tools import check_estimator


class EquilibriumMu(BaseMu):
    r"""Equilibrium Expected Returns (Mu) estimator.

    The Equilibrium is defined as:

        .. math:: risk\_aversion \times \Sigma \cdot w^T

    For Market Cap Equilibrium, the weights are the assets Market Caps.
    For Equal-weighted Equilibrium, the weights are equal-weighted (1/N).

    Parameters
    ----------
    risk_aversion : float, default=1.0
        Risk aversion factor.
        The default value is `1.0`.

    weights : array-like of shape (n_assets,), optional
        Asset weights used to compute the Expected Return Equilibrium.
        The default is to use the equal-weighted equilibrium (1/N).
        For a Market Cap weighted equilibrium, you must provide the asset Market Caps.

    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` used to estimate the
        covariance in the equilibrium formula.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
          Estimated expected returns of the assets.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        risk_aversion: float = 1,
        weights: np.ndarray | None = None,
        covariance_estimator: BaseCovariance | None = None,
    ):
        self.risk_aversion = risk_aversion
        self.weights = weights
        self.covariance_estimator = covariance_estimator

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            covariance_estimator=self.covariance_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "EquilibriumMu":
        """Fit the EquilibriumMu estimator model.

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
        self : EquilibriumMu
            Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

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
        n_assets = X.shape[1]
        if self.weights is None:
            weights = np.ones(n_assets) / n_assets
        else:
            weights = np.asarray(self.weights)
        self.mu_ = self.risk_aversion * self.covariance_estimator_.covariance_ @ weights
        return self
