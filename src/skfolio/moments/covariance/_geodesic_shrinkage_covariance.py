"""Geodesic Shrinkage Covariance Estimators."""

# Copyright (c) 2023-2026
# Author: Lucas Morin <lucas.cr.morin@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# randomcov, Copyright (c) 2024, Peter Cotton. Licensed under the MIT license.
# https://github.com/microprediction/randomcov

from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._empirical_covariance import EmpiricalCovariance
from skfolio.typing import ArrayLike
from skfolio.utils.stats import assert_is_symmetric, cov_geodesic_interpolation
from skfolio.utils.tools import check_estimator


class GeodesicShrinkageCovariance(BaseCovariance):
    r"""Covariance estimator with geodesic shrinkage.

    Classical shrinkage estimators (e.g. :class:`~skfolio.moments.ShrunkCovariance`,
    :class:`~skfolio.moments.LedoitWolf`) regularize the sample covariance matrix `S`
    towards a well-conditioned target `T` using a linear (Euclidean) combination:

    .. math:: (1 - \alpha) \cdot S + \alpha \cdot T

    `GeodesicShrinkageCovariance` instead moves `S` towards `T` along the
    affine-invariant geodesic of the manifold of Symmetric Positive Definite (SPD)
    matrices [1]_ (see :func:`~skfolio.utils.stats.cov_geodesic_interpolation`):

    .. math:: \Sigma(\alpha) = S^{1/2} \left(S^{-1/2} \, T \, S^{-1/2}\right)^{\alpha} S^{1/2}

    At `shrinkage` :math:`\alpha = 0`, the estimate is the unmodified covariance from
    `covariance_estimator`. At :math:`\alpha = 1`, it is exactly `target`.

    This estimator was proposed as an alternative to linear shrinkage while
    investigating a numerical-conditioning issue in the Schur complement
    interpolation between :class:`~skfolio.optimization.HierarchicalRiskParity` and
    Minimum-Variance used by :class:`~skfolio.optimization.SchurComplementary` [2]_.
    Both interpolation schemes remain Symmetric Positive Definite for every
    `shrinkage` in [0, 1] whenever `covariance_estimator` and `target` are
    themselves positive definite, and both recover the sample covariance and
    `target` exactly at the two endpoints. They differ in how they treat the
    eigenvalues in between: when `target` commutes with the sample covariance
    (e.g. the default `"identity"` target, a scalar multiple of the identity
    matrix), the geodesic interpolation is a *geometric* interpolation of the
    eigenvalues, :math:`\lambda_i^{1-\alpha} \cdot \mu^{\alpha}`, whereas linear
    shrinkage is an *arithmetic* interpolation,
    :math:`(1 - \alpha) \cdot \lambda_i + \alpha \cdot \mu`. In particular, for the
    identity target the arithmetic interpolation lowers the condition number faster
    than the geometric one as `shrinkage` increases from 0, since a small `shrinkage`
    already lifts the smallest eigenvalues by an additive :math:`\alpha \mu` term.
    Which interpolation is preferable is an empirical question left to the user;
    both are made available so they can be compared, in keeping with skfolio's
    approach of not imposing a default regularization on optimizers such as
    :class:`~skfolio.optimization.SchurComplementary`.

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` used to compute the
        starting covariance matrix `S` (the estimate returned when `shrinkage=0`).
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    shrinkage : float, default=0.1
        Shrinkage intensity `alpha` in the geodesic interpolation. Must be between 0
        (no shrinkage) and 1 (fully shrunk to `target`) inclusive. The default value
        is `0.1`.

    target : str or array-like of shape (n_assets, n_assets), default="identity"
        The shrinkage target `T`:

            - "identity": :math:`\mu \cdot I`, with :math:`\mu = \mathrm{trace}(S) / n`.
              This is the same default target used by scikit-learn's
              `ShrunkCovariance`. Its condition number is exactly 1, so `shrinkage`
              controls how far the estimate moves from `S` towards a perfectly
              well-conditioned matrix.
            - "diagonal": :math:`\mathrm{diag}(S)`, i.e. the correlations are shrunk
              towards zero while each asset's variance is left unchanged.
            - array-like: a user-provided SPD target matrix of shape
              `(n_assets, n_assets)`.

    nearest : bool, default=True
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition that can be
        computed. The variance is left unchanged.
        A covariance matrix that is not positive definite often occurs in high
        dimensional problems. It can be due to multicollinearity, floating-point
        inaccuracies, or when the number of observations is smaller than the number of
        assets. For more details, see :func:`~skfolio.utils.stats.cov_nearest`.
        The default is `True`.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm is used to find the
        nearest PD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and uses the clipping method as
        the Higham algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iterations of the Higham (2002) algorithm.
        The default value is `100`.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "Positive Definite Matrices".
        Bhatia, R. (2007). Princeton University Press.

    .. [2]  "Gracefully link HRP and min var", GitHub Discussion.
        Cotton, P. & Delatte, H. (2024-2025).
        https://github.com/skfolio/skfolio/discussions/3
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        shrinkage: float = 0.1,
        target: str | ArrayLike = "identity",
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.covariance_estimator = covariance_estimator
        self.shrinkage = shrinkage
        self.target = target

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            covariance_estimator=self.covariance_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: ArrayLike, y=None, **fit_params) -> GeodesicShrinkageCovariance:
        """Fit the Geodesic Shrinkage Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using `sklearn.set_config(enable_metadata_routing=True)`.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : GeodesicShrinkageCovariance
            Fitted estimator.
        """
        if not isinstance(self.shrinkage, numbers.Real) or not (
            0.0 <= self.shrinkage <= 1.0
        ):
            raise ValueError(
                f"shrinkage must be a float between 0 and 1, got {self.shrinkage}"
            )

        routed_params = skm.process_routing(self, "fit", **fit_params)

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
        start = np.asarray(self.covariance_estimator_.covariance_)

        target = self._target_matrix(start=start, n_assets=n_assets)

        if self.shrinkage == 0.0:
            covariance = start
        else:
            covariance = cov_geodesic_interpolation(
                start=start, end=target, alpha=self.shrinkage
            )

        self._set_covariance(covariance)
        return self

    def _target_matrix(self, start: npt.NDArray, n_assets: int) -> npt.NDArray:
        """Build the SPD shrinkage target matrix `T` with the same shape as `start`."""
        if isinstance(self.target, str):
            if self.target == "identity":
                mu = float(np.trace(start)) / n_assets
                return mu * np.identity(n_assets)
            if self.target == "diagonal":
                return np.diag(np.diag(start))
            raise ValueError(
                "target must be 'identity', 'diagonal', or an array-like SPD "
                f"matrix, got string {self.target!r}"
            )

        target = np.asarray(self.target, dtype=float)
        if target.shape != (n_assets, n_assets):
            raise ValueError(
                "target must be a square matrix of shape "
                f"({n_assets}, {n_assets}), got shape {target.shape}"
            )
        assert_is_symmetric(target)
        return target
