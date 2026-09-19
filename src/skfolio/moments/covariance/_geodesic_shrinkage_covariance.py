"""Geodesic Shrinkage Covariance Estimators."""

# Copyright (c) 2023-2026
# Author: Lucas Morin <lucas.cr.morin@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# randomcov, Copyright (c) 2024, Peter Cotton. Licensed under the MIT license.
# https://github.com/microprediction/randomcov

from __future__ import annotations

from enum import auto

import numpy as np
import scipy.linalg as scl
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._empirical_covariance import EmpiricalCovariance
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.stats import assert_is_symmetric, cov_nearest
from skfolio.utils.tools import AutoEnum, _validate_unit_interval, check_estimator


class GeodesicShrinkageTarget(AutoEnum):
    """Target for geodesic covariance shrinkage.

    Attributes
    ----------
    SCALED_IDENTITY : str
        Identity matrix scaled by the average asset variance.
    DIAGONAL : str
        Diagonal matrix containing the asset variances.
    """

    SCALED_IDENTITY = auto()
    DIAGONAL = auto()


class GeodesicShrinkageCovariance(BaseCovariance):
    r"""Covariance estimator with geodesic shrinkage.

    Classical shrinkage estimators (e.g. :class:`~skfolio.moments.ShrunkCovariance`,
    :class:`~skfolio.moments.LedoitWolf`) regularize the sample covariance matrix `S`
    towards a well-conditioned target `T` using a linear (Euclidean) combination:

    .. math:: (1 - \alpha) \cdot S + \alpha \cdot T

    `GeodesicShrinkageCovariance` instead moves `S` towards `T` along the
    geodesic of the manifold of Symmetric Positive Definite (SPD) matrices under
    the affine-invariant Riemannian metric (AIRM) [1]_ [2]_:

    .. math:: \Sigma(\alpha) = S^{1/2} \left(S^{-1/2} \, T \, S^{-1/2}\right)^{\alpha} S^{1/2}

    At `shrinkage` :math:`\alpha = 0`, the estimate is the starting covariance
    from `covariance_estimator`, after any repair requested by `nearest`. At
    :math:`\alpha = 1`, it is `target`, subject to the final `nearest` repair.

    The estimate remains SPD for every `shrinkage` in [0, 1] whenever `S` and
    `target` are themselves positive definite.

    When `target` commutes with `S`, their shared eigenvectors are preserved and
    their eigenvalues are interpolated geometrically rather than arithmetically.
    For the default `SCALED_IDENTITY` target :math:`\mu I`, with
    :math:`\mu = \mathrm{trace}(S) / n`, the interpolated eigenvalues are
    :math:`\lambda_i^{1-\alpha} \mu^{\alpha}`, whereas linear shrinkage gives
    :math:`(1 - \alpha) \lambda_i + \alpha \mu`. The condition number of the
    geodesic estimate is exactly :math:`\kappa(S)^{1-\alpha}`. Unlike linear
    shrinkage, the trace is generally not preserved at intermediate values.

    The general interpolation is computed with a generalized eigendecomposition
    relative to the target, avoiding an explicit inverse square root of `S`.

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

    target : GeodesicShrinkageTarget or array-like of shape (n_assets, n_assets), default=GeodesicShrinkageTarget.SCALED_IDENTITY
        The shrinkage target `T`. The string values `"scaled_identity"` and
        `"diagonal"` are also accepted:

            - `SCALED_IDENTITY`: :math:`\mu \cdot I`, with :math:`\mu = \mathrm{trace}(S) / n`.
              This is the same default target used by scikit-learn's
              `ShrunkCovariance`. Its condition number is exactly 1, so `shrinkage`
              controls how far the estimate moves from `S` towards a perfectly
              well-conditioned matrix.
            - `DIAGONAL`: :math:`\mathrm{diag}(S)`. At `shrinkage=1` the
              variances equal those of `S` and all correlations are zero. At
              intermediate values the geodesic also changes the variances.
            - array-like: a user-provided SPD target matrix of shape
              `(n_assets, n_assets)`. A well-conditioned target is recommended
              for numerical accuracy.

    nearest : bool, default=True
        If this is set to True, the starting covariance is repaired before
        interpolation and the resulting covariance is repaired afterwards. Custom
        targets must be positive definite and are never repaired.
        The covariance is replaced by the nearest covariance
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

    location_ : ndarray of shape (n_assets,)
        Estimated mean, available when the fitted covariance estimator exposes it.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "Positive Definite Matrices".
        Bhatia, R. (2007). Princeton University Press.

    .. [2]  "Geodesically parameterized covariance estimation".
        Musolas, A., Smith, S.T. & Marzouk, Y. (2021).
        SIAM Journal on Matrix Analysis and Applications.
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        shrinkage: float = 0.1,
        target: GeodesicShrinkageTarget
        | ArrayLike = GeodesicShrinkageTarget.SCALED_IDENTITY,
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
        _validate_unit_interval(self.shrinkage, "shrinkage")

        routed_params = skm.process_routing(self, "fit", **fit_params)

        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X, y, **routed_params.covariance_estimator.fit)

        if hasattr(self.covariance_estimator_, "location_"):
            self.location_ = self.covariance_estimator_.location_
        elif hasattr(self, "location_"):
            del self.location_

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = skv.validate_data(self, X)
        start = np.asarray(self.covariance_estimator_.covariance_)
        if self.nearest:
            start = cov_nearest(
                start,
                higham=self.higham,
                higham_max_iteration=self.higham_max_iteration,
                warn=True,
            )
        target, scaled_identity = self._build_target_covariance(start)
        covariance = _geodesic_interpolation(
            start=start,
            end=target,
            alpha=self.shrinkage,
            scaled_identity=scaled_identity,
        )

        self._set_covariance(covariance)
        return self

    def _build_target_covariance(self, start: FloatArray) -> tuple[FloatArray, bool]:
        """Build the SPD shrinkage target matrix `T` with the same shape as `start`.

        Parameters
        ----------
        start : ndarray of shape (n_assets, n_assets)
            Starting covariance matrix, used to determine the target shape and
            the variances of the built-in targets.

        Returns
        -------
        target : ndarray of shape (n_assets, n_assets)
            SPD shrinkage target matrix.

        scaled_identity : bool
            Whether the target was constructed using the scaled-identity option,
            enabling direct eigenvalue interpolation.

        Raises
        ------
        ValueError
            If the target option is unknown or a custom target has an invalid
            shape, is not symmetric, contains non-finite values, or is not
            positive definite.
        """
        n_assets = start.shape[0]
        if isinstance(self.target, str):
            if self.target == GeodesicShrinkageTarget.SCALED_IDENTITY:
                mu = float(np.trace(start)) / n_assets
                return mu * np.identity(n_assets), True
            if self.target == GeodesicShrinkageTarget.DIAGONAL:
                return np.diag(np.diag(start)), False
            raise ValueError(
                "target must be 'scaled_identity', 'diagonal', or an array-like SPD "
                f"matrix, got string {self.target!r}"
            )

        target = np.asarray(self.target, dtype=float)
        if target.shape != (n_assets, n_assets):
            raise ValueError(
                "target must be a square matrix of shape "
                f"({n_assets}, {n_assets}), got shape {target.shape}"
            )
        assert_is_symmetric(target)
        if not np.all(np.isfinite(target)):
            raise ValueError("target must contain only finite values")
        if np.any(np.linalg.eigvalsh(target) <= 0):
            raise ValueError("target must be positive definite")
        return target, False


def _geodesic_interpolation(
    start: FloatArray, end: FloatArray, alpha: float, *, scaled_identity: bool = False
) -> FloatArray:
    r"""Interpolate between two SPD matrices along the affine-invariant geodesic.

    Unlike the linear (Euclidean) interpolation used by classical shrinkage
    estimators, :math:`(1 - \alpha) \cdot S + \alpha \cdot T`, the geodesic
    interpolation moves `start` (:math:`S`) towards `end` (:math:`T`) along the
    shortest path on the Riemannian manifold of Symmetric Positive Definite (SPD)
    matrices equipped with the affine-invariant metric [1]_:

    .. math::
        \Sigma(\alpha) = S^{1/2} \left(S^{-1/2} \, T \, S^{-1/2}\right)^{\alpha} S^{1/2}

    At :math:`\alpha = 0`, :math:`\Sigma = S`, and at :math:`\alpha = 1`,
    :math:`\Sigma = T`. Because it follows the natural geometry of the SPD manifold
    rather than a straight Euclidean line, the interpolated matrix is guaranteed to
    remain SPD for every :math:`\alpha \in [0, 1]` whenever `start` and `end` are
    SPD.

    The general calculation solves :math:`S V = T V \Lambda` with
    :math:`V^T T V = I`. Setting :math:`B = T V` gives the equivalent expression
    :math:`\Sigma(\alpha) = B \Lambda^{1-\alpha} B^T`, avoiding an explicit
    inverse square root of `start`.

    Parameters
    ----------
    start : ndarray of shape (n, n)
        Starting SPD matrix, returned as a copy when `alpha` is 0.

    end : ndarray of shape (n, n)
        Target SPD matrix, returned as a copy when `alpha` is 1.

    alpha : float
        Interpolation intensity between 0 and 1 inclusive.

    scaled_identity : bool, default=False
        Whether `end` is a scalar multiple of the identity. Uses the shared
        eigenvectors to interpolate the eigenvalues directly.

    Returns
    -------
    interpolated : ndarray of shape (n, n)
        The interpolated SPD matrix.

    Raises
    ------
    ValueError
        If `start` and `end` are not square, symmetric, or of the same shape,
        or if `alpha` is outside [0, 1]. Positive definiteness is checked only
        for interior values of `alpha`; endpoints are returned as copies.

    References
    ----------
    .. [1]  "Positive Definite Matrices".
        Bhatia, R. (2007). Princeton University Press.
    """
    assert_is_symmetric(start)
    assert_is_symmetric(end)
    if start.shape != end.shape:
        raise ValueError(
            "`start` and `end` must have the same shape, got "
            f"{start.shape} and {end.shape}"
        )
    _validate_unit_interval(alpha, "alpha")

    if alpha == 0.0:
        return start.copy()
    if alpha == 1.0:
        return end.copy()

    if scaled_identity:
        eigvals_s, eigvecs_s = np.linalg.eigh(start)
        if np.any(eigvals_s <= 0):
            raise ValueError("`start` must be positive definite")
        eigenvalues = eigvals_s ** (1.0 - alpha) * end[0, 0] ** alpha
        interpolated = (eigvecs_s * eigenvalues) @ eigvecs_s.T
        return (interpolated + interpolated.T) / 2.0

    try:
        eigenvalues, eigenvectors = scl.eigh(start, end)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Geodesic eigendecomposition failed; `end` must be positive definite"
        ) from error
    if np.any(eigenvalues <= 0):
        raise ValueError("`start` must be positive definite")

    basis = end @ eigenvectors
    interpolated = (basis * eigenvalues ** (1.0 - alpha)) @ basis.T
    return (interpolated + interpolated.T) / 2.0
