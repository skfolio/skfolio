"""Covariance distance estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.distance._base import BaseDistance
from skfolio.distance._utils import _corr_to_distance
from skfolio.moments import BaseCovariance, GerberCovariance
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.stats import (
    assert_is_symmetric,
    cov_to_corr,
    is_positive_semidefinite,
    symmetrize,
)
from skfolio.utils.tools import _call_estimator, check_estimator
from skfolio.utils.validation import _validate_pairwise_matrix

_FITTED_ATTR = "distance_"


class CovarianceDistance(BaseDistance):
    r"""Covariance Distance estimator.

    The codependence is computed from the correlation matrix of a chosen
    :ref:`covariance estimator <covariance_estimator>` to which is applied
    a power and/or absolute transformation.
    This codependence is then used to compute the distance matrix.
    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`

    Parameters
    ----------
    covariance_estimator : BaseCovariance or {"precomputed"}, optional
       :ref:`Covariance estimator <covariance_estimator>`.
       The default (`None`) is to use :class:`~skfolio.moments.GerberCovariance`.
       With `"precomputed"`, fit converts a covariance matrix directly.
       NaN diagonal entries exclude assets, whose output rows and columns stay
       NaN. The available block must be symmetric and positive semidefinite,
       with strictly positive variances. Validation allows float64 roundoff of
       `64 * eps * n_available_assets` in the implied correlation. No repair is
       applied.

    absolute : bool, default=False
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.
        The default is `False`.

    power : float, default=1
        Exponent of the power transformation applied to the correlation matrix.
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.
        The default value is `1`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    covariance_estimator_ : BaseCovariance or None
        Fitted covariance estimator, or None in precomputed mode.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    Notes
    -----
    Both learned and precomputed covariances must be symmetric and positive
    semidefinite within numerical tolerance. Singular covariances are accepted.
    Invalid covariances raise `ValueError`. For covariance estimators supporting
    `nearest`, keep `nearest=True` to repair their output before conversion.
    Precomputed covariance can be repaired explicitly with
    :func:`~skfolio.utils.stats.cov_nearest`.

    References
    ----------
    .. [1] "Building Diversified Portfolios that Outperform Out-of-Sample",
        López de Prado, Journal of Portfolio Management (2016)
    """

    covariance_estimator_: BaseCovariance | None

    def __init__(
        self,
        covariance_estimator: BaseCovariance | Literal["precomputed"] | None = None,
        absolute: bool = False,
        power: float = 1,
    ) -> None:
        self.covariance_estimator = covariance_estimator
        self.absolute = absolute
        self.power = power

    def fit(
        self,
        X: ArrayLike,
        y: None = None,
        **fit_params: Any,
    ) -> CovarianceDistance:
        """Fit the Covariance Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets) or (n_assets, n_assets)
            Returns in learned mode, or a covariance matrix in precomputed mode.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Metadata routed to the covariance estimator.

        Returns
        -------
        self : CovarianceDistance
            Fitted estimator.
        """
        self._reset()
        return self._fit(X, y, method="fit", **fit_params)

    def partial_fit(
        self,
        X: ArrayLike,
        y: None = None,
        **fit_params: Any,
    ) -> CovarianceDistance:
        """Update covariance from new returns and recompute the distance.

        The covariance estimator must implement `partial_fit`. The first call
        also uses the child's `partial_fit`. With `covariance_estimator="precomputed"`,
        call `fit` with the updated covariance matrix instead.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            New asset returns with the same full schema as previous calls.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Metadata routed to the covariance estimator's partial_fit.

        Returns
        -------
        self : CovarianceDistance
            Updated estimator.

        Raises
        ------
        TypeError
            If the covariance estimator does not implement `partial_fit` or
            `covariance_estimator="precomputed"`.
        """
        if self.requires_covariance_input:
            raise TypeError(
                'partial_fit is not supported with covariance_estimator="precomputed". '
                "Call fit with the updated covariance matrix instead."
            )
        return self._fit(X, y, method="partial_fit", **fit_params)

    @property
    def requires_covariance_input(self) -> bool:
        """Whether `covariance_estimator` is set to `"precomputed"`."""
        return (
            isinstance(self.covariance_estimator, str)
            and self.covariance_estimator == "precomputed"
        )

    def get_metadata_routing(self) -> skm.MetadataRouter:
        """Get metadata routing for this estimator.

        Routes metadata to the corresponding `fit` or `partial_fit` method of
        `covariance_estimator`.

        Returns
        -------
        routing : MetadataRouter
            Metadata routing configuration.
        """
        router = skm.MetadataRouter(owner=self.__class__.__name__)
        if not isinstance(self.covariance_estimator, str):
            router.add(
                covariance_estimator=self.covariance_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        return router

    def _fit(
        self,
        X: ArrayLike,
        y: None,
        method: str,
        **fit_params: Any,
    ) -> CovarianceDistance:
        """Share initialization, metadata routing and covariance conversion."""
        routed = skm.process_routing(self, method, **fit_params)
        first_call = not hasattr(self, _FITTED_ATTR)
        if self.requires_covariance_input:
            covariance = X
            skv.validate_data(self, X, reset=True, skip_check_array=True)
            self.covariance_estimator_ = None
        else:
            if first_call and isinstance(self.covariance_estimator, str):
                raise ValueError(
                    "covariance_estimator must be 'precomputed' or a covariance estimator."
                )
            # The covariance estimator validates values and missing observations.
            skv.validate_data(self, X, reset=first_call, skip_check_array=True)
            if first_call:
                self.covariance_estimator_ = check_estimator(
                    self.covariance_estimator,
                    default=GerberCovariance(),
                    check_type=BaseCovariance,
                )
            _call_estimator(
                self.covariance_estimator_,
                method,
                X,
                y,
                routed_params=routed.covariance_estimator,
            )
            covariance = self.covariance_estimator_.covariance_  # ty: ignore[missing-union-attribute]
        self.codependence_, self.distance_ = _cov_to_distance(
            covariance, absolute=self.absolute, power=self.power
        )
        return self

    def _reset(self) -> None:
        """Reset fitted state for a new learning run."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def __sklearn_tags__(self) -> sku.Tags:
        """Declare missing-data support for the configured input."""
        tags = super().__sklearn_tags__()
        if self.requires_covariance_input:
            tags.input_tags.allow_nan = True
        elif self.covariance_estimator is not None and not isinstance(
            self.covariance_estimator, str
        ):
            tags.input_tags.allow_nan = sku.get_tags(
                self.covariance_estimator
            ).input_tags.allow_nan
        return tags


def _cov_to_distance(
    cov: ArrayLike, absolute: bool, power: float
) -> tuple[FloatArray, FloatArray]:
    """Convert a covariance matrix to codependence and distance matrices.

    NaN diagonal entries exclude assets. Only the available block is converted,
    and excluded rows and columns remain NaN in both outputs. The input is not
    modified.

    Parameters
    ----------
    cov : array-like of shape (n_assets, n_assets)
        Covariance matrix. The available block must be finite, symmetric and
        positive semidefinite, with strictly positive variances. DataFrame row
        and column labels must match in the same order.

    absolute : bool
        If True, apply the absolute transformation to the correlation matrix.

    power : float
        Exponent applied to the correlation matrix after the optional absolute
        transformation. Must be finite and strictly positive, with an integer
        value when `absolute=False`.

    Returns
    -------
    codependence : ndarray of shape (n_assets, n_assets)
        Transformed correlation matrix, with NaN rows and columns for excluded
        assets.

    distance : ndarray of shape (n_assets, n_assets)
        Distance matrix, with zero diagonal entries for available assets and
        NaN rows and columns for excluded assets. If no assets are available,
        both outputs contain only NaN.

    Raises
    ------
    ValueError
        If `cov` is not square, contains infinite values, or has mismatched
        asset labels. Also raised if the available block contains
        missing values, has nonpositive variances, or fails the correlation
        checks described below, or if `power` is invalid.

    See Also
    --------
    _corr_to_distance : Convert correlations to codependence and distances.

    Notes
    -----
    Symmetry and positive semidefiniteness are checked on the implied
    correlation matrix with absolute tolerance
    `64 * eps * n_available_assets`, where `eps` is float64 machine precision.
    Accepted roundoff is handled by symmetrizing correlations and clipping them
    to [-1, 1]. Singular covariances are accepted without eigenvalue correction.
    """
    cov, available = _validate_pairwise_matrix(cov)
    ix = np.ix_(available, available)
    available_cov = cov if available.all() else cov[ix]
    if np.any(np.diag(available_cov) <= 0):
        raise ValueError("Available covariance variances must be strictly positive.")
    corr, _ = cov_to_corr(available_cov)
    tolerance = 64 * np.finfo(np.float64).eps * len(available_cov)
    assert_is_symmetric(corr, rtol=0, atol=tolerance)
    symmetrize(corr)
    if not is_positive_semidefinite(corr, atol=tolerance):
        raise ValueError("The covariance must be positive semidefinite.")
    np.clip(corr, -1, 1, out=corr)
    if available.all():
        return _corr_to_distance(corr, absolute=absolute, power=power)
    codependence = np.full_like(cov, np.nan)
    distance = np.full_like(cov, np.nan)
    codependence[ix], distance[ix] = _corr_to_distance(
        corr, absolute=absolute, power=power
    )
    return codependence, distance
