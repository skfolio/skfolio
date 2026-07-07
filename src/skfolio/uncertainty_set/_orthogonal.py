"""Orthogonal Mu and Covariance Uncertainty Set estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers

import numpy as np
import scipy.linalg as sla
import scipy.stats as st

from skfolio.prior import ReturnDistribution
from skfolio.typing import ArrayLike
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
)
from skfolio.uncertainty_set._model import (
    CompactCovarianceUncertaintySet,
    UncertaintySet,
)
from skfolio.utils.stats import CSWeighting


class OrthogonalMuUncertaintySet(BaseMuUncertaintySet):
    r"""Expected return uncertainty set estimator for directions outside the factor span.

    This estimator builds a norm-ball uncertainty set for expected returns that are in
    the subspace orthogonal to the factor-model loading matrix, under the selected
    cross-sectional weighting metric.

    It is intended for cases where orthogonal expected returns are considered less
    reliable than spanned expected returns and is designed to reduce the tendency of
    optimizers to overallocate in these directions. Rather than shrinking the orthogonal
    expected returns in the prior, this estimator keeps the point estimate
    :math:`\hat{\mu}` unchanged and adds a portfolio-dependent worst-case penalty that
    grows with exposure to the orthogonal subspace.

    Under this uncertainty set, the worst-case expected return for a portfolio with
    weights :math:`w` is

    .. math::

        \inf_{\mu \in U_\mu} w^\top \mu
        \;=\;
        w^\top \hat{\mu}
        -
        \kappa \, \lVert L^\top w \rVert_2,

    where the low-rank geometry factor is

    .. math::

        L = G \Lambda^{1/2}.

    Here, :math:`G` is a basis for the subspace orthogonal to the factor-model span and
    :math:`\Lambda` is a positive semidefinite scaling matrix that controls the
    uncertainty assigned to each orthogonal direction.

    Equivalently, the ellipsoidal shape matrix is

    .. math::

        S_\mu = L L^\top = G \Lambda G^\top.

    If the factor model uses basket-neutral constraints, the loading matrix is first
    reduced to its effective full-rank basis before the orthogonal subspace is computed.

    Parameters
    ----------
    confidence_level : float, default=0.95
        Confidence level :math:`\beta` used to set the uncertainty size

        .. math::

            \kappa = \sqrt{\chi^2_{\mathrm{rank}}(\beta)}.

    cs_weighting : CSWeighting, default=CSWeighting.INVERSE_IDIO_VARIANCE
        Cross-sectional weighting used to define the orthogonality metric.

    uncertainty_shape : {"identity", "idio_variance"}, default="identity"
        Shape used inside the orthogonal subspace.

        * `"identity"` assigns the same uncertainty to all orthogonal directions.
        * `"idio_variance"` scales uncertainty using projected idiosyncratic variance
          in the orthogonal subspace.

    Attributes
    ----------
    uncertainty_set_ : UncertaintySet
        Fitted solver-ready uncertainty set with:

        * `uncertainty_set_.radius = \kappa`
        * `uncertainty_set_.geometry = L = G \Lambda^{1/2}`
        * `uncertainty_set_.norm = 2`

    Notes
    -----
    This estimator requires a factor model in the fitted return distribution. When used
    inside :class:`~skfolio.optimization.MeanRisk`, the `return_distribution` metadata
    is passed automatically by `fit` and `partial_fit`.

    References
    ----------
    .. [1]  "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [2] "Portfolio Optimization: Theory and Application", Chapter 14,
        Daniel P. Palomar (2025)

    .. [3] "Robust Convex Optimization", Mathematics of Operations Research,
        Ben-Tal and Nemirovski (1998).

    .. [4] "Robust Portfolio Selection Problems", Mathematics of Operations Research,
        Goldfarb and Iyengar (2003).
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        cs_weighting: CSWeighting = CSWeighting.INVERSE_IDIO_VARIANCE,
        uncertainty_shape: str = "identity",
    ):
        super().__init__(prior_estimator=None)
        self.confidence_level = confidence_level
        self.cs_weighting = cs_weighting
        self.uncertainty_shape = uncertainty_shape

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        return_distribution: ReturnDistribution | None = None,
        **fit_params,
    ) -> OrthogonalMuUncertaintySet:
        r"""Fit the orthogonal mu uncertainty set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors. The default is `None`.

        return_distribution : ReturnDistribution, optional
            The fitted return distribution from the prior estimator.
            Passed internally by :class:`~skfolio.optimization.MeanRisk`. Must contain a
            `factor_model` with `loading_matrix` and `idio_covariance`.

        **fit_params : dict
            Additional parameters (unused).

        Returns
        -------
        self : OrthogonalMuUncertaintySet
            Fitted estimator.
        """
        return self._fit(
            X,
            y,
            return_distribution=return_distribution,
            **fit_params,
        )

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        return_distribution: ReturnDistribution | None = None,
        **fit_params,
    ) -> OrthogonalMuUncertaintySet:
        r"""Update the orthogonal mu uncertainty set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors. The default is `None`.

        return_distribution : ReturnDistribution, optional
            The fitted return distribution from the prior estimator.
            Passed internally by :class:`~skfolio.optimization.MeanRisk`. Must contain a
            `factor_model` with `loading_matrix` and `idio_covariance`.

        **fit_params : dict
            Additional parameters (unused).

        Returns
        -------
        self : OrthogonalMuUncertaintySet
            Updated estimator.
        """
        return self._fit(
            X,
            y,
            return_distribution=return_distribution,
            **fit_params,
        )

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        return_distribution: ReturnDistribution | None = None,
        **fit_params,
    ) -> OrthogonalMuUncertaintySet:
        """Fit the estimator from a fitted factor-model return distribution."""
        self._validate_params()

        if return_distribution is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `return_distribution` "
                "from a fitted factor model prior. Use it with a "
                "factor-model-based prior_estimator in MeanRisk."
            )

        factor_model = return_distribution.factor_model
        if factor_model is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a factor model in "
                "return_distribution. Use a factor-model-based prior."
            )

        weights = factor_model._resolve_cs_weighting(self.cs_weighting, latest=True)
        n_assets = factor_model.loading_matrix.shape[0]
        if weights is None:
            weights_sqrt = np.ones(n_assets)
        else:
            if weights.ndim != 1 or weights.shape[0] != n_assets:
                raise ValueError("`cs_weighting` produced invalid weights.")
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("Cross-sectional weights must be finite and positive.")
            weights_sqrt = np.sqrt(weights)

        weighted_loading = (
            weights_sqrt[:, np.newaxis] * factor_model.effective_loading_matrix
        )
        factor_basis, singular_values, _ = sla.svd(
            weighted_loading, full_matrices=False, check_finite=False
        )
        if singular_values.size != 0:
            tolerance = (
                max(weighted_loading.shape) * np.finfo(float).eps * singular_values[0]
            )
            rank = int(np.sum(singular_values > tolerance))
            factor_basis = factor_basis[:, :rank]

        projector = np.eye(n_assets) - factor_basis @ factor_basis.T
        if weights is None:
            ortho = projector
        else:
            ortho = projector / weights_sqrt[:, np.newaxis]

        eigvals, eigvecs = np.linalg.eigh(ortho.T @ ortho)

        abs_tol = n_assets * np.finfo(float).eps
        rel_tol = n_assets * np.max(np.abs(eigvals)) * np.finfo(float).eps
        keep = eigvals > max(abs_tol, rel_tol)

        if not np.any(keep):
            ortho_basis = np.zeros((n_assets, 0))
        else:
            ortho_basis = ortho @ eigvecs[:, keep]
            ortho_basis, _ = np.linalg.qr(ortho_basis, mode="reduced")

        rank = ortho_basis.shape[1]

        if rank == 0:
            self.uncertainty_set_ = UncertaintySet(
                radius=0.0, geometry=np.zeros((n_assets, 1)), norm=2
            )
            return self

        # Scale each orthogonal direction.
        idio_covariance = factor_model.idio_covariance
        if self.uncertainty_shape == "identity":
            scaling_sqrt = np.eye(rank)
        elif self.uncertainty_shape == "idio_variance":
            if idio_covariance.ndim == 1:
                projected_var = ortho_basis.T @ np.diag(idio_covariance) @ ortho_basis
            else:
                projected_var = ortho_basis.T @ idio_covariance @ ortho_basis
            scaling_sqrt = sla.sqrtm(projected_var).real

        geometry = ortho_basis @ scaling_sqrt
        radius = np.sqrt(st.chi2.ppf(q=self.confidence_level, df=rank))

        self.uncertainty_set_ = UncertaintySet(radius=radius, geometry=geometry, norm=2)
        return self

    def _validate_params(self) -> None:
        """Validate estimator parameters."""
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, numbers.Real)
            or not 0 < self.confidence_level < 1
        ):
            raise ValueError("`confidence_level` must be a float in (0, 1).")

        if not isinstance(self.cs_weighting, CSWeighting):
            raise TypeError("`cs_weighting` must be a `CSWeighting`.")

        if self.uncertainty_shape not in {"identity", "idio_variance"}:
            raise ValueError(
                "`uncertainty_shape` must be one of {'identity', 'idio_variance'}."
            )


class OrthogonalCovarianceUncertaintySet(BaseCovarianceUncertaintySet):
    r"""Covariance uncertainty set estimator for directions outside the factor span.

    This estimator builds a compact covariance uncertainty set for robust portfolio
    optimization. The robust penalty assigns additional covariance uncertainty to
    portfolio directions that are in the subspace orthogonal to the factor-model
    loading matrix, under the selected cross-sectional weighting metric.

    The base covariance is assumed to have the factor structure

    .. math::

        \Sigma = B F B^\top + D

    where :math:`B` is the loading matrix, :math:`F` is the factor covariance matrix
    and :math:`D` is the idiosyncratic covariance matrix.

    Under this uncertainty set, the worst-case variance for a portfolio with weights
    :math:`w` is

    .. math::

        \sup_{\Sigma \in U} w^\top \Sigma\, w
        =
        w^\top \hat{\Sigma} w
        +
        \kappa
        \min_z
        \lVert C w - Q z \rVert_2^2.

    Here, :math:`Q` is an orthonormal basis of the weighted factor span
    :math:`\operatorname{col}(W^{1/2} B)` and :math:`C = W^{-1/2}`.

    The expression is the compact form of the quadratic penalty

    .. math::

        \kappa C^\top (I - Q Q^\top) C.

    This structured form avoids the lifted SDP formulation used for fully generic
    covariance uncertainty sets and avoids materializing the dense matrix
    :math:`C^\top (I - Q Q^\top) C`.

    If the factor model uses basket-neutral constraints, the loading matrix is first
    reduced to its effective full-rank basis before the orthogonal subspace is computed.

    Parameters
    ----------
    radius : float, default=1.0
        Penalty radius :math:`\kappa`. Controls the magnitude of the orthogonal
        covariance penalty. Must be non-negative.

    cs_weighting : CSWeighting, default=CSWeighting.INVERSE_IDIO_VARIANCE
        Cross-sectional weighting used to define the orthogonality metric.

    Attributes
    ----------
    uncertainty_set_ : CompactCovarianceUncertaintySet
        Fitted solver-ready uncertainty set containing the radius, diagonal metric
        square root and basis.

    Notes
    -----
    This estimator requires a factor model in the return distribution. When used inside
    :class:`~skfolio.optimization.MeanRisk`, the `return_distribution` metadata is
    passed automatically by `fit` and `partial_fit`.

    References
    ----------
    .. [1]  "Robustness properties of mean-variance portfolios",
        Optimization: A Journal of Mathematical Programming and Operations Research,
        Schöttle & Werner (2009).

    .. [2] "Portfolio Optimization: Theory and Application", Chapter 14,
        Daniel P. Palomar (2025)

    .. [3] "Robust Convex Optimization", Mathematics of Operations Research,
        Ben-Tal and Nemirovski (1998).

    .. [4] "Robust Portfolio Selection Problems", Mathematics of Operations Research,
        Goldfarb and Iyengar (2003).
    """

    def __init__(
        self,
        radius: float = 1.0,
        cs_weighting: CSWeighting = CSWeighting.INVERSE_IDIO_VARIANCE,
    ):
        super().__init__(prior_estimator=None)
        self.radius = radius
        self.cs_weighting = cs_weighting

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        return_distribution: ReturnDistribution | None = None,
        **fit_params,
    ) -> OrthogonalCovarianceUncertaintySet:
        r"""Fit the orthogonal covariance uncertainty set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors. The default is `None`.

        return_distribution : ReturnDistribution, optional
            The fitted return distribution from the prior estimator.
            Passed internally by :class:`~skfolio.optimization.MeanRisk`. Must contain a
            `factor_model` with `loading_matrix` and `idio_covariance`.

        **fit_params : dict
            Additional parameters (unused).

        Returns
        -------
        self : OrthogonalCovarianceUncertaintySet
            Fitted estimator.
        """
        return self._fit(
            X,
            y,
            return_distribution=return_distribution,
            **fit_params,
        )

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        return_distribution: ReturnDistribution | None = None,
        **fit_params,
    ) -> OrthogonalCovarianceUncertaintySet:
        r"""Update the orthogonal covariance uncertainty set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors. The default is `None`.

        return_distribution : ReturnDistribution, optional
            The fitted return distribution from the prior estimator.
            Passed internally by :class:`~skfolio.optimization.MeanRisk`. Must contain a
            `factor_model` with `loading_matrix` and `idio_covariance`.

        **fit_params : dict
            Additional parameters (unused).

        Returns
        -------
        self : OrthogonalCovarianceUncertaintySet
            Updated estimator.
        """
        return self._fit(
            X,
            y,
            return_distribution=return_distribution,
            **fit_params,
        )

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        return_distribution: ReturnDistribution | None = None,
        **fit_params,
    ) -> OrthogonalCovarianceUncertaintySet:
        """Fit the estimator from a fitted factor-model return distribution."""
        self._validate_params()

        if return_distribution is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `return_distribution` "
                "from a fitted factor model prior. Use it with a "
                "factor-model-based prior_estimator in MeanRisk."
            )

        factor_model = return_distribution.factor_model
        if factor_model is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a factor model in "
                "return_distribution. Use a factor-model-based prior."
            )

        weights = factor_model._resolve_cs_weighting(self.cs_weighting, latest=True)
        n_assets = factor_model.loading_matrix.shape[0]
        if weights is None:
            weights_sqrt = np.ones(n_assets)
        else:
            if weights.ndim != 1 or weights.shape[0] != n_assets:
                raise ValueError("`cs_weighting` produced invalid weights.")
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("Cross-sectional weights must be finite and positive.")
            weights_sqrt = np.sqrt(weights)

        metric_sqrt = 1.0 / weights_sqrt
        weighted_loading = (
            weights_sqrt[:, np.newaxis] * factor_model.effective_loading_matrix
        )

        # The effective loading matrix removes known structural dependencies, but the
        # selected asset universe can still make factor exposures numerically rank
        # deficient. Keep only stable singular directions so numerical noise does not
        # expand the unpenalized factor span.
        basis, singular_values, _ = sla.svd(
            weighted_loading, full_matrices=False, check_finite=False
        )
        if singular_values.size != 0:
            tolerance = (
                max(weighted_loading.shape) * np.finfo(float).eps * singular_values[0]
            )
            rank = int(np.sum(singular_values > tolerance))
            basis = basis[:, :rank]

        self.uncertainty_set_ = CompactCovarianceUncertaintySet(
            radius=self.radius, metric_sqrt=metric_sqrt, basis=basis
        )
        return self

    def _validate_params(self) -> None:
        """Validate estimator parameters."""
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, numbers.Real)
            or not np.isfinite(self.radius)
            or self.radius < 0
        ):
            raise ValueError("`radius` must be a non-negative float.")

        if not isinstance(self.cs_weighting, CSWeighting):
            raise TypeError("`cs_weighting` must be a `CSWeighting`.")
