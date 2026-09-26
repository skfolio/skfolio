"""Uncertainty Set dataclasses."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skfolio.typing import FloatArray

__all__ = ["CompactCovarianceUncertaintySet", "UncertaintySet"]


# frozen=True with eq=False will lead to an id-based hashing which is needed for
# caching CVX models in Optimization without impacting performance
@dataclass(frozen=True, eq=False)
class UncertaintySet:
    r"""Norm-ball uncertainty set.

    A norm-ball uncertainty set represents deviations of a parameter vector :math:`z`
    from an estimate :math:`\hat{z}` as

    .. math::

        z - \hat{z} = L u,
        \quad
        \lVert u \rVert_p \le \kappa.

    Equivalently, the uncertainty set is

    .. math::

        \mathcal{U}
        =
        \left\{
            \hat{z} + L u :
            \lVert u \rVert_p \le \kappa
        \right\}.

    All common uncertainty sets, including ellipsoidal, box and diamond sets, can be
    represented by choosing the radius :math:`\kappa`, the norm :math:`p` and the
    linear map :math:`L`.

    The radius :math:`\kappa` controls the size of the normalized uncertainty ball.
    The norm :math:`p` selects its canonical shape:

    * :math:`p = 2`: Euclidean ball
    * :math:`p = \infty`: Box
    * :math:`p = 1`: Diamond / Cross-polytope

    The `geometry` parameter stores the linear geometry map :math:`L`. It maps the
    normalized ball into parameter space by scaling and mixing uncertainty directions to
    form deviations of :math:`z` from :math:`\hat{z}`. The estimator precomputes this
    map so the optimizer can work directly with :math:`L`, which may be low-rank
    :math:`(n \times r)` with :math:`r \ll n`.

    For a linear exposure vector :math:`e`, the worst-case deviation over
    :math:`\mathcal{U}` is

    .. math::

        \sup_{z \in \mathcal{U}}
        e^\top(z - \hat{z})
        =
        \kappa \,
        \lVert L^\top e \rVert_q,

    where :math:`q` is the dual norm of :math:`p`.

    Downstream optimizers use this support-function as the uncertainty penalty.
    For expected-return uncertainty, :math:`z` is :math:`\mu` and :math:`e` is the
    portfolio weight vector. For covariance uncertainty, :math:`z` is
    :math:`\operatorname{vec}(\Sigma)` (the vector obtained by stacking the columns of
    :math:`\Sigma`) and :math:`e` has the same vectorized shape.

    Standard choices are:

    * **Ellipsoidal set:** Use `norm=2`. For a full-rank shape matrix :math:`S`, set
      `geometry` to a square-root factor :math:`L` satisfying :math:`S = L L^\top`. This
      gives :math:`(z - \hat{z})^\top S^{-1} (z - \hat{z}) \le \kappa^2`. For a low-rank
      representation :math:`S = G \Lambda G^\top`, set `geometry` to :math:`G \Lambda^{1/2}`.

    * **Box set:** Use `norm=np.inf`. With axis widths :math:`\delta_i`, set `geometry`
      to :math:`\operatorname{diag}(\delta)`. This gives
      :math:`|z_i - \hat{z}_i| \le \kappa \delta_i` for each coordinate and the dual
      norm is :math:`1`.

    * **Diamond set:** Use `norm=1`. With axis scales :math:`\delta_i`, set `geometry`
      to :math:`\operatorname{diag}(\delta)`. This gives
      :math:`\sum_i |z_i - \hat{z}_i| / \delta_i \le \kappa` and the dual norm is
      :math:`\infty`.

    Parameters
    ----------
    radius : float
        Radius :math:`\kappa` of the normalized uncertainty ball
        :math:`\lVert u \rVert_p \le \kappa`.

    geometry : ndarray of shape (n_parameters, n_uncertainties)
        Linear geometry map :math:`L` mapping normalized uncertainty coordinates into
        deviations from :math:`\hat{z}`:

        .. math::

            z - \hat{z} = L u.

        For ellipsoidal uncertainty with shape matrix :math:`S`, `geometry` is a
        square-root factor :math:`L` satisfying :math:`S = L L^\top`.

        For axis-aligned box or diamond uncertainty with widths `delta`, `geometry` is
        typically :math:`\operatorname{diag}(\delta)`.

    norm : float or int, default=2
        Norm :math:`p` defining the normalized uncertainty ball. Must be greater than or
        equal to 1. Common choices are `2` for ellipsoidal uncertainty, `np.inf` for box
        uncertainty, and `1` for diamond uncertainty.

    Attributes
    ----------
    dual_norm : float
        Dual norm :math:`q` associated with `norm`. This is the norm used in the
        support-function penalty :math:`\kappa \lVert L^\top e \rVert_q`.
    """

    radius: float
    geometry: FloatArray
    norm: float | int

    def __post_init__(self) -> None:
        """Validate the norm-ball parameters."""
        geometry = np.asarray(self.geometry)
        if geometry.ndim != 2:
            raise ValueError("`geometry` must be a 2D array.")

        radius = float(self.radius)
        if radius < 0:
            raise ValueError("`radius` must be non-negative.")

        norm = float(self.norm)
        if norm < 1.0:
            raise ValueError("`norm` must be greater than or equal to 1.")

        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "norm", norm)

    @property
    def dual_norm(self) -> float:
        """Dual norm associated with `norm`."""
        if self.norm == 1.0:
            return np.inf
        if self.norm == np.inf:
            return 1.0
        return self.norm / (self.norm - 1)


@dataclass(frozen=True, eq=False)
class CompactCovarianceUncertaintySet:
    r"""Compact representation of a quadratic covariance uncertainty penalty.

    This object stores the data needed to evaluate a worst-case variance penalty in
    reduced projection form, without materializing the equivalent dense positive
    semidefinite matrix.

    Let :math:`C` be a diagonal metric square root and let :math:`Q` be an orthonormal
    basis. For portfolio weights :math:`w`, the optimizer evaluates

    .. math::

        \kappa \min_z \lVert C w - Q z \rVert_2^2.

    This is equivalent to adding the following positive semidefinite matrix to the
    quadratic variance term:

    .. math::

        \kappa C^\top (I - Q Q^\top) C.

    The compact representation avoids building this dense matrix. The optimizer only
    needs the diagonal entries of :math:`C` and the basis :math:`Q`.

    Parameters
    ----------
    radius : float
        Non-negative multiplier :math:`\kappa` applied to the quadratic covariance
        penalty.

    metric_sqrt : ndarray of shape (n_assets,)
        Diagonal of the metric square root :math:`C`.

    basis : ndarray of shape (n_assets, rank)
        Orthonormal basis :math:`Q` of the subspace projected out by the quadratic
        penalty.

    Attributes
    ----------
    radius : float
        Non-negative multiplier :math:`\kappa`.

    metric_sqrt : ndarray of shape (n_assets,)
        Diagonal of the metric square root :math:`C`.

    basis : ndarray of shape (n_assets, rank)
        Orthonormal basis :math:`Q`.
    """

    radius: float
    metric_sqrt: FloatArray
    basis: FloatArray

    def __post_init__(self) -> None:
        """Validate parameters."""
        radius = float(self.radius)
        if radius < 0:
            raise ValueError("`radius` must be non-negative.")

        metric_sqrt = np.asarray(self.metric_sqrt, dtype=float)
        if metric_sqrt.ndim != 1:
            raise ValueError("`metric_sqrt` must be a 1D array.")
        if np.any(~np.isfinite(metric_sqrt)) or np.any(metric_sqrt < 0):
            raise ValueError("`metric_sqrt` must contain finite non-negative values.")

        basis = np.asarray(self.basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError("`basis` must be a 2D array.")
        if basis.shape[0] != metric_sqrt.shape[0]:
            raise ValueError(
                "`basis` and `metric_sqrt` must have the same number of assets."
            )
        if np.any(~np.isfinite(basis)):
            raise ValueError("`basis` must contain finite values.")

        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "metric_sqrt", metric_sqrt)
        object.__setattr__(self, "basis", basis)
