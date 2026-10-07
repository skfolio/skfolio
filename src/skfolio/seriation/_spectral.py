"""Spectral asset seriation with online orientation alignment."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation inspired by:
# Allocation, Copyright (c) 2026, Peter Cotton, Licensed under MIT.

from __future__ import annotations

import numpy as np
import scipy.linalg as scl

from skfolio.seriation._base import BaseSeriation
from skfolio.typing import ArrayLike, FloatArray, IntArray

_FITTED_ATTR = "ordering_"


class SpectralSeriation(BaseSeriation):
    r"""Sort assets by spectral coordinates with persistent orientation.

    The spectral approach is based on Atkins, Boman and Hendrickson [1]_.
    Each `partial_fit` recomputes coordinates from the complete current
    distance matrix. Previous coordinates and ordering align the orientation
    and resolve ties on assets shared with the previous snapshot. `fit` starts
    a new ordering and discards this history.

    For recursive portfolio allocation, spectral seriation can reduce turnover
    compared with hierarchical seriation. It derives coordinates from the full
    distance matrix without discrete cluster merges. Small distance changes can
    then leave allocation groups unchanged. Preserving the previous solution
    through `partial_fit` also avoids arbitrary changes between equivalent
    spectral solutions. See :ref:`seriation_turnover` for the benefits and limits.

    Attributes
    ----------
    ordering_ : ndarray of shape (n_investable_assets,)
        Positions in the original input matrix, listed in the computed order.
        Each investable asset appears exactly once. Empty when no assets are
        investable. A single investable asset produces its original position.

    investable_mask_ : ndarray of shape (n_assets,)
        Boolean mask selecting investable assets for the current ordering.

    coordinates_ : ndarray of shape (n_assets,)
        Spectral coordinates, with NaN for non-investable assets. The vector over
        investable assets has unit Euclidean norm when at least two assets are
        investable. A single investable asset has coordinate zero.

    eigenvalue_multiplicity_ : int
        Dimension of the selected eigenspace. Zero for fewer than two investable
        assets.

    spectral_gap_ : float
        Relative separation of the two largest nonconstant eigenvalues of the
        scaled Laplacian, :math:`(\lambda_1 - \lambda_2) / \lambda_1`. Zero when
        these eigenvalues are numerically tied. NaN for fewer than three
        investable assets or when all distances are zero.

    n_features_in_ : int
        Number of assets in the full schema.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Asset names, defined when the input names are all strings.

    Notes
    -----
    For the investable distance matrix :math:`D`, form :math:`B=(D/\max(D))^2`
    and :math:`L=\mathrm{diag}(B\mathbf{1})-B`. When all distances are zero,
    set :math:`B=0`. Select the largest-eigenvalue eigenspace of :math:`L` on
    the subspace orthogonal to the constant vector. For all-zero distances,
    this is the whole nonconstant subspace.

    The selected subspace is the Fiedler eigenspace of the unnormalized
    Laplacian of the affinity :math:`A=\mathbf{1}\mathbf{1}^{\mathsf{T}}-B`.
    This also holds for repeated eigenvalues. With angular distances, it is
    the same Fiedler subspace as the dense affinity :math:`(1+\rho)/2` used by
    Peter Cotton's allocation package [2]_.

    Repeated eigenspaces use projected previous coordinates, or a deterministic
    projected anchor when history is unavailable. Ties use input positions.
    Surviving assets preserve their previous relative order within coordinate
    ties. Asset names are used for schema validation and do not affect the
    computed coordinates or ordering.

    Coordinates can change sharply when the leading eigenvalues are close, and
    sorting can change allocation groups when coordinates cross. The estimator
    does not minimize turnover or guarantee continuous weights.

    A small `spectral_gap_` indicates weak separation of the leading direction.
    A repeated leading eigenvalue gives a zero gap even if its eigenspace is
    separated from the remaining directions. Small perturbations can split that
    eigenvalue and change the selected coordinates. The gap is not a turnover
    prediction.

    Dense decomposition costs :math:`O(k^3)` time and :math:`O(k^2)` working memory
    for :math:`k` investable assets. Only :math:`O(n_{assets})` orientation history
    persists.

    References
    ----------
    .. [1] "A Spectral Algorithm for Seriation and the Consecutive Ones Problem".
        Jonathan E. Atkins, Erik G. Boman and Bruce Hendrickson,
        SIAM Journal on Computing (1998).

    .. [2] "allocation: Streaming online portfolio construction".
        Peter Cotton (2026). https://github.com/microprediction/allocation
    """

    coordinates_: FloatArray
    eigenvalue_multiplicity_: int
    spectral_gap_: float

    def fit(self, X: ArrayLike, y: None = None) -> SpectralSeriation:
        """Start a new ordering from a distance snapshot.

        Parameters
        ----------
        X : array-like of shape (n_assets, n_assets)
            Distance snapshot. NaN diagonal entries mark non-investable assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SpectralSeriation
            Fitted estimator.
        """
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X: ArrayLike, y: None = None) -> SpectralSeriation:
        """Order a replacement snapshot aligned to the previous ordering.

        Recompute spectral coordinates from the complete current distance
        matrix. Use `fit` to change the full asset universe or discard history.

        Parameters
        ----------
        X : array-like of shape (n_assets, n_assets)
            Complete distance matrix with the same assets in the same row and
            column order as previous calls. NaN diagonal entries mark assets
            that are currently non-investable.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SpectralSeriation
            Updated estimator.
        """
        first_call = not hasattr(self, _FITTED_ATTR)
        distance, investable_mask = self._validate_distance(X, reset=first_call)
        indices = np.flatnonzero(investable_mask)
        n_investable = len(indices)
        vector = np.full(self.n_features_in_, np.nan)
        ordering = indices
        multiplicity = 0
        gap = np.nan
        if n_investable == 1:
            vector[indices] = 0
        elif n_investable > 1:
            eps = np.finfo(np.float64).eps
            tolerance = 64 * eps * n_investable
            scale = np.max(distance)
            if scale > 0:
                # Reuse the validated distance copy to build the Laplacian.
                laplacian = distance
                laplacian /= scale
                np.square(laplacian, out=laplacian)
                degrees = laplacian.sum(axis=1)
                laplacian *= -1
                np.fill_diagonal(laplacian, degrees)
                # The positive leading eigenspace is orthogonal to the constant
                # vector, whose eigenvalue is zero.
                eigenvalues, eigenvectors = scl.eigh(laplacian)
                selected = eigenvalues[-1] - eigenvalues <= (
                    tolerance * abs(eigenvalues[-1])
                )
                eigenspace = eigenvectors[:, selected]
                if n_investable > 2:
                    gap = (
                        0.0
                        if selected[-2]
                        else float(
                            (eigenvalues[-1] - eigenvalues[-2]) / eigenvalues[-1]
                        )
                    )
            else:
                # A zero Laplacian selects the whole nonconstant subspace.
                eigenspace = scl.helmert(n_investable, full=False).T
            multiplicity = eigenspace.shape[1]

            coordinate = np.zeros(n_investable)
            if not first_call:
                previous = self.coordinates_[indices]
                overlap = np.isfinite(previous)
                if overlap.any():
                    coordinate[overlap] = previous[overlap] - previous[overlap].mean()
                    reference_norm = np.linalg.norm(coordinate)
                    if reference_norm > 0:
                        # Project with U @ (U.T @ x) without forming U @ U.T
                        coordinate = eigenspace @ (
                            eigenspace.T @ (coordinate / reference_norm)
                        )
            projection_norm = np.linalg.norm(coordinate)
            if projection_norm >= np.sqrt(eps):
                coordinate /= projection_norm
            elif multiplicity == 1:
                coordinate = eigenspace[:, 0]
                magnitude = np.abs(coordinate)
                anchors = magnitude.max() - magnitude <= tolerance
                anchor = np.flatnonzero(anchors)[0]
                coordinate *= 1 if coordinate[anchor] >= 0 else -1
            else:
                projector_diagonal = np.sum(eigenspace**2, axis=1)
                anchors = projector_diagonal.max() - projector_diagonal <= tolerance
                anchor = np.flatnonzero(anchors)[0]
                coordinate = eigenspace @ eigenspace[anchor]
                coordinate /= np.linalg.norm(coordinate)

            previous_order = None if first_call else self.ordering_
            ordering = indices[
                _order_coordinates(coordinate, indices, previous_order, tolerance)
            ]
            vector[indices] = coordinate
        self.investable_mask_ = investable_mask
        self.ordering_ = ordering
        self.coordinates_ = vector
        self.eigenvalue_multiplicity_ = multiplicity
        self.spectral_gap_ = gap
        return self

    def _reset(self) -> None:
        """Reset the ordering and initialize history on the next update."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)


def _order_coordinates(
    coordinate: FloatArray,
    indices: IntArray,
    previous_order: IntArray | None,
    tolerance: float,
) -> IntArray:
    """Sort coordinates and resolve numerical ties using input positions and history.

    Sort in ascending coordinate order, then form each tie group from values
    within `tolerance` of its first coordinate. Within a group, assets present
    in the previous ordering come first in their previous relative order.
    New and returning assets follow in input order.

    Parameters
    ----------
    coordinate : ndarray of shape (n_investable_assets,)
        Spectral coordinates in the investable assets' input order.

    indices : ndarray of shape (n_investable_assets,)
        Positions of investable assets in the full input matrix, aligned with
        `coordinate`.

    previous_order : ndarray of shape (n_previous_assets,) or None
        Previous ordering expressed as positions in the full input matrix.
        None when starting without history.

    tolerance : float
        Maximum coordinate difference from the first element of a tie group.

    Returns
    -------
    order : ndarray of shape (n_investable_assets,)
        Permutation of positions within `coordinate`. Use `indices[order]` to
        obtain the ordering in the full input matrix.
    """
    ranks = np.arange(len(indices))
    if previous_order is not None:
        previous_ranks = {asset: rank for rank, asset in enumerate(previous_order)}
        ranks = np.array(
            [
                previous_ranks.get(asset, len(previous_order) + i)
                for i, asset in enumerate(indices)
            ]
        )
    order = np.argsort(coordinate, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and coordinate[order[end]] - coordinate[order[start]] <= tolerance
        ):
            end += 1
        group = order[start:end]
        order[start:end] = group[np.argsort(ranks[group], kind="stable")]
        start = end
    return order
