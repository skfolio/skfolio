"""Compact family-constraint basis for constrained factor families."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from skfolio.typing import FloatArray, IntArray, StrArray
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_positive_integer,
)

__all__ = [
    "FamilyConstraint",
    "FamilyConstraintBasis",
    "compute_family_constraint_basis",
]


@dataclass(frozen=True)
class FamilyConstraintBasis:
    r"""Compact representation of family zero-sum constraints.

    This class represents the time-varying change of basis between the full factor
    space :math:`\mathbb{R}^K` and a reduced full-rank factor space
    :math:`\mathbb{R}^{K_{\mathrm{red}}}` obtained by imposing benchmark-weighted
    zero-sum constraints within selected factor families.

    The reduced basis is ordered as the full factor basis with dropped factors removed.
    For each constrained family, exactly one redundant factor is dropped. All other
    factors, including unconstrained passthrough factors, retain their original
    full-order relative position.

    For each constrained family, the reduced basis spans the subspace where the family
    factor returns satisfy:

    .. math::

        c_t^\top f_{\mathrm{family}}(t) = 0.

    For each observation :math:`t`, there is a conceptual basis matrix
    :math:`R_t \in \mathbb{R}^{K \times K_{\mathrm{red}}}` such that:

    .. math::

        B_{\mathrm{red}}(t) = B_{\mathrm{full}}(t) R_t.

    The dense time-varying basis tensor :math:`R_t` is not stored. Instead, the class
    stores family metadata and time-varying constraint ratios :math:`c_t(j) / c_t(k)`,
    where :math:`j` is a retained factor and :math:`k` is the dropped factor in the
    family.

    This yields an :math:`O(T \cdot C)` representation, where :math:`C` is the total
    number of retained constrained factors. Exposure, coordinate, and factor-return
    transforms are applied family-wise without materializing :math:`R_t`. Covariance
    expansion uses compact dropped-factor reconstruction weights rather than a dense
    time-varying basis tensor.

    The basis is useful for:

    * Cross-sectional regression: regress on :math:`B_{\mathrm{red}}(t)` to estimate
      factor returns in a full-rank space without explicit linear equality constraints.
    * Factor covariance estimation: estimate moments in the reduced basis, where the
      constrained families are full-rank.
    * Optimization and Cholesky decomposition: work in the reduced basis to avoid
      singular factor covariance matrices.
    * Attribution and reporting: reconstruct full named factor returns via
      :math:`f_{\mathrm{full}}(t) = R_t g(t)`. Retained factors keep their original
      names and economic meaning; only the dropped factor is implicit and recovered
      from the zero-sum relation.

    Parameters
    ----------
    n_full_factors : int
        Full factor dimension :math:`K`.

    family_constraints : tuple of FamilyConstraint
        One entry per constrained family.

    constraint_ratios : ndarray of shape (n_observations, n_retained_constrained)
        Time-varying ratios :math:`c_t(j) / c_t(k)` for all constrained families,
        concatenated in family order. Within each family, columns follow
        :attr:`FamilyConstraint.retained_local_indices`.
    """

    n_full_factors: int
    family_constraints: tuple[FamilyConstraint, ...]
    constraint_ratios: FloatArray

    def __post_init__(self):
        _validate_positive_integer(self.n_full_factors, "n_full_factors")

        family_constraints = self.family_constraints
        if not family_constraints:
            raise ValueError("family_constraints must contain at least one family.")
        if not all(
            isinstance(family, FamilyConstraint) for family in family_constraints
        ):
            raise ValueError("family_constraints must contain only FamilyConstraint.")

        constrained_full_indices = _stack_family_full_indices(family_constraints)
        if np.any(
            (constrained_full_indices < 0)
            | (constrained_full_indices >= self.n_full_factors)
        ):
            raise ValueError("Family factor indices must lie in [0, n_full_factors).")
        if np.unique(constrained_full_indices).size != constrained_full_indices.size:
            raise ValueError("Family constraints must be disjoint.")

        constraint_ratios = self.constraint_ratios
        if constraint_ratios.ndim != 2:
            raise ValueError("constraint_ratios must be a 2D array.")

        expected_width = sum(family.family_size - 1 for family in family_constraints)
        if constraint_ratios.shape[1] != expected_width:
            raise ValueError(
                f"constraint_ratios must have shape (n_observations, {expected_width})."
            )
        if not np.all(np.isfinite(constraint_ratios)):
            raise ValueError("constraint_ratios must contain only finite values.")

    def __getitem__(self, observation_key) -> FamilyConstraintBasis:
        """Return a basis sliced along the observation axis."""
        if isinstance(observation_key, tuple):
            raise TypeError(
                "FamilyConstraintBasis only supports indexing along the "
                "observation axis."
            )

        sliced_ratios = self.constraint_ratios[observation_key]
        if sliced_ratios.ndim == 1:
            sliced_ratios = sliced_ratios[np.newaxis, :]

        return FamilyConstraintBasis(
            n_full_factors=self.n_full_factors,
            family_constraints=self.family_constraints,
            constraint_ratios=sliced_ratios,
        )

    def with_constraint_ratios(
        self,
        constraint_ratios: FloatArray,
    ) -> FamilyConstraintBasis:
        """Return this basis with replaced time-varying constraint ratios."""
        return FamilyConstraintBasis(
            n_full_factors=self.n_full_factors,
            family_constraints=self.family_constraints,
            constraint_ratios=constraint_ratios,
        )

    def append_passthrough_factors(self, n_new_factors: int) -> FamilyConstraintBasis:
        """Embed this basis in a larger full factor space.

        The new factors are appended after the existing full factors and are
        unconstrained passthrough factors.
        """
        _validate_non_negative_integer(n_new_factors, "n_new_factors")
        n_new_factors = int(n_new_factors)
        if n_new_factors == 0:
            return self

        return FamilyConstraintBasis(
            n_full_factors=self.n_full_factors + n_new_factors,
            family_constraints=self.family_constraints,
            constraint_ratios=self.constraint_ratios,
        )

    @property
    def n_observations(self) -> int:
        """Number of observations in the time axis."""
        return self.constraint_ratios.shape[0]

    @property
    def n_constraints(self) -> int:
        """Number of constrained families, which equals the dropped factor count."""
        return len(self.family_constraints)

    @property
    def n_reduced_factors(self) -> int:
        r"""Reduced factor dimension :math:`K_{\mathrm{red}}`."""
        return self.n_full_factors - self.n_constraints

    @property
    def dropped_full_indices(self) -> IntArray:
        """Full-basis indices of dropped factors."""
        return np.array(
            [family.dropped_full_index for family in self.family_constraints],
            dtype=int,
        )

    @cached_property
    def retained_full_indices(self) -> IntArray:
        """Full-basis indices retained in the reduced basis, in full-factor order."""
        keep = np.ones(self.n_full_factors, dtype=bool)
        keep[self.dropped_full_indices] = False
        return np.nonzero(keep)[0]

    @cached_property
    def full_to_reduced_index(self) -> IntArray:
        """Map each full-basis index to its reduced-basis index, or -1 if dropped."""
        out = np.full(self.n_full_factors, -1, dtype=int)
        out[self.retained_full_indices] = np.arange(self.n_reduced_factors)
        return out

    @cached_property
    def _family_reduced_indices(self) -> list[IntArray]:
        """Reduced-basis indices of each constrained family's retained factors."""
        return [
            self.full_to_reduced_index[family.retained_full_indices]
            for family in self.family_constraints
        ]

    @cached_property
    def _family_ratio_slices(self) -> list[slice]:
        """One constraint-ratio column slice per constrained family."""
        slices = []
        offset = 0
        for family in self.family_constraints:
            width = family.family_size - 1
            slices.append(slice(offset, offset + width))
            offset += width
        return slices

    def _iter_families(self) -> Iterator[tuple[FamilyConstraint, IntArray, slice]]:
        """Iterate over `(family, reduced_indices, ratio_slice)`."""
        return zip(
            self.family_constraints,
            self._family_reduced_indices,
            self._family_ratio_slices,
            strict=True,
        )

    def _validate_observation_count(self, n_observations: int, name: str) -> None:
        """Validate that a time-varying input matches the basis length."""
        if n_observations != self.n_observations:
            raise ValueError(
                f"`{name}` has {n_observations} observations but the basis "
                f"has {self.n_observations}. Slice the FamilyConstraintBasis "
                "first to align both time axes."
            )

    def reduced_factor_names(self, full_factor_names: StrArray) -> StrArray:
        """Return factor names ordered in the reduced basis.

        The reduced basis follows full factor order with dropped constrained-family
        factors removed.
        """
        full_factor_names = np.asarray(full_factor_names)
        if full_factor_names.ndim != 1:
            raise ValueError(
                "full_factor_names must be a 1D array of shape (n_full_factors,)."
            )
        _validate_factor_dimension(
            len(full_factor_names), self.n_full_factors, "full_factor_names"
        )
        return full_factor_names[self.retained_full_indices]

    def reduce_exposures(self, exposures: FloatArray) -> FloatArray:
        r"""Map full-basis exposures to the reduced basis.

        For each constrained family with dropped factor :math:`k` and retained factors
        :math:`j \neq k`:

        .. math::

            X^{\mathrm{red}}_t(:, j)
            = X_t(:, j) - \frac{c_t(j)}{c_t(k)} X_t(:, k).

        Factors outside the constrained families are copied unchanged in their
        full-order reduced-basis positions.

        Parameters
        ----------
        exposures : ndarray of shape (n_observations, n_assets, n_full_factors)
            Full-basis exposures. The time axis must match the basis.

        Returns
        -------
        exposures_reduced : ndarray of shape (n_observations, n_assets, n_reduced_factors)
            Reduced-basis exposures with the same observation and asset axes.
        """
        exposures = np.asarray(exposures)
        if exposures.ndim != 3:
            raise ValueError(
                "exposures must be a 3D array of shape "
                "(n_observations, n_assets, n_full_factors)."
            )

        n_observations = exposures.shape[0]
        self._validate_observation_count(n_observations, "exposures")
        _validate_factor_dimension(exposures.shape[2], self.n_full_factors, "exposures")

        exposures_reduced = exposures[:, :, self.retained_full_indices].astype(
            float, copy=True
        )

        for family, reduced_indices, ratio_slice in self._iter_families():
            family_ratios = self.constraint_ratios[:, ratio_slice]
            retained = exposures[:, :, family.retained_full_indices]
            dropped = exposures[:, :, family.dropped_full_slice]
            exposures_reduced[:, :, reduced_indices] = (
                retained - dropped * family_ratios[:, np.newaxis, :]
            )

        return exposures_reduced

    def reduce_loading_matrix(
        self, loading_matrix: FloatArray, observation_index: int = -1
    ) -> FloatArray:
        r"""Map a point-in-time full loading matrix to the reduced basis.

        This is the single-date specialization of :meth:`reduce_exposures`. It applies
        the constraint ratios at `observation_index` directly to a 2D loading matrix.

        Parameters
        ----------
        loading_matrix : ndarray of shape (n_assets, n_full_factors)
            Full-basis point-in-time loading matrix.

        observation_index : int, default=-1
            Index into the time axis selecting which observation's ratios to use.

        Returns
        -------
        loading_reduced : ndarray of shape (n_assets, n_reduced_factors)
            Reduced-basis loading matrix at the selected observation.
        """
        loading_matrix = np.asarray(loading_matrix)
        if loading_matrix.ndim != 2:
            raise ValueError(
                "loading_matrix must be a 2D array of shape (n_assets, n_full_factors)."
            )
        _validate_factor_dimension(
            loading_matrix.shape[1], self.n_full_factors, "loading_matrix"
        )

        loading_reduced = loading_matrix[:, self.retained_full_indices].astype(
            float, copy=True
        )

        for family, reduced_indices, ratio_slice in self._iter_families():
            family_ratios = self.constraint_ratios[observation_index, ratio_slice]
            retained = loading_matrix[:, family.retained_full_indices]
            dropped = loading_matrix[:, family.dropped_full_slice]
            loading_reduced[:, reduced_indices] = (
                retained - dropped * family_ratios[np.newaxis, :]
            )

        return loading_reduced

    def reduce_factor_returns(self, factor_returns: FloatArray) -> FloatArray:
        """Reduce full-basis factor returns to reduced-basis factor returns.

        Factor returns are coordinates in factor-return space. To express full factor
        returns in the reduced basis, this method only drops the redundant
        constrained-family factors and keeps the retained full returns in reduced
        order. No `constraint_ratios` are applied.

        Parameters
        ----------
        factor_returns : ndarray of shape (n_observations, n_full_factors) or (n_full_factors,)
            Full-basis factor returns.

        Returns
        -------
        factor_returns_reduced : ndarray of shape (n_observations, n_reduced_factors) or (n_reduced_factors,)
            Retained factor-return columns ordered in the reduced basis.
        """
        factor_returns = np.asarray(factor_returns, dtype=float)
        if factor_returns.ndim not in (1, 2):
            raise ValueError("factor_returns must be a 1D or 2D array.")

        squeeze = factor_returns.ndim == 1
        if squeeze:
            factor_returns = factor_returns[np.newaxis, :]

        _validate_factor_dimension(
            factor_returns.shape[1], self.n_full_factors, "factor_returns"
        )

        reduced = factor_returns[:, self.retained_full_indices]
        return reduced[0] if squeeze else reduced

    def reduce_factor_mu(self, factor_mu: FloatArray) -> FloatArray:
        """Reduce a full-basis factor expected-returns vector to the retained factors.

        The reduced expected returns are the retained components of the full expected
        returns, kept in reduced-basis order. No `constraint_ratios` are applied. This
        is the inverse of :meth:`expand_factor_mu`.

        Parameters
        ----------
        factor_mu : ndarray of shape (n_full_factors,)
            Full-basis vector of expected factor returns.

        Returns
        -------
        reduced_factor_mu : ndarray of shape (n_reduced_factors,)
            Retained expected factor returns ordered in the reduced basis.
        """
        factor_mu = np.asarray(factor_mu, dtype=float)
        if factor_mu.ndim != 1:
            raise ValueError("factor_mu must be a 1D array.")
        _validate_factor_dimension(len(factor_mu), self.n_full_factors, "factor_mu")
        return factor_mu[self.retained_full_indices]

    def reduce_factor_covariance(self, factor_covariance: FloatArray) -> FloatArray:
        r"""Reduce a full-basis factor covariance to the retained full-rank block.

        The reduced factor returns are the retained components of the full factor
        returns, so the reduced covariance is the retained-retained block:

        .. math::

            \Sigma^{\mathrm{red}} = \Sigma^{\mathrm{full}}[\mathcal{R}, \mathcal{R}],

        where :math:`\mathcal{R}` are the retained full-basis indices. No
        `constraint_ratios` are applied. This is the inverse of
        :meth:`expand_factor_covariance` and the covariance analog of
        :meth:`reduce_factor_returns`.

        Parameters
        ----------
        factor_covariance : ndarray of shape (n_full_factors, n_full_factors) or (n_observations, n_full_factors, n_full_factors)
            Full-basis factor covariance.

        Returns
        -------
        reduced_factor_covariance : ndarray of shape (n_reduced_factors, n_reduced_factors) or (n_observations, n_reduced_factors, n_reduced_factors)
            Retained full-rank covariance block.
        """
        factor_covariance = np.asarray(factor_covariance, dtype=float)
        if factor_covariance.ndim not in (2, 3):
            raise ValueError("factor_covariance must be a 2D or 3D array.")
        if factor_covariance.shape[-2:] != (
            self.n_full_factors,
            self.n_full_factors,
        ):
            raise ValueError(
                "factor_covariance must have trailing shape "
                f"({self.n_full_factors}, {self.n_full_factors})."
            )

        retained = self.retained_full_indices
        if factor_covariance.ndim == 2:
            return factor_covariance[np.ix_(retained, retained)]
        return factor_covariance[:, retained[:, np.newaxis], retained]

    def expand_factor_returns(self, reduced_factor_returns: FloatArray) -> FloatArray:
        r"""Reconstruct full-basis factor returns from reduced-basis returns.

        For each constrained family, retained factor returns are copied through and the
        dropped factor's return is recovered from the zero-sum relation:

        .. math::

            f^{\mathrm{full}}_j(t) = g_j(t), \quad j \neq k, \qquad
            f^{\mathrm{full}}_k(t)
            = -\sum_{j \neq k} \frac{c_t(j)}{c_t(k)} g_j(t).

        For 1D input, the ratios from the last observation are used.

        Parameters
        ----------
        reduced_factor_returns : ndarray of shape (n_observations, n_reduced_factors) or (n_reduced_factors,)
            Reduced-basis factor returns. For 2D input, the observation axis must match
            the basis.

        Returns
        -------
        factor_returns : ndarray of shape (n_observations, n_full_factors) or (n_full_factors,)
            Full-basis factor returns reconstructed from the reduced basis.
        """
        reduced_factor_returns = np.asarray(reduced_factor_returns, dtype=float)
        if reduced_factor_returns.ndim not in (1, 2):
            raise ValueError("reduced_factor_returns must be a 1D or 2D array.")

        squeeze = reduced_factor_returns.ndim == 1
        if squeeze:
            reduced_factor_returns = reduced_factor_returns[np.newaxis, :]

        n_observations = reduced_factor_returns.shape[0]
        _validate_factor_dimension(
            reduced_factor_returns.shape[1],
            self.n_reduced_factors,
            "reduced_factor_returns",
        )
        if not squeeze:
            self._validate_observation_count(n_observations, "reduced_factor_returns")

        factor_returns = np.empty((n_observations, self.n_full_factors), dtype=float)
        factor_returns[:, self.retained_full_indices] = reduced_factor_returns

        for family, reduced_indices, ratio_slice in self._iter_families():
            family_ratios = (
                self.constraint_ratios[-1:, ratio_slice]
                if squeeze
                else self.constraint_ratios[:, ratio_slice]
            )
            retained_returns = reduced_factor_returns[:, reduced_indices]
            factor_returns[:, family.dropped_full_index] = -np.sum(
                family_ratios * retained_returns,
                axis=-1,
            )

        return factor_returns[0] if squeeze else factor_returns

    def expand_factor_mu(
        self, reduced_factor_mu: FloatArray, observation_index: int = -1
    ) -> FloatArray:
        """Reconstruct a full-basis vector of expected factor returns from reduced-basis
        expected returns.

        Parameters
        ----------
        reduced_factor_mu : ndarray of shape (n_reduced_factors,)
            Reduced-basis vector of expected factor returns at the selected observation.

        observation_index : int, default=-1
            Index into the time axis selecting which observation's ratios to use.

        Returns
        -------
        factor_mu : ndarray of shape (n_full_factors,)
            Full-basis vector of expected factor returns reconstructed from the reduced
            basis.
        """
        reduced_factor_mu = np.asarray(reduced_factor_mu, dtype=float)
        if reduced_factor_mu.ndim != 1:
            raise ValueError("reduced_factor_mu must be a 1D array.")
        _validate_factor_dimension(
            len(reduced_factor_mu),
            self.n_reduced_factors,
            "reduced_factor_mu",
        )

        factor_mu = np.empty(self.n_full_factors, dtype=float)
        factor_mu[self.retained_full_indices] = reduced_factor_mu

        for family, reduced_indices, ratio_slice in self._iter_families():
            family_ratios = self.constraint_ratios[observation_index, ratio_slice]
            retained_mu = reduced_factor_mu[reduced_indices]
            factor_mu[family.dropped_full_index] = -family_ratios @ retained_mu

        return factor_mu

    def expand_factor_covariance(
        self, reduced_factor_covariance: FloatArray, observation_index: int = -1
    ) -> FloatArray:
        r"""Reconstruct full-basis factor covariance from reduced-basis covariance.

        Applies:

        .. math::

            \Sigma^{\mathrm{full}} = R_t \Sigma^{\mathrm{red}} R_t^\top.

        The dense matrix :math:`R_t` is not materialized. The implementation fills the
        retained-retained block directly and reconstructs dropped rows and columns using
        compact dropped-factor weights. For 3D input, all observations are expanded with
        vectorized batched matrix multiplications.

        Parameters
        ----------
        reduced_factor_covariance : ndarray of shape (n_reduced_factors, n_reduced_factors) or (n_observations, n_reduced_factors, n_reduced_factors)
            Reduced-basis factor covariance. When 3D, each slice along the first axis is
            mapped back using the corresponding observation's ratios, and the
            observation axis must match the basis.

        observation_index : int, default=-1
            Index into the time axis of the ratios. Only used when
            `reduced_factor_covariance` is 2D.

        Returns
        -------
        factor_covariance : ndarray of shape (n_full_factors, n_full_factors) or (n_observations, n_full_factors, n_full_factors)
            Full-basis factor covariance.
        """
        reduced_factor_covariance = np.asarray(reduced_factor_covariance, dtype=float)
        if reduced_factor_covariance.ndim not in (2, 3):
            raise ValueError("reduced_factor_covariance must be a 2D or 3D array.")
        if reduced_factor_covariance.shape[-2:] != (
            self.n_reduced_factors,
            self.n_reduced_factors,
        ):
            raise ValueError(
                "reduced_factor_covariance must have trailing shape "
                f"({self.n_reduced_factors}, {self.n_reduced_factors})."
            )

        if reduced_factor_covariance.ndim == 2:
            dropped_weights = self._dropped_factor_reduced_weights(
                observation_index=observation_index
            )
            full_covariance = np.empty(
                (self.n_full_factors, self.n_full_factors), dtype=float
            )

            retained = self.retained_full_indices
            dropped = self.dropped_full_indices
            dropped_retained = dropped_weights @ reduced_factor_covariance
            retained_dropped = reduced_factor_covariance @ dropped_weights.T
            dropped_dropped = dropped_retained @ dropped_weights.T

            full_covariance[np.ix_(retained, retained)] = reduced_factor_covariance
            full_covariance[np.ix_(dropped, retained)] = dropped_retained
            full_covariance[np.ix_(retained, dropped)] = retained_dropped
            full_covariance[np.ix_(dropped, dropped)] = dropped_dropped
            return full_covariance

        self._validate_observation_count(
            reduced_factor_covariance.shape[0], "reduced_factor_covariance"
        )
        dropped_weights = self._dropped_factor_reduced_weights()
        full_covariance = np.empty(
            (
                reduced_factor_covariance.shape[0],
                self.n_full_factors,
                self.n_full_factors,
            ),
            dtype=float,
        )

        retained = self.retained_full_indices
        dropped = self.dropped_full_indices
        dropped_retained = dropped_weights @ reduced_factor_covariance
        retained_dropped = reduced_factor_covariance @ np.swapaxes(
            dropped_weights, -1, -2
        )
        dropped_dropped = dropped_retained @ np.swapaxes(dropped_weights, -1, -2)

        full_covariance[:, retained[:, np.newaxis], retained] = (
            reduced_factor_covariance
        )
        full_covariance[:, dropped[:, np.newaxis], retained] = dropped_retained
        full_covariance[:, retained[:, np.newaxis], dropped] = retained_dropped
        full_covariance[:, dropped[:, np.newaxis], dropped] = dropped_dropped
        return full_covariance

    def project_factor_coordinates(self, factor_coordinates: FloatArray) -> FloatArray:
        r"""Project full factor-space coordinates into reduced-basis coordinates.

        This applies :math:`R_t^\top`. It is not the same operation as selecting
        retained factor-return columns because dropped factor coordinates contribute to
        retained reduced coordinates. It should be used for coordinates such as
        portfolio factor exposures. For 1D input, the ratios from the last observation
        are used.

        Parameters
        ----------
        factor_coordinates : ndarray of shape (n_full_factors,) or (n_observations, n_full_factors)
            Full factor-space coordinates. For 2D input, the observation axis must match
            the basis.

        Returns
        -------
        reduced_factor_coordinates : ndarray of shape (n_reduced_factors,) or (n_observations, n_reduced_factors)
            Reduced-basis coordinates after applying :math:`R_t^\top`.
        """
        factor_coordinates = np.asarray(factor_coordinates, dtype=float)
        if factor_coordinates.ndim not in (1, 2):
            raise ValueError("factor_coordinates must be a 1D or 2D array.")

        squeeze = factor_coordinates.ndim == 1
        if squeeze:
            factor_coordinates = factor_coordinates[np.newaxis, :]

        n_observations = factor_coordinates.shape[0]
        _validate_factor_dimension(
            factor_coordinates.shape[1],
            self.n_full_factors,
            "factor_coordinates",
        )
        if not squeeze:
            self._validate_observation_count(n_observations, "factor_coordinates")

        reduced = factor_coordinates[:, self.retained_full_indices].copy()

        for family, reduced_indices, ratio_slice in self._iter_families():
            family_ratios = (
                self.constraint_ratios[-1:, ratio_slice]
                if squeeze
                else self.constraint_ratios[:, ratio_slice]
            )
            retained = factor_coordinates[:, family.retained_full_indices]
            dropped = factor_coordinates[:, family.dropped_full_slice]
            reduced[:, reduced_indices] = retained - family_ratios * dropped

        return reduced[0] if squeeze else reduced

    def _dropped_factor_reduced_weights(
        self, observation_index: int | None = None
    ) -> FloatArray:
        """Return reduced-basis weights reconstructing dropped factors."""
        if observation_index is None:
            weights = np.zeros(
                (self.n_observations, self.n_constraints, self.n_reduced_factors),
                dtype=float,
            )
            for family_idx, (_, reduced_indices, ratio_slice) in enumerate(
                self._iter_families()
            ):
                weights[:, family_idx, reduced_indices] = -self.constraint_ratios[
                    :, ratio_slice
                ]
            return weights

        weights = np.zeros(
            (self.n_constraints, self.n_reduced_factors),
            dtype=float,
        )
        for family_idx, (_, reduced_indices, ratio_slice) in enumerate(
            self._iter_families()
        ):
            weights[family_idx, reduced_indices] = -self.constraint_ratios[
                observation_index, ratio_slice
            ]
        return weights


@dataclass(frozen=True)
class FamilyConstraint:
    r"""Metadata for one benchmark-weighted zero-sum constrained family.

    A constrained family contains the full-basis factor indices for one family,
    and identifies which factor is dropped.

    For a family of :math:`m` factors, the dropped factor :math:`k` is defined through
    the benchmark-weighted zero-sum constraint:

    .. math::

        c_t^\top f_{\mathrm{family}}(t) = 0, \qquad
        c_t(j) = \sum_i w^{\mathrm{bench}}_{t,i} B_{t,i,j}.

    The retained factors are parameterized by the constraint ratios:

    .. math::

        r_t(j) = \frac{c_t(j)}{c_t(k)}, \qquad j \neq k.

    Parameters
    ----------
    family_name : str
        Name of the constrained factor family, for example `industry`.

    full_factor_indices : ndarray of int, shape (m,)
        Absolute column indices of this family in the full factor basis.

    dropped_index_in_family : int
        Local position within `full_factor_indices` of the dropped factor.
    """

    family_name: str
    full_factor_indices: IntArray
    dropped_index_in_family: int

    def __post_init__(self):
        full_factor_indices = self.full_factor_indices
        if full_factor_indices.ndim != 1:
            raise ValueError("full_factor_indices must be a 1D array.")
        if full_factor_indices.size < 2:
            raise ValueError("A constrained family must contain at least two factors.")
        if np.unique(full_factor_indices).size != full_factor_indices.size:
            raise ValueError("full_factor_indices must be unique.")
        if not 0 <= self.dropped_index_in_family < full_factor_indices.size:
            raise ValueError("dropped_index_in_family is out of bounds.")

    @property
    def family_size(self) -> int:
        """Number of full-basis factors in this family."""
        return self.full_factor_indices.size

    @cached_property
    def retained_local_indices(self) -> IntArray:
        """Local positions within the family of retained factors."""
        return np.delete(np.arange(self.family_size), self.dropped_index_in_family)

    @property
    def dropped_full_index(self) -> int:
        """Absolute full-basis index of the dropped factor."""
        return int(self.full_factor_indices[self.dropped_index_in_family])

    @property
    def dropped_full_slice(self) -> slice:
        """Length-1 slice selecting the dropped factor in the full basis."""
        return slice(self.dropped_full_index, self.dropped_full_index + 1)

    @cached_property
    def retained_full_indices(self) -> IntArray:
        """Absolute full-basis indices of retained factors."""
        return self.full_factor_indices[self.retained_local_indices]


def compute_family_constraint_basis(
    constrained_families: list[tuple[str, str | None]],
    factor_exposures: FloatArray,
    benchmark_weights: FloatArray,
    factor_names: StrArray,
    factor_families: StrArray,
    tol: float = 1e-10,
) -> tuple[FamilyConstraintBasis, list[tuple[str, str]]]:
    r"""Build the compact basis for requested constrained families.

    The returned :class:`FamilyConstraintBasis` represents the time-varying change of
    basis between the full factor space :math:`\mathbb{R}^K` and the reduced full-rank
    factor space :math:`\mathbb{R}^{K_{\mathrm{red}}}`. The dense basis matrices
    :math:`R_t` are not materialized. Instead, the function stores one compact
    constraint-ratio matrix and the associated structural metadata.

    For each constrained family of :math:`m` factors, the full-basis factor returns
    satisfy the benchmark-weighted zero-sum condition:

    .. math::

        c_t^\top f_{\mathrm{family}}(t) = 0, \qquad
        c_t(j) = \sum_{i=1}^N w^{\mathrm{bench}}_{t,i} B_{t,i,j}.

    One factor :math:`k` is dropped and the remaining :math:`m - 1` reduced exposure
    columns are:

    .. math::

        z_j(t) = x_j(t) - \frac{c_t(j)}{c_t(k)} x_k(t),
        \qquad j \neq k.

    If `factor_to_drop` is `None`, the dropped factor is chosen as the one with the
    largest time-average absolute benchmark-weighted exposure within the family. This
    usually keeps the ratios :math:`c_t(j) / c_t(k)` moderate and improves numerical
    conditioning.
    Missing exposures are zero-filled before computing benchmark-weighted averages.

    Parameters
    ----------
    constrained_families : list[tuple[str, str | None]]
        `(family_name, factor_to_drop)` pairs. `factor_to_drop` may be `None` for
        automatic selection.

    factor_exposures : ndarray of shape (n_observations, n_assets, n_factors)
        Full-basis factor exposures.

    benchmark_weights : ndarray of shape (n_observations, n_assets)
        Non-negative benchmark weights, normalized internally.

    factor_names : ndarray of shape (n_factors,)
        Full-basis factor names.

    factor_families : ndarray of shape (n_factors,)
        Family label for each factor.

    tol : float, default=1e-10
        Absolute tolerance for the per-observation denominator check on the dropped
        factor's benchmark-weighted exposure :math:`c_t(k)`.

    Returns
    -------
    basis : FamilyConstraintBasis
        Compact basis representation.

    resolved_constraints : list[tuple[str, str]]
        One `(family_name, dropped_factor_name)` pair per constrained family, in the
        same order as the input. Any `None` `factor_to_drop` from the input is replaced
        by the heuristically chosen dropped factor name.

    Raises
    ------
    ValueError
        On invalid inputs, duplicate or missing families, unknown factor names, negative
        benchmark weights, near-singular benchmark-weighted exposures, or insufficient
        remaining factors.
    """
    factor_exposures = np.asarray(factor_exposures)
    benchmark_weights = np.asarray(benchmark_weights)
    factor_names = np.asarray(factor_names)
    factor_families = np.asarray(factor_families)

    if factor_exposures.ndim != 3:
        raise ValueError(
            "factor_exposures must be a 3D array of shape "
            "(n_observations, n_assets, n_factors)."
        )
    _, _, n_factors = factor_exposures.shape
    expected_weights_shape = factor_exposures.shape[:2]

    if benchmark_weights.shape != expected_weights_shape:
        raise ValueError(
            f"benchmark_weights shape {benchmark_weights.shape} does not "
            f"match (n_observations, n_assets) = {expected_weights_shape}."
        )
    if factor_names.shape != (n_factors,):
        raise ValueError(
            f"factor_names shape {factor_names.shape} does not match "
            f"(n_factors,) = ({n_factors},)."
        )
    if factor_families.shape != (n_factors,):
        raise ValueError(
            f"factor_families shape {factor_families.shape} does not match "
            f"(n_factors,) = ({n_factors},)."
        )
    if np.unique(factor_names).size != n_factors:
        raise ValueError("factor_names must be unique.")

    finite_mask = np.isfinite(benchmark_weights)
    if np.any(finite_mask & (benchmark_weights < 0)):
        raise ValueError("benchmark_weights must be non-negative.")
    benchmark_weights = np.where(finite_mask, benchmark_weights, 0.0)

    name_to_index = {v: i for i, v in enumerate(factor_names)}
    parsed_constraints: list[tuple[IntArray, int | None, str]] = []
    seen_families: set[str] = set()

    for family_name, factor_to_drop in constrained_families:
        if family_name in seen_families:
            raise ValueError(
                f"Family '{family_name}' appears more than once in "
                "constrained_families."
            )
        seen_families.add(family_name)

        family_indices = np.where(factor_families == family_name)[0].astype(
            int, copy=False
        )
        if family_indices.size == 0:
            raise ValueError(
                f"Factor family '{family_name}' not found. "
                f"Valid families: {np.unique(factor_families).tolist()}"
            )
        if family_indices.size < 2:
            raise ValueError(
                f"Factor family '{family_name}' must contain at least two factors."
            )

        if factor_to_drop is not None:
            if factor_to_drop not in name_to_index:
                raise ValueError(
                    f"Factor to drop '{factor_to_drop}' not found in "
                    f"{factor_names.tolist()}"
                )
            drop_full_index = name_to_index[factor_to_drop]
            if not np.any(family_indices == drop_full_index):
                raise ValueError(
                    f"Factor to drop '{factor_to_drop}' does not belong to "
                    f"family '{family_name}'."
                )
        else:
            drop_full_index = None

        parsed_constraints.append((family_indices, drop_full_index, family_name))

    if not parsed_constraints:
        raise ValueError("constrained_families must contain at least one family.")

    if n_factors <= len(parsed_constraints):
        raise ValueError(
            f"n_factors={n_factors} must exceed number of "
            f"constraints={len(parsed_constraints)}."
        )

    benchmark_weight_sums = benchmark_weights.sum(axis=1, keepdims=True)
    if np.any(benchmark_weight_sums <= 0):
        raise ValueError(
            "benchmark_weights must have a strictly positive sum at each "
            "observation after removing invalid entries."
        )
    weights = benchmark_weights / benchmark_weight_sums
    exposures_filled = np.nan_to_num(factor_exposures, nan=0.0)
    weighted_full_exposures = np.einsum(
        "tn,tnk->tk", weights, exposures_filled, optimize=True
    )

    family_constraints_resolved: list[FamilyConstraint] = []
    ratio_blocks = []
    resolved_constraints: list[tuple[str, str]] = []

    for family_indices, drop_full_index, family_name in parsed_constraints:
        family_size = family_indices.size
        weighted_family_exposures = weighted_full_exposures[:, family_indices]

        if drop_full_index is not None:
            dropped_local_index = int(np.where(family_indices == drop_full_index)[0][0])
        else:
            mean_absolute_exposure = np.mean(np.abs(weighted_family_exposures), axis=0)
            dropped_local_index = int(np.argmax(mean_absolute_exposure))

        retained_local_indices = np.delete(np.arange(family_size), dropped_local_index)

        dropped_exposure = weighted_family_exposures[:, dropped_local_index]
        if np.any(np.abs(dropped_exposure) < tol):
            raise ValueError(
                f"Family '{family_name}': dropped factor "
                f"'{factor_names[family_indices[dropped_local_index]]}' has "
                "near-zero benchmark-weighted exposure on some observations."
            )

        family_ratios = (
            weighted_family_exposures[:, retained_local_indices]
            / dropped_exposure[:, np.newaxis]
        )

        family_constraints_resolved.append(
            FamilyConstraint(
                family_name=family_name,
                full_factor_indices=family_indices,
                dropped_index_in_family=dropped_local_index,
            )
        )
        ratio_blocks.append(family_ratios)

        resolved_constraints.append(
            (
                family_name,
                str(factor_names[family_indices[dropped_local_index]]),
            )
        )

    basis = FamilyConstraintBasis(
        n_full_factors=n_factors,
        family_constraints=tuple(family_constraints_resolved),
        constraint_ratios=(
            ratio_blocks[0]
            if len(ratio_blocks) == 1
            else np.concatenate(ratio_blocks, axis=1)
        ),
    )

    return basis, resolved_constraints


def _stack_family_full_indices(
    family_constraints: tuple[FamilyConstraint, ...],
) -> IntArray:
    """Concatenate the full-basis indices of all constrained families."""
    if not family_constraints:
        return np.array([], dtype=int)
    return np.concatenate([family.full_factor_indices for family in family_constraints])


def _validate_factor_dimension(actual: int, expected: int, name: str) -> None:
    """Validate the factor dimension of an input."""
    if actual != expected:
        raise ValueError(
            f"`{name}` has factor dimension {actual}, expected {expected}."
        )
