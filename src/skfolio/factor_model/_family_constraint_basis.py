"""Compact family-constraint basis for constrained factor families."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from skfolio.typing import FloatArray, ObjArray

__all__ = [
    "ConstrainedFamily",
    "FamilyConstraintBasis",
    "compute_family_constraint_basis",
]


@dataclass(frozen=True)
class FamilyConstraintBasis:
    r"""Compact representation of a family-constrained factor basis.

    This class represents the time-varying change of basis between the full factor space
    :math:`\mathbb{R}^K` and a reduced full-rank factor space
    :math:`\mathbb{R}^{K_{\mathrm{red}}}` obtained by imposing benchmark-weighted
    zero-sum constraints within selected factor families.

    The goal is to remove collinearity while preserving an economically interpretable
    basis (e.g. industry and/or country factors measured relative to the market factor).

    For each constrained family, the reduced basis spans the subspace where the family
    factor returns satisfy

    .. math::

        c_t^\top f_{\mathrm{family}}(t) = 0.

    For each observation :math:`t`, there is a conceptual basis matrix
    :math:`R_t \in \mathbb{R}^{K \times K_{\mathrm{red}}}` such that

    .. math::

        B_{\mathrm{red}}(t) = B_{\mathrm{full}}(t)\, R_t.

    The dense tensor :math:`R_t` is not stored. Instead, the representation stores the
    structural metadata of each constrained family and the time-varying basis
    coefficients :math:`r_t(\ell) = c_t(j_\ell) / c_t(k)`.

    This yields an :math:`O(T \cdot C)` representation, where :math:`C` is the total
    number of retained constrained factors, while preserving the original economic
    interpretation of the retained factors. Exposure transforms are applied family-wise
    without materializing the dense basis tensor and the same compact representation is
    reused for factor-return, mean, covariance, and attribution calculations.

    The basis is useful for:

    * **Cross-sectional regression**: regress on :math:`B_{\mathrm{red}}(t)` to estimate
      factor returns in a full-rank space without explicit linear equality constraints.
    * **Factor covariance estimation**: estimate moments in the reduced basis, where the
      constrained families are full-rank.
    * **Optimization / Cholesky**: work in the reduced basis to avoid singular factor
      covariance matrices.
    * **Attribution / reporting**: reconstruct full named factor returns via
      :math:`f_{\mathrm{full}}(t) = R_t\, g(t)`. Retained factors keep their original
      names and economic meaning Only the dropped factor is implicit, recovered from the
      zero-sum relation.

    Parameters
    ----------
    n_factors : int
        Full factor dimension :math:`K`.

    constrained_families : tuple of ConstrainedFamily
        One entry per constrained family.  Each family owns its own basis coefficients.
        All families must share the same observation count.
    """

    n_factors: int
    constrained_families: tuple[ConstrainedFamily, ...]

    def __post_init__(self):
        if self.n_factors < 1:
            raise ValueError("n_factors must be at least 1.")
        if not self.constrained_families:
            raise ValueError("constrained_families must contain at least one family.")

        constrained_full_indices = _stack_constrained_full_indices(
            self.constrained_families
        )
        if np.any(
            (constrained_full_indices < 0)
            | (constrained_full_indices >= self.n_factors)
        ):
            raise ValueError("Family factor indices must lie in [0, n_factors).")
        if np.unique(constrained_full_indices).size != constrained_full_indices.size:
            raise ValueError("Constrained families must be disjoint.")

        n_observations = self.constrained_families[0].n_observations
        for family in self.constrained_families[1:]:
            if family.n_observations != n_observations:
                raise ValueError(
                    "All constrained families must share the same number of "
                    "observations in basis_coefficients."
                )

    def __getitem__(self, observation_key) -> FamilyConstraintBasis:
        """Return a basis sliced along the observation axis."""
        if isinstance(observation_key, tuple):
            raise TypeError(
                "FamilyConstraintBasis only supports indexing along the "
                "observation axis."
            )

        sliced_families = []
        for family in self.constrained_families:
            sliced_coefficients = family.basis_coefficients[observation_key]
            if sliced_coefficients.ndim == 1:
                sliced_coefficients = sliced_coefficients[np.newaxis, :]
            sliced_families.append(
                ConstrainedFamily(
                    family_name=family.family_name,
                    full_factor_indices=family.full_factor_indices,
                    dropped_index_in_family=family.dropped_index_in_family,
                    basis_coefficients=sliced_coefficients,
                )
            )

        return FamilyConstraintBasis(
            n_factors=self.n_factors,
            constrained_families=tuple(sliced_families),
        )

    @property
    def n_observations(self) -> int:
        """Number of observations in the time axis."""
        return self.constrained_families[0].n_observations

    @property
    def n_constraints(self) -> int:
        """Number of constrained families (number of dropped factors)."""
        return len(self.constrained_families)

    @property
    def n_factors_reduced(self) -> int:
        r"""Reduced factor dimension :math:`K_{\mathrm{red}}`."""
        return self.n_factors - self.n_constraints

    @property
    def dropped_full_indices(self) -> np.ndarray:
        """Full-basis indices of the dropped factors."""
        return np.array(
            [f.dropped_full_index for f in self.constrained_families], dtype=int
        )

    @cached_property
    def _passthrough_full_indices(self) -> np.ndarray:
        """Full-basis indices not in any constrained family."""
        constrained_full_indices = _stack_constrained_full_indices(
            self.constrained_families
        )
        passthrough_mask = np.ones(self.n_factors, dtype=bool)
        passthrough_mask[constrained_full_indices] = False
        return np.nonzero(passthrough_mask)[0]

    @cached_property
    def _passthrough_reduced_cols(self) -> np.ndarray:
        """Reduced-basis column indices for passthrough factors."""
        return np.arange(len(self._passthrough_full_indices))

    @cached_property
    def _family_reduced_slices(self) -> list[slice]:
        """One reduced-basis column slice per constrained family."""
        slices = []
        offset = len(self._passthrough_full_indices)
        for family in self.constrained_families:
            n_retained = family.family_size - 1
            slices.append(slice(offset, offset + n_retained))
            offset += n_retained
        return slices

    @cached_property
    def _family_reduced_columns(self) -> list[np.ndarray]:
        """Reduced-basis column index arrays per constrained family."""
        return [np.arange(s.start, s.stop) for s in self._family_reduced_slices]

    def reduced_factor_names(self, full_factor_names: ObjArray) -> ObjArray:
        """Return factor names ordered in the reduced basis.

        The returned names align with reduced-basis outputs from methods such as
        `to_reduced_exposures`, `to_reduced_loading_matrix`, and
        `to_reduced_factor_returns`: passthrough factors first, followed by the
        retained factors of each constrained family in family order.

        Parameters
        ----------
        full_factor_names : ndarray of shape (n_factors,)
            Factor names in the full basis.

        Returns
        -------
        reduced_factor_names : ndarray of shape (n_factors_reduced,)
            Factor names aligned with the reduced basis.
        """
        full_factor_names = np.asarray(full_factor_names)
        if full_factor_names.ndim != 1:
            raise ValueError(
                "`full_factor_names` must be a 1D array of shape (n_factors,)."
            )
        _validate_factor_dimension(
            len(full_factor_names), self.n_factors, "full_factor_names"
        )

        parts = []
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            parts.append(full_factor_names[passthrough_full])
        for family in self.constrained_families:
            parts.append(full_factor_names[family.kept_full_indices])
        return np.concatenate(parts)

    def _iter_families(self) -> Iterator[tuple[ConstrainedFamily, slice]]:
        """Iterate over `(family, reduced_slice)` for each constrained family."""
        return zip(self.constrained_families, self._family_reduced_slices, strict=True)

    def _validate_observation_count(self, n_observations: int, name: str) -> None:
        """Validate that a time-varying input matches the basis length."""
        if n_observations != self.n_observations:
            raise ValueError(
                f"`{name}` has {n_observations} observations but the basis "
                f"has {self.n_observations}. Slice the FamilyConstraintBasis "
                "first to align both time axes."
            )

    def to_reduced_exposures(self, exposures: FloatArray) -> FloatArray:
        r"""Map full-basis exposures to the reduced basis.

        For each constrained family with dropped factor :math:`k` and retained factors
        :math:`j \neq k`:

        .. math::

            X^{\mathrm{red}}_t(:, j)
            = X_t(:, j) - r_t(j)\, X_t(:, k)

        Factors outside the constrained families are copied unchanged.

        Parameters
        ----------
        exposures : ndarray of shape (n_observations, n_assets, n_factors)
            Full-basis exposures. The time axis must match the basis.

        Returns
        -------
        exposures_reduced : ndarray of shape (n_observations, n_assets, n_factors_reduced)
            Reduced-basis exposures with the same observation and asset axes.
        """
        if exposures.ndim != 3:
            raise ValueError(
                "`exposures` must be a 3D array of shape "
                "(n_observations, n_assets, n_factors)."
            )
        n_obs, n_assets, _ = exposures.shape
        self._validate_observation_count(n_obs, "exposures")
        _validate_factor_dimension(exposures.shape[2], self.n_factors, "exposures")

        exposures_reduced = np.empty(
            (n_obs, n_assets, self.n_factors_reduced), dtype=float
        )
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            exposures_reduced[:, :, self._passthrough_reduced_cols] = exposures[
                :, :, passthrough_full
            ]

        for family, reduced_slice in self._iter_families():
            retained = exposures[:, :, family.kept_full_indices]
            dropped = exposures[:, :, family.dropped_full_slice]
            exposures_reduced[:, :, reduced_slice] = (
                retained - dropped * family.basis_coefficients[:, np.newaxis, :]
            )

        return exposures_reduced

    def to_reduced_loading_matrix(
        self, loading_matrix: FloatArray, observation_index: int = -1
    ) -> FloatArray:
        r"""Map a point-in-time loading matrix to the reduced basis.

        This is the single-date specialization of :meth:`to_reduced_exposures`.
        It applies the basis coefficients at `observation_index` directly to a 2D
        loading matrix.

        For each constrained family with dropped factor :math:`k` and retained factors
        :math:`j \neq k`:

        .. math::

            X^{\mathrm{red}}(:, j)
            = X(:, j) - r(j)\, X(:, k)

        where :math:`r(j)` are the basis coefficients at the selected observation.
        Factors outside the constrained families are copied unchanged.

        Parameters
        ----------
        loading_matrix : ndarray of shape (n_assets, n_factors)
            Full-basis point-in-time loading matrix.

        observation_index : int, default=-1
            Index into the time axis selecting which observation's coefficients to use.

        Returns
        -------
        loading_reduced : ndarray of shape (n_assets, n_factors_reduced)
            Reduced-basis loading matrix at the selected observation.
        """
        if loading_matrix.ndim != 2:
            raise ValueError(
                "`loading_matrix` must be a 2D array of shape (n_assets, n_factors)."
            )
        _validate_factor_dimension(
            loading_matrix.shape[1], self.n_factors, "loading_matrix"
        )

        n_assets = loading_matrix.shape[0]
        loading_reduced = np.empty((n_assets, self.n_factors_reduced), dtype=float)
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            loading_reduced[:, self._passthrough_reduced_cols] = loading_matrix[
                :, passthrough_full
            ]

        for family, reduced_slice in self._iter_families():
            family_coefficients = family.basis_coefficients[observation_index]
            retained = loading_matrix[:, family.kept_full_indices]
            dropped = loading_matrix[:, family.dropped_full_slice]
            loading_reduced[:, reduced_slice] = (
                retained - dropped * family_coefficients[np.newaxis, :]
            )

        return loading_reduced

    def to_full_factor_returns(self, reduced_factor_returns: FloatArray) -> FloatArray:
        r"""Map reduced factor returns back to full named factor returns.

        For each constrained family, the retained factor returns are copied through and
        the dropped factor's return is recovered from the zero-sum relation:

        .. math::

            f^{\mathrm{full}}_j(t) = g_j(t), \quad j \neq k, \qquad
            f^{\mathrm{full}}_k(t) = -\sum_{j \neq k} r_t(j)\, g_j(t)

        The retained factors therefore keep their original names and economic
        interpretation in the reduced basis. For 1D input, the basis coefficients from
        the last observation are used.

        Parameters
        ----------
        reduced_factor_returns : ndarray of shape (n_observations, n_factors_reduced) or (n_factors_reduced,)
            Reduced-basis factor returns. For 2D input, the observation axis must match
            the basis. For 1D input, the basis coefficients from the last observation
            are used.

        Returns
        -------
        factor_returns_full : ndarray of shape (n_observations, n_factors) or (n_factors,)
            Full-basis factor returns reconstructed from the reduced basis.
        """
        if reduced_factor_returns.ndim not in (1, 2):
            raise ValueError("`reduced_factor_returns` must be a 1D or 2D array.")
        squeeze = reduced_factor_returns.ndim == 1
        if squeeze:
            reduced_factor_returns = reduced_factor_returns[np.newaxis, :]

        n_obs = reduced_factor_returns.shape[0]
        _validate_factor_dimension(
            reduced_factor_returns.shape[1],
            self.n_factors_reduced,
            "reduced_factor_returns",
        )
        if not squeeze:
            self._validate_observation_count(n_obs, "reduced_factor_returns")

        factor_returns = np.empty((n_obs, self.n_factors), dtype=float)
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            factor_returns[:, passthrough_full] = reduced_factor_returns[
                :, self._passthrough_reduced_cols
            ]

        for family, reduced_slice in self._iter_families():
            family_coefficients = (
                family.basis_coefficients[-1:] if squeeze else family.basis_coefficients
            )
            retained_returns = reduced_factor_returns[:, reduced_slice]
            factor_returns[:, family.kept_full_indices] = retained_returns
            factor_returns[:, family.dropped_full_index] = -np.sum(
                family_coefficients * retained_returns, axis=-1
            )

        return factor_returns[0] if squeeze else factor_returns

    def to_reduced_factor_returns(self, factor_returns: FloatArray) -> FloatArray:
        """Extract reduced factor returns from full factor returns.

        The reduced factor returns are simply the retained full-basis factor returns.
        No basis coefficients are applied in this transform, it simply selects the
        retained factors.

        Parameters
        ----------
        factor_returns : ndarray of shape (n_observations, n_factors) or (n_factors,)
            Full-basis factor returns. For 2D input, the observation axis must match the
            basis.

        Returns
        -------
        factor_returns_reduced : ndarray of shape (n_observations, n_factors_reduced) or (n_factors_reduced,)
            Reduced-basis factor returns obtained by selecting passthrough and retained
            family factors.
        """
        if factor_returns.ndim not in (1, 2):
            raise ValueError("`factor_returns` must be a 1D or 2D array.")
        squeeze = factor_returns.ndim == 1
        if squeeze:
            factor_returns = factor_returns[np.newaxis, :]

        n_obs = factor_returns.shape[0]
        _validate_factor_dimension(
            factor_returns.shape[1], self.n_factors, "factor_returns"
        )

        factor_returns_reduced = np.empty((n_obs, self.n_factors_reduced), dtype=float)
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            factor_returns_reduced[:, self._passthrough_reduced_cols] = factor_returns[
                :, passthrough_full
            ]

        for family, reduced_slice in self._iter_families():
            factor_returns_reduced[:, reduced_slice] = factor_returns[
                :, family.kept_full_indices
            ]

        return factor_returns_reduced[0] if squeeze else factor_returns_reduced

    def to_full_factor_mu(
        self, reduced_factor_mu: FloatArray, observation_index: int = -1
    ) -> FloatArray:
        r"""Map a reduced factor mean vector back to the full basis.

        The basis coefficients at `observation_index` are used to reconstruct the
        dropped factor mean in each constrained family. Passthrough and retained factors
        are copied unchanged.

        Parameters
        ----------
        reduced_factor_mu : ndarray of shape (n_factors_reduced,)
            Reduced-basis factor mean vector at the selected observation.

        observation_index : int, default=-1
            Index into the time axis of the basis coefficients.

        Returns
        -------
        mu_full : ndarray of shape (n_factors,)
            Full-basis factor mean vector reconstructed from the reduced basis.
        """
        if reduced_factor_mu.ndim != 1:
            raise ValueError("`reduced_factor_mu` must be a 1D array.")
        _validate_factor_dimension(
            len(reduced_factor_mu), self.n_factors_reduced, "reduced_factor_mu"
        )

        factor_mu = np.empty(self.n_factors, dtype=float)
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            factor_mu[passthrough_full] = reduced_factor_mu[
                self._passthrough_reduced_cols
            ]

        for family, reduced_slice in self._iter_families():
            family_coefficients = family.basis_coefficients[observation_index]
            retained_means = reduced_factor_mu[reduced_slice]
            factor_mu[family.kept_full_indices] = retained_means
            factor_mu[family.dropped_full_index] = -family_coefficients @ retained_means

        return factor_mu

    def to_full_factor_covariance(
        self, reduced_factor_covariance: FloatArray, observation_index: int = -1
    ) -> FloatArray:
        r"""Map reduced factor covariance to the full factor basis.

        Applies

        .. math::

            \Sigma^{\mathrm{full}} = R\,\Sigma^{\mathrm{red}}\,R^\top

        using the transient dense basis matrix :math:`R`.

        Parameters
        ----------
        reduced_factor_covariance : ndarray of shape (n_factors_reduced, n_factors_reduced) or (n_observations, n_factors_reduced, n_factors_reduced)
            Reduced-basis factor covariance. When 3D, each slice along the first axis is
            mapped back using the corresponding observation's basis, and the observation
            axis must match the basis.

        observation_index : int, default=-1
            Index into the time axis of the basis coefficients. Only used when
            `reduced_factor_covariance` is 2D; ignored for 3D input.

        Returns
        -------
        cov_full : ndarray of shape (n_factors, n_factors) or (n_observations, n_factors, n_factors)
            Full-basis factor covariance.
        """
        if reduced_factor_covariance.ndim not in (2, 3):
            raise ValueError("`reduced_factor_covariance` must be a 2D or 3D array.")
        if reduced_factor_covariance.shape[-2:] != (
            self.n_factors_reduced,
            self.n_factors_reduced,
        ):
            raise ValueError(
                "`reduced_factor_covariance` must have trailing shape "
                f"({self.n_factors_reduced}, {self.n_factors_reduced})."
            )

        if reduced_factor_covariance.ndim == 2:
            dense_basis = self._dense_basis_at(observation_index)
            return dense_basis @ reduced_factor_covariance @ dense_basis.T

        self._validate_observation_count(
            reduced_factor_covariance.shape[0], "reduced_factor_covariance"
        )
        dense_basis = self._dense_basis_tensor(reduced_factor_covariance.shape[0])
        return dense_basis @ reduced_factor_covariance @ dense_basis.transpose(0, 2, 1)

    def to_reduced_factor_coordinates(
        self, factor_coordinates: FloatArray
    ) -> FloatArray:
        r"""Map a full-basis vector to the reduced basis via :math:`R_t^\top v`.

        Unlike :meth:`to_reduced_factor_returns`, this method applies the full transpose
        :math:`R_t^\top`. It therefore includes the contribution of the dropped factors
        and should be used for vectors such as portfolio factor exposures
        :math:`g = B^\top w`. For 1D input, the basis  coefficients from the last
        observation are used.

        Parameters
        ----------
        factor_coordinates : ndarray of shape (n_factors,) or (n_observations, n_factors)
            Full-basis vector. For 2D input, the observation axis must match the
            basis. For 1D input, the basis coefficients from the last observation are
            used.

        Returns
        -------
        reduced_factor_coordinates : ndarray of shape (n_factors_reduced,) or (n_observations, n_factors_reduced)
            Reduced-basis vector after applying :math:`R_t^\top`.
        """
        if factor_coordinates.ndim not in (1, 2):
            raise ValueError("`factor_coordinates` must be a 1D or 2D array.")
        squeeze = factor_coordinates.ndim == 1
        if squeeze:
            factor_coordinates = factor_coordinates[np.newaxis, :]

        n_obs = factor_coordinates.shape[0]
        _validate_factor_dimension(
            factor_coordinates.shape[1], self.n_factors, "factor_coordinates"
        )
        if not squeeze:
            self._validate_observation_count(n_obs, "factor_coordinates")

        reduced = np.empty((n_obs, self.n_factors_reduced), dtype=float)
        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            reduced[:, self._passthrough_reduced_cols] = factor_coordinates[
                :, passthrough_full
            ]

        for family, reduced_slice in self._iter_families():
            family_coefficients = (
                family.basis_coefficients[-1:] if squeeze else family.basis_coefficients
            )
            retained = factor_coordinates[:, family.kept_full_indices]
            dropped = factor_coordinates[:, family.dropped_full_slice]
            reduced[:, reduced_slice] = retained - family_coefficients * dropped

        return reduced[0] if squeeze else reduced

    def _dense_basis_tensor(self, n_obs: int) -> np.ndarray:
        r"""Build the dense basis tensor
        :math:`R \in \mathbb{R}^{T \times K \times K_{\mathrm{red}}}`.
        """
        self._validate_observation_count(n_obs, "reduced_factor_covariance")
        dense_basis = np.zeros(
            (n_obs, self.n_factors, self.n_factors_reduced), dtype=float
        )

        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            dense_basis[:, passthrough_full, self._passthrough_reduced_cols] = 1.0

        for family, reduced_columns in zip(
            self.constrained_families, self._family_reduced_columns, strict=True
        ):
            dense_basis[:, family.kept_full_indices, reduced_columns] = 1.0
            dense_basis[
                :, family.dropped_full_index, reduced_columns
            ] = -family.basis_coefficients

        return dense_basis

    def _dense_basis_at(self, observation_index: int) -> np.ndarray:
        r"""Build the dense basis
        :math:`R \in \mathbb{R}^{K \times K_{\mathrm{red}}}` for one observation.
        """
        dense_basis = np.zeros((self.n_factors, self.n_factors_reduced), dtype=float)

        passthrough_full = self._passthrough_full_indices
        if passthrough_full.size:
            dense_basis[passthrough_full, self._passthrough_reduced_cols] = 1.0

        for family, reduced_columns in zip(
            self.constrained_families, self._family_reduced_columns, strict=True
        ):
            family_coefficients = family.basis_coefficients[observation_index]
            dense_basis[family.kept_full_indices, reduced_columns] = 1.0
            dense_basis[
                family.dropped_full_index, reduced_columns
            ] = -family_coefficients

        return dense_basis


@dataclass(frozen=True)
class ConstrainedFamily:
    r"""Structural metadata and basis coefficients for one constrained factor family.

    A constrained family contains the full-basis factor indices for one family,
    identifies which factor is dropped, and store the time-varying basis coefficients
    used to express the retained factors relative to the dropped one.

    For a family of :math:`m` factors, the dropped factor :math:`k` is defined through
    the benchmark-weighted zero-sum constraint

    .. math::

        c_t^\top f_{\mathrm{family}}(t) = 0, \qquad
        c_t(j) = \sum_{i} w^{\mathrm{bench}}_{t,i}\, B_{t,i,j}.

    The retained factors are parameterized by the basis coefficients

    .. math::

        r_t(\ell) = \frac{c_t(j_\ell)}{c_t(k)},
        \qquad \ell = 1, \ldots, m-1.

    Parameters
    ----------
    family_name : str
        Name of the constrained factor family (e.g. `"industry"`).

    full_factor_indices : ndarray of int, shape (m,)
        Absolute column indices of this family in the full factor basis.

    dropped_index_in_family : int
        Local position (within `full_factor_indices`) of the dropped factor.

    basis_coefficients : ndarray of shape (n_observations, m - 1)
        Time-varying basis coefficients :math:`r_t(\ell)` for the retained factors.
        Column order matches `kept_local_indices`.
    """

    family_name: str
    full_factor_indices: np.ndarray
    dropped_index_in_family: int
    basis_coefficients: FloatArray

    def __post_init__(self):
        full_factor_indices = np.asarray(self.full_factor_indices, dtype=int)
        if full_factor_indices.ndim != 1:
            raise ValueError("full_factor_indices must be a 1D array.")
        if full_factor_indices.size < 2:
            raise ValueError("full_factor_indices must contain at least two factors.")
        if np.unique(full_factor_indices).size != full_factor_indices.size:
            raise ValueError("full_factor_indices must not contain duplicates.")
        if not 0 <= self.dropped_index_in_family < full_factor_indices.size:
            raise ValueError(
                "dropped_index_in_family must be a valid position in "
                "full_factor_indices."
            )

        basis_coefficients = np.asarray(self.basis_coefficients, dtype=float)
        if basis_coefficients.ndim != 2:
            raise ValueError("basis_coefficients must be a 2D array.")
        if basis_coefficients.shape[1] != full_factor_indices.size - 1:
            raise ValueError(
                "basis_coefficients must have shape (n_observations, family_size - 1)."
            )
        if not np.all(np.isfinite(basis_coefficients)):
            raise ValueError("basis_coefficients must contain only finite values.")

        object.__setattr__(self, "full_factor_indices", full_factor_indices)
        object.__setattr__(self, "basis_coefficients", basis_coefficients)

    @property
    def n_observations(self) -> int:
        """Number of observations in the time axis."""
        return self.basis_coefficients.shape[0]

    @property
    def family_size(self) -> int:
        """Number of full-basis factors in this family."""
        return len(self.full_factor_indices)

    @cached_property
    def kept_local_indices(self) -> np.ndarray:
        """Local positions (within the family) of the retained factors."""
        return np.delete(
            np.arange(len(self.full_factor_indices)), self.dropped_index_in_family
        )

    @cached_property
    def dropped_full_index(self) -> int:
        """Absolute full-basis index of the dropped factor."""
        return int(self.full_factor_indices[self.dropped_index_in_family])

    @cached_property
    def kept_full_indices(self) -> np.ndarray:
        """Absolute full-basis indices of the retained factors."""
        return self.full_factor_indices[self.kept_local_indices]

    @cached_property
    def dropped_full_slice(self) -> slice:
        """Length-1 slice selecting the dropped factor in the full basis."""
        return slice(self.dropped_full_index, self.dropped_full_index + 1)


def compute_family_constraint_basis(
    constrained_families: list[tuple[str, str | None]],
    factor_exposures: FloatArray,
    benchmark_weights: FloatArray,
    factor_names: ObjArray,
    factor_families: ObjArray,
    tol: float = 1e-10,
) -> tuple[FamilyConstraintBasis, list[tuple[str, str]]]:
    r"""Build the compact basis for the requested constrained families.

    The returned :class:`FamilyConstraintBasis` represents the time-varying change of
    basis between the full factor space :math:`\mathbb{R}^K` and the reduced full-rank
    factor space :math:`\mathbb{R}^{K_{\mathrm{red}}}`. The dense basis matrices
    :math:`R_t` are not materialized. Instead, the function stores only the per-family
    basis coefficients and the associated structural metadata.

    For each constrained family of :math:`m` factors, the full-basis factor returns
    satisfy the benchmark-weighted zero-sum condition

    .. math::

        c_t^\top f_{\mathrm{family}}(t) = 0, \qquad
        c_t(j) = \sum_{i=1}^N w^{\mathrm{bench}}_{t,i}\, B_{t,i,j}.

    One factor :math:`k` is dropped and the remaining :math:`m - 1` reduced exposure
    columns are

    .. math::

        z_j(t) = x_j(t) - \frac{c_t(j)}{c_t(k)}\, x_k(t),
        \qquad j \neq k.

    If :math:`g(t)` is the reduced factor-return vector for that family, the full
    factor-return vector is reconstructed as :math:`f_{\mathrm{full}}(t) = R_t\, g(t)`
    with

    * for every retained factor :math:`j \neq k`: :math:`g_j(t) = f^{\mathrm{full}}_j(t)`;
    * only the dropped factor is implicit, recovered from the zero-sum relation.

    The retained factors therefore keep their original names and economic interpretation
    in the reduced basis.

    When `factor_to_drop` is `None`, the dropped factor is chosen as the one with the
    largest time-average absolute benchmark-weighted exposure within the family. This
    usually keeps the basis coefficients :math:`c_t(j) / c_t(k)` moderate and improves
    numerical conditioning.

    Missing exposures are zero-filled before computing the benchmark-weighted averages.

    Parameters
    ----------
    constrained_families : list[tuple[str, str | None]]
        `(family_name, factor_to_drop)` pairs. `factor_to_drop` may be `None` for
        automatic selection (largest time-average absolute benchmark-weighted exposure).

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
        One `(family_name, dropped_factor_name)` pair per constrained family,
        in the same order as the input. Any `None` `factor_to_drop` from the
        input is replaced by the heuristically chosen dropped factor name.

    Raises
    ------
    ValueError
        On invalid inputs (shape mismatches, duplicate or missing families,
        unknown factor names, negative benchmark weights, near-singular
        benchmark-weighted exposures, or insufficient remaining factors).
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

    # Parse and validate constraints.
    name_to_index = {v: i for i, v in enumerate(factor_names)}
    parsed_constraints: list[tuple[np.ndarray, int | None, str]] = []
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

    constrained_families_resolved: list[ConstrainedFamily] = []
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

        family_coefficients = (
            weighted_family_exposures[:, retained_local_indices]
            / dropped_exposure[:, np.newaxis]
        )

        constrained_families_resolved.append(
            ConstrainedFamily(
                family_name=family_name,
                full_factor_indices=family_indices,
                dropped_index_in_family=dropped_local_index,
                basis_coefficients=family_coefficients,
            )
        )

        resolved_constraints.append(
            (
                family_name,
                str(factor_names[family_indices[dropped_local_index]]),
            )
        )

    basis = FamilyConstraintBasis(
        n_factors=n_factors,
        constrained_families=tuple(constrained_families_resolved),
    )

    return basis, resolved_constraints


def _stack_constrained_full_indices(
    constrained_families: tuple[ConstrainedFamily, ...],
) -> np.ndarray:
    """Concatenate the full-basis indices of all constrained families."""
    if not constrained_families:
        return np.array([], dtype=int)
    return np.concatenate([f.full_factor_indices for f in constrained_families])


def _validate_factor_dimension(actual: int, expected: int, name: str) -> None:
    """Validate the factor dimension of an input."""
    if actual != expected:
        raise ValueError(
            f"`{name}` has factor dimension {actual}, expected {expected}."
        )
