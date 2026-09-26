"""Covariance square root dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

from skfolio.typing import FloatArray

__all__ = ["CovarianceSqrt"]


@dataclass(frozen=True, eq=False)
class CovarianceSqrt:
    r"""Matrix square root decomposition of a covariance matrix.

    Encodes :math:`\Sigma = \sum_i A_i A_i^\top + \operatorname{diag}(d)^2` in a form
    suitable for second-order cone (SOC) constraints:

    .. math::

        \left\lVert
        \begin{pmatrix}
            A_1^\top w \\
            \vdots \\
            A_m^\top w \\
            d \odot w
        \end{pmatrix}
        \right\rVert_2
        \le v
        \;\Longleftrightarrow\;
        w^\top \Sigma\, w \le v^2

    This representation avoids forming a full :math:`(n \times n)` Cholesky factor when
    the covariance has lower-dimensional components and a diagonal component.

    Attributes
    ----------
    components : tuple of ndarray of shape (n, k_i)
        Matrices :math:`A_i` of shape :math:`(n, k_i)` contributing
        :math:`\sum_i A_i A_i^\top` to the covariance.

    diagonal : ndarray of shape (n,) or None
        Vector :math:`d` contributing :math:`\operatorname{diag}(d)^2` to the
        covariance.
    """

    components: tuple[FloatArray, ...] = ()
    diagonal: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate component dimensions."""
        if len(self.components) == 0 and self.diagonal is None:
            raise ValueError(
                "At least one covariance square root component is required."
            )

        n_assets = None
        for component in self.components:
            if component.ndim != 2:
                raise ValueError("Covariance square root components must be 2D arrays.")
            if component.shape[0] == 0 or component.shape[1] == 0:
                raise ValueError("Covariance square root components cannot be empty.")
            if n_assets is None:
                n_assets = component.shape[0]
            elif component.shape[0] != n_assets:
                raise ValueError(
                    "Covariance square root components must have matching row counts."
                )

        if self.diagonal is None:
            return

        if self.diagonal.ndim != 1:
            raise ValueError("Covariance square root diagonal must be a 1D array.")
        if self.diagonal.shape[0] == 0:
            raise ValueError("Covariance square root diagonal cannot be empty.")
        if n_assets is not None and self.diagonal.shape[0] != n_assets:
            raise ValueError(
                "Covariance square root diagonal must match component row counts."
            )
