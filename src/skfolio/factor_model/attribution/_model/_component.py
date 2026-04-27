"""Component Dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

from skfolio.typing import FloatArray

__all__ = ["Component"]


@dataclass(frozen=True)
class Component:
    r"""Portfolio attribution component.

    Represents one component of the portfolio attribution: systematic, idiosyncratic,
    unexplained, or total. Each component stores volatility and return contributions,
    their percentages of total portfolio variance and return, standalone volatility,
    correlation with the portfolio and optional return uncertainty.

    For single-point attribution, fields are floats. For rolling attribution (from
    :func:`rolling_realized_factor_attribution`), fields are 1D arrays of shape
    `(n_windows,)`.

    Attributes
    ----------
    vol_contrib : float or ndarray of shape (n_windows,)
        Volatility contribution to total portfolio volatility.

    pct_total_variance : float or ndarray of shape (n_windows,)
        Percentage of total portfolio variance.

    mu_contrib : float or ndarray of shape (n_windows,)
        Return contribution to total portfolio return (expected return for predicted
        attribution and mean return for realized attribution).

    pct_total_mu : float or ndarray of shape (n_windows,)
        Percentage of total portfolio return.

    vol : float or ndarray of shape (n_windows,)
        Standalone component volatility.

    corr_with_ptf : float or ndarray of shape (n_windows,)
        Correlation with portfolio returns.

    mu_uncertainty : float or ndarray of shape (n_windows,) or None
        Standard error of the mean return attribution, reflecting estimation uncertainty
        in the cross-sectional factor return regression. The systematic and
        idiosyncratic values are equal because their estimation errors sum to zero
        (the total portfolio return is observed). `None` when uncertainty is not
        computed.
    """

    vol_contrib: float | FloatArray
    pct_total_variance: float | FloatArray
    mu_contrib: float | FloatArray
    pct_total_mu: float | FloatArray

    vol: float | FloatArray
    corr_with_ptf: float | FloatArray

    mu_uncertainty: float | FloatArray | None = None

    @property
    def mu(self) -> float | FloatArray:
        """Standalone component return (expected return for predicted attribution and
        mean return for realized attribution).

        Components do not store a separate standalone return statistic. Unlike `vol`,
        the component-level return is already its contribution to total portfolio
        return, so `mu` is equal to `mu_contrib`.
        """
        return self.mu_contrib
