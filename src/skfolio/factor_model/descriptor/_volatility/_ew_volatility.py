"""Exponentially weighted volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.factor_model.descriptor._volatility._base import _BaseEWVolatility


class EWVolatility(_BaseEWVolatility):
    r"""Exponentially weighted volatility descriptor.

    Computes return volatility with an EWMA variance estimate:

    .. math::

        S_i(t) = \lambda \cdot S_i(t-1) + (1 - \lambda) \cdot r_i(t)^2

        \text{output}_i(t) = \sqrt{\frac{S_i(t)}{1 - \lambda^{n_i(t)}}}

    where :math:`\lambda = \exp(-\ln(2)/\text{half\_life})` and :math:`n_i(t)` is the
    number of valid returns for asset :math:`i`.

    This descriptor uses raw returns, so the estimate includes both systematic and
    idiosyncratic risk. Use :class:`EWResidualVolatility` to remove market exposure
    first.

    Parameters
    ----------
    half_life : float, default=40.0
        EWMA half-life in observations.

    min_periods : int or None, default=None
        Minimum number of valid returns required for each asset. Until an asset
        reaches this count, its output is NaN. This warm-up period avoids exposing
        early EWMA values before the volatility estimate has sufficiently converged
        from its zero initialization. If `None`, defaults to
        :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    volatility_ : ndarray of shape (n_assets,)
        Last computed EWMA volatility. Contains NaN for inactive assets and assets
        that have not reached `min_periods` valid returns.

    Notes
    -----
    The EWMA variance accumulator is initialized to zero and bias-corrected at output
    time using each asset's valid observation count, matching
    :class:`~skfolio.moments.EWVariance`.

    NaNs are treated as missing observations. Active assets with missing returns keep
    their previous EWMA state and inactive assets output NaN and restart their warm-up
    period when they become active again. Non-missing returns must be finite.

    The variance is computed assuming centered returns (no demeaning), which is the
    standard convention for EWMA variance estimation in cross-sectional factor models.

    References
    ----------
    .. [1] "The cross-section of volatility and expected returns"
        The Journal of Finance. Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006).

    See Also
    --------
    EWDownsideVolatility : Downside variant using semi-deviation of returns.
    EWResidualVolatility : CAPM residual volatility (market exposure removed).

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import EWVolatility
    >>> descriptor = EWVolatility()
    >>> vol = descriptor.fit_transform(X)
    >>>
    >>> # Weekly data with faster-adapting vol
    >>> descriptor = EWVolatility(half_life=8)
    >>> vol = descriptor.fit_transform(X)
    """

    def __init__(self, half_life: float = 40.0, min_periods: int | None = None):
        super().__init__(
            half_life=half_life, min_acceptable_return=None, min_periods=min_periods
        )
