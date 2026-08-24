"""Exponentially weighted downside return volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.descriptor._volatility._base import _BaseEWVolatility


class EWDownsideVolatility(_BaseEWVolatility):
    r"""Exponentially weighted downside return volatility descriptor.

    Computes the downside semi-deviation of asset returns using EWMA estimation [1]_.
    Only returns below the `min_acceptable_return` threshold contribute to the variance
    estimate:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        D_i(t)
            &= \min(r_i(t) - \text{mar},\; 0) \\[0.75em]
        S_{\text{down},i}(t)
            &= \lambda \cdot S_{\text{down},i}(t-1)
               + (1 - \lambda) \cdot D_i(t)^2 \\[0.75em]
        \text{output}_i(t)
            &= \sqrt{\frac{S_{\text{down},i}(t)}
                         {1 - \lambda^{n_i(t)}}}
        \end{aligned}
        \]

    where :math:`\lambda = \exp(-\ln(2)/\text{half\_life})` and :math:`n_i(t)` is the
    number of valid returns for asset :math:`i`.

    Parameters
    ----------
    half_life : float, default=40.0
        EWMA half-life in observations.

    min_acceptable_return : float, default=0.0
        Threshold below which returns are considered "downside". The default of `0.0`
        defines downside as negative returns (losses).

    min_periods : int, optional
        Minimum number of valid returns required for each asset. Until an asset reaches
        this count, its output is NaN. This warm-up period avoids exposing early EWMA
        values before the downside volatility estimate has sufficiently converged from
        its zero initialization. If `None`, defaults to
        :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    volatility_ : ndarray of shape (n_assets,)
        Last computed EWMA downside volatility. Contains NaN for inactive assets and
        assets that have not reached `min_periods` valid returns.

    Notes
    -----
    The EWMA variance accumulator is initialized to zero and bias-corrected at output
    time using each asset's valid observation count, matching
    :class:`~skfolio.moments.EWVariance`.

    NaNs are treated as missing observations. Active assets with missing returns keep
    their previous EWMA state and inactive assets output NaN and restart their warm-up
    period when they become active again. Non-missing returns must be finite.

    References
    ----------
    .. [1] "The cross-section of volatility and expected returns"
        The Journal of Finance. Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006).

    See Also
    --------
    EWVolatility : Total (non-downside) variant.
    EWResidualDownsideVolatility : Downside CAPM residual volatility.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWDownsideVolatility
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = EWDownsideVolatility()
    >>> downside_volatility = descriptor.fit_transform(X)
    >>>
    >>> # Custom threshold
    >>> descriptor = EWDownsideVolatility(min_acceptable_return=-0.01)
    >>> downside_volatility = descriptor.fit_transform(X)
    """

    def __init__(
        self,
        half_life: float = 40.0,
        min_acceptable_return: float = 0.0,
        min_periods: int | None = None,
    ):
        super().__init__(
            half_life=half_life,
            min_acceptable_return=min_acceptable_return,
            min_periods=min_periods,
        )
