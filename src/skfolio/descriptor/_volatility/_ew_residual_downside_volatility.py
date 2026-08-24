"""Exponentially weighted downside CAPM residual volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.descriptor._volatility._base import _BaseEWResidualVolatility


class EWResidualDownsideVolatility(_BaseEWResidualVolatility):
    r"""Exponentially weighted downside CAPM residual volatility descriptor.

    Computes downside volatility of CAPM residuals with an EWMA variance estimate.
    Only residuals below the `min_acceptable_return` threshold contribute:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        \epsilon_i(t)
            &= r_i(t) - \hat\beta_i(t) \cdot r_m(t) \\[0.75em]
        D_i(t)
            &= \min(\epsilon_i(t) - \text{mar},\; 0) \\[0.75em]
        S_{\text{down},i}(t)
            &= \lambda_v \cdot S_{\text{down},i}(t-1)
               + (1 - \lambda_v) \cdot D_i(t)^2 \\[0.75em]
        \text{output}_i(t)
            &= \sqrt{\frac{S_{\text{down},i}(t)}
                         {1 - \lambda_v^{n_i(t)}}}
        \end{aligned}
        \]

    where :math:`\hat\beta_i(t)` is the EWMA beta estimated with decay
    :math:`\lambda_\beta = \exp(-\ln(2)/\text{beta\_half\_life})`,
    :math:`\lambda_v = \exp(-\ln(2)/\text{half\_life})` and :math:`n_i(t)` is the
    number of valid returns for asset :math:`i`. The zero-initialized residual variance
    accumulator is bias-corrected at output time using each asset's valid observation
    count.

    This measures stock-specific downside risk after removing market exposure.

    Parameters
    ----------
    half_life : float, default=40.0
        EWMA half-life in observations for the residual variance estimator.

    beta_half_life : float, default=60.0
        EWMA half-life in observations for the beta estimator.

    min_acceptable_return : float, default=0.0
        Threshold below which residuals are considered "downside". The default of `0.0`
        defines downside as negative residuals (losses after removing market exposure).

    min_periods : int, optional
        Minimum number of valid returns required for each asset. Until an asset
        reaches this count, its output is NaN. This warm-up period avoids exposing
        early EWMA values before the downside residual volatility estimate has
        sufficiently converged from its zero initialization. If `None`, defaults to
        :math:`\lceil\max(\text{half\_life}, \text{beta\_half\_life})\rceil`, with a
        minimum of 1.

    eps : float, default=1e-12
        Small constant for numerical stability in :math:`1 / \text{Var}(r_m)` when
        computing beta.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    residual_volatility_ : ndarray of shape (n_assets,)
        Last computed EWMA downside residual volatility. Contains NaN for inactive
        assets and assets that have not reached `min_periods` valid returns.

    Notes
    -----
    NaNs are treated as missing observations. Active assets with missing returns
    keep their previous asset-specific EWMA state; inactive assets output NaN and
    restart their warm-up period when they become active again. Non-missing returns
    must be finite.

    Market returns are computed from the estimation universe (`estimation_mask` of
    :class:`~skfolio.containers.AssetPanel`). If no estimable asset has both finite
    returns and finite `market_cap` at an observation, the market return is undefined
    and a `ValueError` is raised.

    See Also
    --------
    EWResidualVolatility : Total (non-downside) variant.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWResidualDownsideVolatility
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # Downside residual volatility (losses only)
    >>> descriptor = EWResidualDownsideVolatility()
    >>> residual_downside_volatility = descriptor.fit_transform(X)
    >>>
    >>> # Custom threshold
    >>> descriptor = EWResidualDownsideVolatility(min_acceptable_return=-0.01)
    >>> residual_downside_volatility = descriptor.fit_transform(X)
    """

    def __init__(
        self,
        half_life: float = 40.0,
        beta_half_life: float = 60.0,
        min_acceptable_return: float = 0.0,
        min_periods: int | None = None,
        eps: float = 1e-12,
    ):
        super().__init__(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_acceptable_return=min_acceptable_return,
            min_periods=min_periods,
            eps=eps,
        )
