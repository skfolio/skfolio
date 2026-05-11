"""Exponentially weighted CAPM residual volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.factor_model.descriptor._volatility._base import _BaseEWResidualVolatility


class EWResidualVolatility(_BaseEWResidualVolatility):
    r"""Exponentially weighted CAPM residual volatility descriptor.

    Computes volatility of CAPM residuals with an EWMA variance estimate:

    .. math::

        \epsilon_i(t) = r_i(t) - \hat\beta_i(t) \cdot r_m(t)

        S_{\epsilon,i}(t) = \lambda_v \cdot S_{\epsilon,i}(t-1)
            + (1 - \lambda_v) \cdot \epsilon_i(t)^2

        \text{output}_i(t) =
            \sqrt{\frac{S_{\epsilon,i}(t)}
            {1 - \lambda_v^{n_i(t)}}}

    where :math:`\hat\beta_i(t)` is the EWMA beta estimated with decay
    :math:`\lambda_\beta = \exp(-\ln(2)/\text{beta\_half\_life})` and the residual
    variance uses decay :math:`\lambda_v = \exp(-\ln(2)/\text{half\_life})`. The
    zero-initialized residual variance accumulator is bias-corrected at output
    time using each asset's valid observation count :math:`n_i(t)`.

    The market return :math:`r_m(t)` is computed as the cap-weighted average of returns
    in the estimation universe.

    Residual volatility isolates the part of return variation not explained by the
    market. This can be useful when market beta is already modeled separately and the
    intended signal is stock-specific risk after removing market exposure [1]_.

    Parameters
    ----------
    half_life : float, default=40.0
        EWMA half-life in observations for the residual variance estimator.

    beta_half_life : float, default=60.0
        EWMA half-life in observations for the beta estimator.

    min_periods : int or None, default=None
        Minimum number of valid returns required for each asset. Until an asset
        reaches this count, its output is NaN. This warm-up period avoids exposing
        early EWMA values before the residual volatility estimate has sufficiently
        converged from its zero initialization. If `None`, defaults to
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
        Last computed EWMA residual volatility. Contains NaN for inactive assets and
        assets that have not reached `min_periods` valid returns.

    Notes
    -----
    NaNs are treated as missing observations. Active assets with missing returns
    keep their previous asset-specific EWMA state; inactive assets output NaN and
    restart their warm-up period when they become active again. Non-missing returns
    must be finite.

    Market returns are computed from the estimation universe (`estimation_mask` of
    :class:`AssetPanel`). If no estimable asset has both finite returns and finite
    `market_cap` at an observation, the market return is undefined and a `ValueError`
    is raised.

    References
    ----------
    .. [1] "The cross-section of volatility and expected returns"
        The Journal of Finance. Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006).

    See Also
    --------
    EWResidualDownsideVolatility : Downside variant using semi-deviation of residuals.

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import EWResidualVolatility
    >>> # Daily data with default half-lives
    >>> descriptor = EWResidualVolatility()
    >>> vol = descriptor.fit_transform(X)
    >>>
    >>> # Weekly data with faster-adapting vol
    >>> descriptor = EWResidualVolatility(half_life=8, beta_half_life=12)
    >>> vol = descriptor.fit_transform(X)
    """

    def __init__(
        self,
        half_life: float = 40.0,
        beta_half_life: float = 60.0,
        min_periods: int | None = None,
        eps: float = 1e-12,
    ):
        super().__init__(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_acceptable_return=None,
            min_periods=min_periods,
            eps=eps,
        )
