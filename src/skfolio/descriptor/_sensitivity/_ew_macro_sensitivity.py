"""EWMA macro sensitivity descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import numpy as np
import sklearn.utils.metadata_routing as skm

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import _market_returns
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "macro_sensitivity_"


class EWMacroSensitivity(BaseDescriptor):
    r"""EWMA macro sensitivity after removing market exposure.

    The descriptor estimates the partial regression coefficient of asset returns on an
    external reference series (e.g. FX, rates, commodity basket) after removing the
    linear exposure to market returns in a bivariate EWMA regression:

    .. math::

        r_{i,t} = \alpha_i + \beta^M_i\, r_{\text{market},t}
                 + \beta^{\text{ref}}_i\, r_{\text{ref},t}
                 + \varepsilon_{i,t}

    where :math:`r_{i,t}` is the return of asset :math:`i` at time :math:`t`,
    :math:`r_{\text{market},t}` (denoted :math:`r_{m,t}`) is the cap-weighted market
    return computed on the estimation universe (`estimation_mask`) and
    :math:`r_{\text{ref},t}` is the external reference return.

    The output is :math:`\beta^{\text{ref}}_i`, the sensitivity to the reference series
    after removing market exposure.

    The partial beta is computed in closed form via the Frisch-Waugh decomposition,
    using only EWMA moments and no matrix inversion:

    .. math::

        \beta^{\text{ref}}_i =
        \frac{C_{y_i f} - C_{y_i m}\, C_{mf} / V_m}
             {V_f - C_{mf}^2 / V_m}

    where :math:`V_m, V_f` are EWMA variances of market and reference, :math:`C_{mf}` is
    their EWMA covariance and :math:`C_{y_i m}, C_{y_i f}` are the EWMA covariances of
    asset :math:`i` with market and reference respectively.

    Parameters
    ----------
    half_life : float, default=60.0
        EWMA half-life in units of aggregated periods.

    aggregation_period : int, default=1
        Number of consecutive observations to aggregate before updating
        EWMA statistics.

    min_periods : int, optional
        Minimum number of market/reference observations and valid asset returns
        required before computing macro sensitivities. Until both counts reach this
        value, the asset's output is NaN. This warm-up period avoids exposing early
        EWMA values before the sensitivity estimate has sufficiently converged from its
        zero initialization. If `None`, defaults to
        :math:`\lceil\text{half_life}\rceil`, with a minimum of 1.

    eps : float, default=1e-12
        Small constant for numerical stability in denominators.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    macro_sensitivity_ : ndarray of shape (n_assets,)
        Last fitted partial beta to the reference series for each asset.

    References
    ----------
    .. [1] "The share of systematic variation in bilateral exchange rates"
        The Journal of Finance. Verdelhan, A. (2018).

    Notes
    -----
    NaNs are allowed as missing observations. Non-missing `returns` and
    `reference_returns` values must be finite. A missing reference return freezes the
    full EWMA state for that observation or aggregated window. Asset covariances are
    updated only for assets with valid returns, and each asset's valid-observation count
    controls when its output starts.

    Market returns are computed from the estimation universe (`estimation_mask` of
    :class:`AssetPanel`). If no estimable asset has both finite returns and finite
    `market_cap` at an observation, the market return is undefined and a `ValueError` is
    raised.

    See Also
    --------
    EWMarketBeta : Univariate EWMA beta to the market portfolio.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWMacroSensitivity
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> fx_basket = np.random.default_rng(0).standard_normal(X.n_observations)
    >>> rate_returns = np.random.default_rng(1).standard_normal(X.n_observations)
    >>>
    >>> # FX sensitivity (daily updates with default memory)
    >>> descriptor = EWMacroSensitivity()
    >>> macro_sensitivity = descriptor.fit_transform(X, reference_returns=fx_basket)
    >>>
    >>> # Interest rate sensitivity
    >>> descriptor = EWMacroSensitivity()
    >>> macro_sensitivity = descriptor.fit_transform(X, reference_returns=rate_returns)
    """

    def __init__(
        self,
        half_life: float = 60.0,
        aggregation_period: int = 1,
        min_periods: int | None = None,
        eps: float = 1e-12,
    ):
        self.half_life = half_life
        self.aggregation_period = aggregation_period
        self.min_periods = min_periods
        self.eps = eps

    def get_metadata_routing(self):
        """Return metadata routing for the external reference series."""
        request = skm.MetadataRequest(owner=self.__class__.__name__)
        # AssetPanel transformers route fit_transform metadata through the fit bucket,
        # and partial_fit_transform metadata through the partial_fit bucket.
        # `True` requests `reference_returns` under the same name by default.
        request.fit.add_request(param="reference_returns", alias=True)
        request.partial_fit.add_request(param="reference_returns", alias=True)
        return request

    def fit_transform(
        self, X: AssetPanel, y=None, reference_returns=None, **fit_params
    ) -> FloatArray:
        """Compute exponentially weighted macro sensitivities.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        reference_returns : array-like of shape (n_observations,)
            External reference return series aligned with `X` (e.g., macro factor).

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        sensitivities : ndarray of shape (n_observations, n_assets)
            Partial beta to the reference series for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(
            X, y, reference_returns=reference_returns, **fit_params
        )

    def partial_fit_transform(
        self, X: AssetPanel, y=None, reference_returns=None, **fit_params
    ) -> FloatArray:
        """Update EWMA state and return macro sensitivities for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `"returns"` and `"market_cap"`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        reference_returns : array-like of shape (n_observations,)
            External macro reference returns, e.g. FX basket returns.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        sensitivities : ndarray of shape (n_observations, n_assets)
            Partial beta to the reference series after removing market exposure.
        """
        if reference_returns is None:
            raise ValueError(
                "reference_returns must be provided via fit_params, "
                "e.g. fit_transform(X, reference_returns=fx_returns)"
            )

        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["returns", "market_cap"],
            finite_or_nan=["returns"],
            finite_when_active=["market_cap"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        reference_returns = np.asarray(reference_returns, dtype=float)

        if reference_returns.ndim != 1:
            raise ValueError(
                "reference_returns must be a 1D array aligned with X observations; "
                f"got shape {reference_returns.shape}."
            )
        if reference_returns.shape[0] != n_observations:
            raise ValueError(
                f"reference_returns length ({reference_returns.shape[0]}) "
                f"must match n_observations ({n_observations})"
            )
        if np.any(np.isinf(reference_returns)):
            raise ValueError(
                "reference_returns contains infinite values. "
                "EWMacroSensitivity requires finite reference returns or NaN."
            )

        asset_rets = X["returns"]
        market_caps = X["market_cap"]

        market_rets = _market_returns(
            asset_returns=asset_rets,
            weights=market_caps,
            estimation_mask=X.estimation_mask,
        )

        result = np.empty((n_observations, n_assets), dtype=float)

        if self._aggregation_period == 1:
            for t in range(n_observations):
                self._update_ewma(
                    ret_assets=asset_rets[t],
                    ret_market=market_rets[t],
                    ret_ref=reference_returns[t],
                )
                result[t] = self._ref_betas
        else:
            for i in range(n_observations):
                self._buffer_assets[self._buffer_idx] = asset_rets[i]
                self._buffer_market[self._buffer_idx] = market_rets[i]
                self._buffer_ref[self._buffer_idx] = reference_returns[i]
                self._buffer_idx += 1

                if self._buffer_idx >= self._aggregation_period:
                    self._update_from_buffer()

                result[i] = self._ref_betas

        # Mask output for inactive assets
        result = np.where(X.active_mask, result, np.nan)

        self.macro_sensitivity_ = (
            result[-1].copy() if n_observations > 1 else result[-1]
        )

        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_integer(self.aggregation_period, "aggregation_period")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _validate_positive_real(self.eps, "eps")

    def _initialize(self) -> None:
        """Initialize EWMA state and aggregation buffers."""
        n_assets = self.n_assets_
        self._decay = half_life_to_decay_factor(self.half_life)

        if self.min_periods is None:
            self._min_periods = max(1, int(np.ceil(self.half_life)))
        else:
            self._min_periods = int(self.min_periods)

        # EWMA means
        self._mu_market = 0.0
        self._mu_ref = 0.0
        self._mu_assets = np.zeros(n_assets, dtype=float)

        # EWMA variances and covariances
        self._var_market = 0.0
        self._var_ref = 0.0
        self._cov_market_ref = 0.0
        self._cov_assets_market = np.zeros(n_assets, dtype=float)
        self._cov_assets_ref = np.zeros(n_assets, dtype=float)

        # Count of aggregated periods processed
        self._t = 0

        # Current betas, held during incomplete aggregation windows.
        self._ref_betas = np.full(n_assets, np.nan, dtype=float)
        self._market_betas = np.full(n_assets, np.nan, dtype=float)
        self._n_valid_assets = np.zeros(n_assets, dtype=int)

        # Aggregation buffers
        self._aggregation_period = int(self.aggregation_period)
        if self._aggregation_period > 1:
            self._buffer_assets = np.empty(
                (self._aggregation_period, n_assets), dtype=float
            )
            self._buffer_market = np.empty(self._aggregation_period, dtype=float)
            self._buffer_ref = np.empty(self._aggregation_period, dtype=float)
            self._buffer_idx = 0

    def _update_ewma(
        self, ret_assets: FloatArray, ret_market: float, ret_ref: float
    ) -> None:
        r"""Update EWMA statistics and compute partial betas.

        Uses centered EWMA with **lagged-mean deviations**: deviations
        are computed from the *previous* step's means, then means are
        updated. This is the EWMA analogue of Welford's online algorithm
        and avoids the systematic downward bias of the naive approach
        (which computes deviations from the already-updated mean).

        The partial beta to the reference series uses the Frisch-Waugh
        closed form (no matrix inverse):

        .. math::

            \beta^{\text{ref}}_i =
            \frac{C_{yf,i} - C_{ym,i} \cdot C_{mf} / V_m}
                 {V_f - C_{mf}^2 / V_m}

        If `ret_ref` is non-finite, the full EWMA state is frozen and
        this call is skipped.
        """
        if not np.isfinite(ret_ref):
            return

        decay = self._decay
        self._t += 1

        # Deviations from lagged means.
        market_deviation = ret_market - self._mu_market
        ref_deviation = ret_ref - self._mu_ref

        # Update EWMA means.
        self._mu_market = decay * self._mu_market + (1 - decay) * ret_market
        self._mu_ref = decay * self._mu_ref + (1 - decay) * ret_ref

        # Update scalar variances and covariance.
        self._var_market = decay * self._var_market + (1 - decay) * (
            market_deviation * market_deviation
        )
        self._var_ref = decay * self._var_ref + (1 - decay) * (
            ref_deviation * ref_deviation
        )
        self._cov_market_ref = decay * self._cov_market_ref + (1 - decay) * (
            market_deviation * ref_deviation
        )

        # Update per-asset covariances for valid returns.
        valid = np.isfinite(ret_assets)
        self._n_valid_assets[valid] += 1
        asset_deviations = ret_assets[valid] - self._mu_assets[valid]

        self._mu_assets[valid] = (
            decay * self._mu_assets[valid] + (1 - decay) * ret_assets[valid]
        )

        self._cov_assets_market[valid] = (
            decay * self._cov_assets_market[valid]
            + (1 - decay) * asset_deviations * market_deviation
        )
        self._cov_assets_ref[valid] = (
            decay * self._cov_assets_ref[valid]
            + (1 - decay) * asset_deviations * ref_deviation
        )

        # Compute partial betas with Frisch-Waugh closed form.
        if self._t >= self._min_periods:
            market_variance = self._var_market + self.eps
            reference_residual_variance = (
                self._var_ref - self._cov_market_ref**2 / market_variance + self.eps
            )
            asset_ready = valid & (self._n_valid_assets >= self._min_periods)

            self._ref_betas[asset_ready] = (
                self._cov_assets_ref[asset_ready]
                - self._cov_assets_market[asset_ready]
                * self._cov_market_ref
                / market_variance
            ) / reference_residual_variance

            # Market beta from the same bivariate regression
            self._market_betas[asset_ready] = (
                self._cov_assets_market[asset_ready]
                - self._ref_betas[asset_ready] * self._cov_market_ref
            ) / market_variance

    def _update_from_buffer(self) -> None:
        """Aggregate buffered returns and update EWMA statistics."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice", category=RuntimeWarning
            )
            agg_ref = np.nanmean(self._buffer_ref)
            agg_assets = np.nanmean(self._buffer_assets, axis=0)
        agg_market = np.mean(self._buffer_market)

        self._buffer_idx = 0
        self._update_ewma(agg_assets, agg_market, agg_ref)
