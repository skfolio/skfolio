"""EWMA macro sensitivity descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.utils.metadata_routing as skm

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.descriptor._utils import (
    _EWRegression,
    _ReturnAverager,
    _warn_deprecated_eps,
)
from skfolio.typing import ArrayLike, FloatArray
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
    external reference series (e.g. FX, rates, inflation, commodity basket) after
    removing the linear exposure to market returns in a bivariate EWMA regression:

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
        Number of consecutive observations to aggregate before updating EWMA statistics.
        Asset, market and `reference_returns` are averaged over the same observations,
        as described in Notes.

    min_periods : int, optional
        Minimum number of observations required before computing each asset's macro
        sensitivity. With aggregation enabled, each average counts as one observation.
        Until this count is reached, the asset's output is NaN. If `None`, defaults to
        :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    eps : float, optional
        Deprecated and ignored. It will be removed in version 2.0.
        Supplying a positive value emits a `FutureWarning`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    macro_sensitivity_ : ndarray of shape (n_assets,)
        Last fitted partial beta to the reference series for each asset.

    Notes
    -----
    NaNs are allowed as missing observations. Non-missing `returns` and
    `reference_returns` values must be finite. Each asset's regression uses the same
    observations for asset, market and reference returns, with `X.active_mask=True` and
    no missing returns. Missing returns do not reset the regression. Setting
    `X.active_mask=False` resets the regression and sets output to NaN.

    Too little variation in market returns or `reference_returns`, or almost perfect
    correlation between them, produces NaN. An exactly constant market is handled by
    regressing asset returns on `reference_returns` alone with an intercept.

    Market returns are the `X["market_cap"]`-weighted average of `X["returns"]` where
    `X.estimation_mask=True` and both fields are finite. A `ValueError` is raised if
    these assets have no positive total market capitalization at any observation.

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
    """

    def __init__(
        self,
        half_life: float = 60.0,
        aggregation_period: int = 1,
        min_periods: int | None = None,
        eps: float | None = None,
    ) -> None:
        self.half_life = half_life
        self.aggregation_period = aggregation_period
        self.min_periods = min_periods
        self.eps = eps

    def get_metadata_routing(self) -> skm.MetadataRequest:
        """Return metadata routing for the external reference series."""
        request = skm.MetadataRequest(owner=self.__class__.__name__)
        # AssetPanel transformers route fit_transform metadata through the fit bucket,
        # and partial_fit_transform metadata through the partial_fit bucket.
        # `True` requests `reference_returns` under the same name by default.
        request.fit.add_request(param="reference_returns", alias=True)
        request.partial_fit.add_request(param="reference_returns", alias=True)
        return request

    def fit_transform(
        self,
        X: AssetPanel,
        y: None = None,
        reference_returns: ArrayLike | None = None,
        **fit_params: Any,
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
        self,
        X: AssetPanel,
        y: None = None,
        reference_returns: ArrayLike | None = None,
        **fit_params: Any,
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
            Partial beta to the reference series for each observation and asset.
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
                "reference_returns must be a 1D array aligned with X observations. "
                f"Got shape {reference_returns.shape}."
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

        for t in range(n_observations):
            active_mask = X.active_mask[t]
            # Reset assets with active_mask=False even if reference_returns is NaN.
            self._regression.reset(~active_mask)
            self._ref_betas[~active_mask] = np.nan

            returns = np.vstack(
                (
                    np.full(n_assets, market_rets[t]),
                    np.full(n_assets, reference_returns[t]),
                    asset_rets[t],
                )
            )
            aggregated = self._averager.update(returns=returns, active_mask=active_mask)
            if aggregated is not None:
                averaged_returns, valid_mask = aggregated
                self._regression.update(returns=averaged_returns, valid_mask=valid_mask)
                self._ref_betas = self._regression.beta(min_periods=self._min_periods)

            result[t] = self._ref_betas

        self.macro_sensitivity_ = (
            result[-1].copy() if n_observations > 1 else result[-1]
        )

        return result

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_integer(self.aggregation_period, "aggregation_period")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _warn_deprecated_eps(self.eps)

    def _initialize(self) -> None:
        """Initialize regression statistics and return averaging."""
        self._min_periods = (
            max(1, int(np.ceil(self.half_life)))
            if self.min_periods is None
            else int(self.min_periods)
        )
        self._regression = _EWRegression(
            n_assets=self.n_assets_,
            decay=half_life_to_decay_factor(self.half_life),
            include_macro_returns=True,
        )
        # Reuse the last sensitivities while collecting returns for the next average.
        self._ref_betas = np.full(self.n_assets_, np.nan)
        self._averager = _ReturnAverager(
            n_assets=self.n_assets_,
            n_variables=3,
            aggregation_period=int(self.aggregation_period),
        )
