"""Synthetic characteristics AssetPanel generator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pandas as pd

from skfolio.containers import MISSING_CATEGORY_CODE, AssetPanel, FieldCategorical
from skfolio.typing import FloatArray

__all__ = ["make_synthetic_characteristics"]

_PERIODS_PER_YEAR = 252

# Generic industry labels used in the synthetic panel, with sampling weights used to
# assign assets to industries (make some industries larger than others).
_INDUSTRY_SAMPLING_WEIGHTS = (
    ("Real Estate", 390.0),
    ("Software", 280.0),
    ("Banks", 270.0),
    ("Energy", 240.0),
    ("Capital Goods", 230.0),
    ("Commercial", 210.0),
    ("Financials", 200.0),
    ("Tech Hardware", 170.0),
    ("Pharma & Biotech", 170.0),
    ("Health Care", 160.0),
    ("Materials", 160.0),
    ("Food & Beverage", 150.0),
    ("Utilities", 150.0),
    ("Retail", 140.0),
    ("Insurance", 140.0),
    ("Semiconductors", 130.0),
)

# Market (global) factor annualized volatility and mean.
_MARKET_ANN_VOL = 0.18
_MARKET_ANN_MEAN = 0.13

# Average industry-factor annualized volatility and its cross-industry dispersion.
_INDUSTRY_ANN_VOL = 0.11
_INDUSTRY_ANN_VOL_SPREAD = 0.04

# Per-asset market-beta dispersion (std of the cross-sectional beta distribution).
_BETA_SPREAD = 0.40

# Cross-sectional dispersion of idiosyncratic volatility and the Student-t tail
# thickness (degrees of freedom) shared by market and idiosyncratic shocks.
_IDIO_LOG_SIGMA = 0.5
_TAIL_DOF = 6.0

# Idiosyncratic returns mix unit-variance components whose weights sum to one, so the
# total idiosyncratic variance is preserved. They include persistent momentum,
# mean-reverting price pressure, a small predictable bearish component and a
# transitory fat-tailed shock for the remaining variance.
_IDIO_MOMENTUM_AUTOCORR = 0.98
_IDIO_MOMENTUM_WEIGHT = 0.005
_IDIO_REVERSAL_AUTOCORR = 0.95
_IDIO_REVERSAL_WEIGHT = 0.3
_BEARISH_SIGNAL_AUTOCORR = 0.99
_BEARISH_SIGNAL_IDIO_VARIANCE_WEIGHT = 0.000625
_BEARISH_SIGNAL_DESCRIPTOR_SENSITIVITY = 0.8

# Explicit style factors recovered by the default descriptors, each given as
# (annualized volatility, annualized mean, AR(1) coefficient). Market beta, momentum
# and reversal are intentionally excluded: beta emerges from the market component and
# momentum/reversal emerge from the realized return path.
_STYLE_FACTORS = {
    "size": (0.033, 0.005, -0.05),
    "value": (0.014, -0.008, 0.01),
    "earnings_yield": (0.017, 0.015, 0.06),
    "profitability": (0.012, 0.002, 0.02),
    "growth": (0.011, -0.001, 0.04),
    "investment": (0.010, 0.000, 0.04),
    "leverage": (0.013, -0.006, 0.11),
    "dividend_yield": (0.012, 0.001, 0.04),
    "liquidity": (0.019, 0.007, -0.01),
    "volatility": (0.031, 0.003, 0.02),
}

# Central value of each characteristic (e.g. book-to-price ~0.34 , share price ~$42).
# Each asset starts from these values and is then shifted by its hidden style traits and
# random noise, so the generated data stays realistic while differing across assets. A
# couple of entries (log_market_cap, price) are size/price levels, not ratios.
_RATIO_MEDIANS = {
    "log_market_cap": 8.5,
    "price": 42.0,
    "book_to_price": 0.34,
    "earnings_to_price": 0.05,
    "forward_earnings_to_price": 0.06,
    "sales_to_price": 0.45,
    "cash_flow_to_price": 0.07,
    "gross_margin": 0.36,
    "ebitda_margin": 0.18,
    "asset_turnover": 0.57,
    "market_leverage": 0.19,
    "sales_growth": 0.065,
    "capex_intensity": 0.025,
    "daily_turnover": 0.008,
    "dividend_yield": 0.02,
    "forward_dividend_yield": 0.017,
    "short_interest_ratio": 0.02,
    "eps_dispersion_ratio": 0.06,
    "cash_to_assets": 0.10,
}

# Per-field finite fraction among active observations (mirrors realistic coverage).
_FIELD_COVERAGE = {
    "ebitda_ttm": 0.90,
    "enterprise_value": 0.90,
    "cost_of_revenue_ttm": 0.90,
    "eps_ntm": 0.80,
    "dps_ntm": 0.80,
    "eps_ntm_std": 0.80,
}

# Fields whose values are always populated on active cells.
_PROTECTED_FIELDS = frozenset(
    {"returns", "adj_close", "adj_volume", "adj_shares_outstanding", "market_cap"}
)


def make_synthetic_characteristics(
    n_assets: int = 500,
    n_observations: int = 2520,
    *,
    n_industries: int = 10,
    start_date: str = "2015-01-01",
    systematic_variance_ratio: float = 0.5,
    late_listing_proba: float = 0.15,
    delisting_proba: float = 0.15,
    missing_ratio: float = 0.01,
    random_state: int | None = None,
) -> AssetPanel:
    r"""Generate a synthetic characteristics :class:`~skfolio.containers.AssetPanel`.

    The panel generated contains the minimal set of fields required by the default
    `skfolio` descriptors and :class:`~skfolio.prior.CharacteristicsFactorModel`. It is
    designed so that fitting a characteristics factor model produces realistic
    diagnostics: a cross-sectional regression :math:`R^2` away from the degenerate
    values of :math:`0` and :math:`1`, non-trivial information coefficients and
    idiosyncratic returns with fat tails.

    Returns are drawn from a factor structure

    .. math::

        r_{i,t} = \beta_i\,f^{\mathrm{mkt}}_t + f^{\mathrm{ind}(i)}_t
                  + \sum_k B_{i,k}\,f^{k}_t + \varepsilon_{i,t},

    where the per-asset loadings :math:`B` are persistent traits. Characteristics are
    then constructed so that the descriptor of each style is a noisy proxy of the
    corresponding loading, while accounting identities (for example
    :math:`\text{market\_cap} = \text{adj\_close} \times
    \text{adj\_shares\_outstanding}`) are preserved.

    Fields produced:
    `returns`, `adj_close`, `adj_volume`, `adj_shares_outstanding`, `market_cap`,
    `ebitda_ttm`, `enterprise_value`, `net_income_ttm`, `sales_ttm`, `dividends_ttm`,
    `net_buybacks_ttm`, `book_equity`, `operating_cash_flow_ttm`, `total_debt`,
    `total_assets`, `industry`, `cost_of_revenue_ttm`, `capex_ttm`, `short_interest`,
    `eps_ntm`, `dps_ntm`, `eps_ntm_std`.

    Parameters
    ----------
    n_assets : int, default=500
        Number of assets (coverage universe).

    n_observations : int, default=2520
        Number of observations.

    n_industries : int, default=10
        Number of industry groups. Must not exceed 16.

    start_date : str, default="2015-01-01"
        First observation date. Observations follow a business-day calendar.

    systematic_variance_ratio : float, default=0.5
        Share of cross-sectional return variance explained by the factor structure.
        The realized cross-sectional regression :math:`R^2` of a fitted model is close
        to this value. Must lie in the open interval :math:`(0, 1)`.

    late_listing_proba : float, default=0.15
        Probability that an asset lists after the first observation.

    delisting_proba : float, default=0.15
        Probability that an asset delists before the last observation.

    missing_ratio : float, default=0.01
        Fraction of active fundamental observations set to NaN to emulate reporting
        gaps. Price, volume, shares and market cap are left intact.

    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    panel : AssetPanel
        Synthetic asset panel with the fields listed above. `industry` is a
        :class:`~skfolio.containers.FieldCategorical`.

    Notes
    -----
    The generator is driven by a small set of time-invariant latent asset traits
    (size, value, quality, risk, growth and liquidity). These traits set the factor
    loadings :math:`B`, the market beta and the idiosyncratic volatility level and
    anchor the level of every fundamental and market field so that accounting
    identities (such as :math:`\text{market\_cap} = \text{adj\_close} \times
    \text{adj\_shares\_outstanding}`) hold. Each characteristic is therefore a noisy
    proxy of the trait that drives its matching style factor.

    Factor returns combine a fat-tailed market factor, zero-mean industry factors and
    mean-reverting style factors. Idiosyncratic returns mix a transitory shock with a
    slow persistent component and a fast mean-reverting component. These give the
    momentum and short-term reversal factors a realistic positive Sharpe without
    changing the idiosyncratic variance.

    To support alpha-research examples, the idiosyncratic shock also contains a small
    predictable component driven by a persistent latent bearish signal
    :math:`z_{i,t}`. Short interest and analyst forecast dispersion are constructed as
    noisy increasing functions of :math:`z_{i,t}`, while the next-period return
    contribution is

    .. math::

        \varepsilon^{\mathrm{signal}}_{i,t+1}
        = -\sigma_i\sqrt{w_{\mathrm{signal}}}\,z_{i,t}.

    Consequently, high values of either descriptor predict lower future
    idiosyncratic returns without same-period look-ahead.

    Forward-looking and lower-coverage fields (`eps_ntm`, `dps_ntm`, `eps_ntm_std`,
    `enterprise_value`, `ebitda_ttm` and `cost_of_revenue_ttm`) carry partial coverage
    to mirror real data, while price, volume, shares and market cap are always
    populated on active assets.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> panel = make_synthetic_characteristics(n_assets=200, n_observations=1000)
    >>> panel.n_assets, panel.n_observations
    (200, 1000)
    """
    if not 1 <= n_industries <= len(_INDUSTRY_SAMPLING_WEIGHTS):
        raise ValueError(
            f"n_industries must be between 1 and {len(_INDUSTRY_SAMPLING_WEIGHTS)}, "
            f"got {n_industries}."
        )
    if not 0.0 < systematic_variance_ratio < 1.0:
        raise ValueError(
            "systematic_variance_ratio must be in the open interval (0, 1)."
        )

    rng = np.random.default_rng(random_state)
    n_observations, n_assets = int(n_observations), int(n_assets)
    dates = pd.bdate_range(start=start_date, periods=n_observations)
    asset_names = np.array([f"A{i:05d}" for i in range(n_assets)])

    # Industry assignment, weighted by relative membership.
    labels, weights = zip(*_INDUSTRY_SAMPLING_WEIGHTS[:n_industries], strict=True)
    industry_labels = np.array(labels)
    industry_weights = np.array(weights, dtype=float)
    industry_weights /= industry_weights.sum()
    industry_index = rng.choice(n_industries, size=n_assets, p=industry_weights)

    # Latent persistent asset traits (independent standard normals).
    trait_size = rng.standard_normal(n_assets)
    trait_value = rng.standard_normal(n_assets)
    trait_quality = rng.standard_normal(n_assets)
    trait_risk = rng.standard_normal(n_assets)
    trait_growth = rng.standard_normal(n_assets)
    trait_liquidity = rng.standard_normal(n_assets)

    def noise(scale: float) -> FloatArray:
        return scale * rng.standard_normal(n_assets)

    # Per-asset characteristic ratios derived from the latent traits.
    log_market_cap = _RATIO_MEDIANS["log_market_cap"] + 1.4 * trait_size
    base_market_cap = np.exp(log_market_cap)
    base_price = np.exp(np.log(_RATIO_MEDIANS["price"]) + noise(0.85))
    base_shares = base_market_cap / base_price

    book_to_price = np.exp(
        np.log(_RATIO_MEDIANS["book_to_price"]) + 0.55 * trait_value + noise(0.45)
    )
    earnings_to_price = (
        _RATIO_MEDIANS["earnings_to_price"]
        + 0.045 * (0.6 * trait_value + 0.6 * trait_quality)
        + noise(0.05)
    )
    forward_earnings_to_price = np.clip(
        _RATIO_MEDIANS["forward_earnings_to_price"]
        + 0.04 * trait_value
        + 0.03 * trait_quality
        + noise(0.03),
        0.003,
        None,
    )
    sales_to_price = np.exp(
        np.log(_RATIO_MEDIANS["sales_to_price"])
        - 0.3 * trait_size
        + 0.5 * trait_value
        + noise(0.5)
    )
    cash_flow_to_price = (
        _RATIO_MEDIANS["cash_flow_to_price"] + 0.05 * trait_quality + noise(0.05)
    )
    gross_margin = np.clip(
        _RATIO_MEDIANS["gross_margin"] + 0.18 * trait_quality + noise(0.08),
        0.03,
        0.95,
    )
    ebitda_margin = np.clip(
        _RATIO_MEDIANS["ebitda_margin"] + 0.08 * trait_quality + noise(0.05),
        -0.05,
        0.6,
    )
    asset_turnover = np.exp(
        np.log(_RATIO_MEDIANS["asset_turnover"]) - 0.3 * trait_quality + noise(0.4)
    )
    market_leverage = np.clip(
        _RATIO_MEDIANS["market_leverage"]
        + 0.13 * trait_value
        - 0.05 * trait_quality
        + noise(0.08),
        0.0,
        0.85,
    )

    # A fraction of firms carry no debt.
    market_leverage[rng.random(n_assets) < 0.18] = 0.0

    sales_growth = _RATIO_MEDIANS["sales_growth"] + 0.12 * trait_growth + noise(0.06)
    capex_intensity = np.clip(
        _RATIO_MEDIANS["capex_intensity"] + 0.012 * trait_growth + noise(0.01),
        0.0,
        0.2,
    )
    base_daily_turnover = np.exp(
        np.log(_RATIO_MEDIANS["daily_turnover"])
        + 0.4 * trait_liquidity
        - 0.2 * trait_size
        + noise(0.4)
    )

    payer_proba = np.clip(0.55 + 0.18 * trait_value + 0.18 * trait_quality, 0.05, 0.95)
    is_payer = rng.random(n_assets) < payer_proba
    dividend_yield = np.where(
        is_payer,
        np.clip(
            _RATIO_MEDIANS["dividend_yield"] + 0.012 * trait_value + noise(0.012),
            0.0,
            0.1,
        ),
        0.0,
    )
    forward_dividend_yield = np.where(
        is_payer,
        np.clip(
            _RATIO_MEDIANS["forward_dividend_yield"] + 0.01 * trait_value + noise(0.01),
            0.0,
            0.1,
        ),
        0.0,
    )
    buyback_yield = 0.01 + 0.02 * trait_quality + noise(0.03)

    # Asset-specific offsets keep the two signal descriptors heterogeneous and noisy.
    short_interest_log_offset = -0.3 * trait_quality + noise(0.4)
    analyst_dispersion_log_offset = noise(0.4)
    cash_to_assets = np.clip(_RATIO_MEDIANS["cash_to_assets"] + noise(0.05), 0.0, 0.6)

    beta = np.clip(1.0 + _BETA_SPREAD * trait_risk, 0.1, 3.0)

    # Cross-sectional idiosyncratic-vol shape (unit-scale lognormal). The absolute
    # level is set later from `systematic_variance_ratio`.
    idio_shape = np.exp(
        0.35 * trait_risk
        - 0.15 * trait_size
        + _IDIO_LOG_SIGMA * rng.standard_normal(n_assets)
    )

    # Exposure-aligned loadings (standardized) for the explicit style factors.
    gross_profitability = gross_margin * asset_turnover
    loadings = np.column_stack(
        [
            _standardize(log_market_cap),  # size
            _standardize(np.log(book_to_price)),  # value
            _standardize(earnings_to_price),  # earnings_yield
            _standardize(gross_profitability),  # profitability
            _standardize(sales_growth),  # growth
            _standardize(capex_intensity),  # investment
            _standardize(market_leverage),  # leverage
            _standardize(dividend_yield),  # dividend_yield
            _standardize(np.log(base_daily_turnover)),  # liquidity
            _standardize(np.log(idio_shape)),  # volatility
        ]
    )

    # Factor returns: styles and industries are Gaussian AR(1) and the market has fat
    # tails
    style_ann_vol, style_ann_mean, style_autocorr = (
        np.array(values) for values in zip(*_STYLE_FACTORS.values(), strict=True)
    )
    style_returns = _ar1_paths(
        rng, n_observations, style_ann_vol, style_ann_mean, style_autocorr
    )

    market_daily_vol = _MARKET_ANN_VOL / np.sqrt(_PERIODS_PER_YEAR)
    market_daily_mean = _MARKET_ANN_MEAN / _PERIODS_PER_YEAR
    market_returns = market_daily_mean + market_daily_vol * _fat_tailed_normal(
        rng, (n_observations,), _TAIL_DOF
    )

    industry_ann_vol = np.abs(
        _INDUSTRY_ANN_VOL + _INDUSTRY_ANN_VOL_SPREAD * rng.standard_normal(n_industries)
    )
    industry_returns = _ar1_paths(
        rng,
        n_observations,
        industry_ann_vol,
        np.zeros(n_industries),
        np.zeros(n_industries),
    )

    systematic = (
        beta[None, :] * market_returns[:, None]
        + industry_returns[:, industry_index]
        + style_returns @ loadings.T
    )

    # Scale idiosyncratic returns to hit the target systematic-variance share.
    systematic_var = systematic.var(axis=1).mean()
    idio_var = (
        systematic_var * (1.0 - systematic_variance_ratio) / systematic_variance_ratio
    )
    idio_daily_vol = np.sqrt(idio_var) * idio_shape / np.sqrt(np.mean(idio_shape**2))

    # Allocate the idiosyncratic variance budget across transitory, momentum,
    # reversal and predictable bearish components. The reversal component is the
    # unit-variance first difference of a persistent AR(1) process.
    transitory_component = _fat_tailed_normal(
        rng, (n_observations, n_assets), _TAIL_DOF
    )
    momentum_component = _ar1_filter(
        rng.standard_normal((n_observations, n_assets)), _IDIO_MOMENTUM_AUTOCORR
    )
    pressure_level = _ar1_filter(
        rng.standard_normal((n_observations, n_assets)), _IDIO_REVERSAL_AUTOCORR
    )
    reversal_component = np.zeros_like(pressure_level)
    reversal_component[1:] = (pressure_level[1:] - pressure_level[:-1]) / np.sqrt(
        2.0 * (1.0 - _IDIO_REVERSAL_AUTOCORR)
    )

    # A spawned stream keeps the signal reproducible without consuming draws from the
    # main stream and shifting the calibrated factor paths.
    signal_seed = np.random.SeedSequence(random_state).spawn(1)[0]
    signal_rng = np.random.default_rng(signal_seed)
    latent_bearish_signal = _ar1_filter(
        signal_rng.standard_normal((n_observations, n_assets)),
        _BEARISH_SIGNAL_AUTOCORR,
    )
    # Cross-sectional normalization gives the sensitivity and variance weight stable
    # meanings across dates and universe sizes.
    latent_bearish_signal -= latent_bearish_signal.mean(axis=1, keepdims=True)
    latent_bearish_signal_std = latent_bearish_signal.std(axis=1, keepdims=True)
    latent_bearish_signal /= np.where(
        latent_bearish_signal_std == 0.0, 1.0, latent_bearish_signal_std
    )

    idio_shock = (
        np.sqrt(
            1.0
            - _IDIO_MOMENTUM_WEIGHT
            - _IDIO_REVERSAL_WEIGHT
            - _BEARISH_SIGNAL_IDIO_VARIANCE_WEIGHT
        )
        * transitory_component
        + np.sqrt(_IDIO_MOMENTUM_WEIGHT) * momentum_component
        + np.sqrt(_IDIO_REVERSAL_WEIGHT) * reversal_component
    )
    # The one-period lag makes observable descriptors predict future returns.
    idio_shock[1:] -= (
        np.sqrt(_BEARISH_SIGNAL_IDIO_VARIANCE_WEIGHT) * latent_bearish_signal[:-1]
    )
    idio = idio_daily_vol[None, :] * idio_shock

    returns = np.maximum(systematic + idio, -0.99)

    # Listing and delisting windows define the active mask.
    list_start = np.zeros(n_assets, dtype=int)
    list_end = np.full(n_assets, n_observations, dtype=int)
    is_late_listing = rng.random(n_assets) < late_listing_proba
    list_start[is_late_listing] = rng.integers(
        1, max(2, int(0.6 * n_observations)), size=int(is_late_listing.sum())
    )
    is_delisted = rng.random(n_assets) < delisting_proba
    if is_delisted.any():
        earliest_delisting = np.clip(
            list_start[is_delisted] + _PERIODS_PER_YEAR, 1, n_observations
        )
        list_end[is_delisted] = rng.integers(earliest_delisting, n_observations + 1)
    obs_index = np.arange(n_observations)[:, None]
    active = (obs_index >= list_start[None, :]) & (obs_index < list_end[None, :])

    # Guarantee every observation has at least one active asset.
    active[:, 0] = True

    # Price, shares, volume and market cap (identity-consistent).
    adj_close = base_price[None, :] * np.exp(
        np.cumsum(np.log1p(np.where(active, returns, 0.0)), axis=0)
    )
    issuance_drift = (np.exp(sales_growth) ** (1.0 / _PERIODS_PER_YEAR) - 1.0) * 0.3
    share_steps = issuance_drift[None, :] + 5e-5 * rng.standard_normal(
        (n_observations, n_assets)
    )
    shares = base_shares[None, :] * np.exp(
        np.cumsum(np.where(active, share_steps, 0.0), axis=0)
    )
    market_cap = adj_close * shares

    # Shares outstanding are in millions. Traded and shorted shares are raw counts.
    turnover_noise = np.exp(0.4 * rng.standard_normal((n_observations, n_assets)))
    adj_volume = shares * 1e6 * base_daily_turnover[None, :] * turnover_noise

    # Fundamentals grow at each asset's expected return so valuation ratios are
    # stationary in expectation, plus a zero-median tilt for the growth factor.
    style_daily_mean = style_ann_mean / _PERIODS_PER_YEAR
    fundamental_drift = (
        beta * market_daily_mean
        + loadings @ style_daily_mean
        + issuance_drift
        + (sales_growth - np.median(sales_growth)) / _PERIODS_PER_YEAR
    )
    growth_path = np.exp(
        np.arange(n_observations)[:, None] * fundamental_drift[None, :]
    )

    sales = (sales_to_price * base_market_cap)[None, :] * growth_path
    net_income = (earnings_to_price * base_market_cap)[None, :] * growth_path
    book_equity = (book_to_price * base_market_cap)[None, :] * growth_path
    operating_cash_flow = (cash_flow_to_price * base_market_cap)[None, :] * growth_path
    total_debt = (market_leverage * base_market_cap)[None, :] * growth_path
    total_assets = sales / asset_turnover[None, :]
    cost_of_revenue = (1.0 - gross_margin)[None, :] * sales
    ebitda = ebitda_margin[None, :] * sales
    capex = capex_intensity[None, :] * total_assets
    dividends = (dividend_yield * base_market_cap)[None, :] * growth_path
    net_buybacks = (buyback_yield * base_market_cap)[None, :] * growth_path
    cash = cash_to_assets[None, :] * total_assets
    enterprise_value = market_cap + total_debt - cash

    eps_ntm = forward_earnings_to_price[None, :] * adj_close
    dps_ntm = forward_dividend_yield[None, :] * adj_close

    # Both descriptors observe the same bearish signal through independent noisy
    # offsets. Sensitivity controls their signal-to-noise ratio in log space.
    eps_dispersion_ratio = np.exp(
        np.log(_RATIO_MEDIANS["eps_dispersion_ratio"])
        + analyst_dispersion_log_offset[None, :]
        + _BEARISH_SIGNAL_DESCRIPTOR_SENSITIVITY * latent_bearish_signal
    )
    short_interest_ratio = np.clip(
        np.exp(
            np.log(_RATIO_MEDIANS["short_interest_ratio"])
            + short_interest_log_offset[None, :]
            + _BEARISH_SIGNAL_DESCRIPTOR_SENSITIVITY * latent_bearish_signal
        ),
        0.0,
        0.4,
    )
    eps_ntm_std = eps_dispersion_ratio * np.abs(eps_ntm)
    short_interest = short_interest_ratio * shares * 1e6

    fields = {
        "returns": returns,
        "adj_close": adj_close,
        "adj_volume": adj_volume,
        "adj_shares_outstanding": shares,
        "market_cap": market_cap,
        "ebitda_ttm": ebitda,
        "enterprise_value": enterprise_value,
        "net_income_ttm": net_income,
        "sales_ttm": sales,
        "dividends_ttm": dividends,
        "net_buybacks_ttm": net_buybacks,
        "book_equity": book_equity,
        "operating_cash_flow_ttm": operating_cash_flow,
        "total_debt": total_debt,
        "total_assets": total_assets,
        "cost_of_revenue_ttm": cost_of_revenue,
        "capex_ttm": capex,
        "short_interest": short_interest,
        "eps_ntm": eps_ntm,
        "dps_ntm": dps_ntm,
        "eps_ntm_std": eps_ntm_std,
    }

    # Apply coverage-driven missingness then enforce the inactive asset NaN rule.
    inactive = ~active
    for name, array in fields.items():
        if name not in _PROTECTED_FIELDS:
            keep = _FIELD_COVERAGE.get(name, 1.0 - missing_ratio)
            if keep < 1.0:
                array[(rng.random((n_observations, n_assets)) > keep) & active] = np.nan
        array[inactive] = np.nan
        fields[name] = array.astype(np.float32)

    # Industry is constant over time per asset and missing outside the active universe.
    industry_codes = np.broadcast_to(
        industry_index.astype(np.int32)[None, :], (n_observations, n_assets)
    ).copy()
    industry_codes[inactive] = MISSING_CATEGORY_CODE
    fields["industry"] = FieldCategorical(industry_codes, levels=industry_labels)

    # Estimation mask drops a small set of assets.
    estimation_mask = active.copy()
    n_excluded = max(1, int(0.05 * n_assets))
    excluded_index = rng.choice(
        np.arange(1, n_assets), size=min(n_excluded, n_assets - 1), replace=False
    )
    estimation_mask[:, excluded_index] = False

    return AssetPanel(
        fields=fields,
        observations=dates.values,
        asset_names=asset_names,
        active_mask=active,
        estimation_mask=estimation_mask,
    )


def _standardize(x: FloatArray) -> FloatArray:
    """Standardize to zero mean and unit standard deviation."""
    std = x.std()
    if std == 0:
        return x - x.mean()
    return (x - x.mean()) / std


def _ar1_paths(
    rng: np.random.Generator,
    n_observations: int,
    ann_vol: FloatArray,
    ann_mean: FloatArray,
    autocorr: FloatArray,
) -> FloatArray:
    """Simulate stationary AR(1) factor-return paths.

    Parameters are annualized and converted to per-period units. The innovation scale
    is set so the stationary volatility matches `ann_vol` regardless of `autocorr`.
    """
    daily_vol = ann_vol / np.sqrt(_PERIODS_PER_YEAR)
    daily_mean = ann_mean / _PERIODS_PER_YEAR
    n_factors = len(ann_vol)
    innovation = rng.standard_normal((n_observations, n_factors))
    innovation *= daily_vol * np.sqrt(1.0 - autocorr**2)
    paths = np.empty((n_observations, n_factors))
    paths[0] = daily_mean + daily_vol * rng.standard_normal(n_factors)
    for t in range(1, n_observations):
        paths[t] = daily_mean + autocorr * (paths[t - 1] - daily_mean) + innovation[t]
    return paths


def _fat_tailed_normal(
    rng: np.random.Generator, shape: tuple[int, ...], dof: float
) -> FloatArray:
    """Draw unit-variance shocks from a Student-t scale mixture of normals."""
    z = rng.standard_normal(shape)
    scale = np.sqrt(dof / rng.chisquare(dof, size=shape))
    # Rescale to unit variance (a Student-t variable has variance dof / (dof - 2)).
    return z * scale * np.sqrt((dof - 2.0) / dof)


def _ar1_filter(innovations: FloatArray, autocorr: float) -> FloatArray:
    r"""Apply an AR(1) recursion along axis 0, preserving unit stationary variance.

    Unit-variance `innovations` are turned into a unit-variance AR(1) process by
    scaling the innovation term by :math:`\sqrt{1 - \rho^2}`.
    """
    scale = np.sqrt(1.0 - autocorr**2)
    out = np.empty_like(innovations)
    out[0] = innovations[0]
    for t in range(1, len(innovations)):
        out[t] = autocorr * out[t - 1] + scale * innovations[t]
    return out
