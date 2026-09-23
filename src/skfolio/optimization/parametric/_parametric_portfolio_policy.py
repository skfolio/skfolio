"""Parametric Portfolio Policy optimization estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio._constants import _BENCHMARK_WEIGHTS, _MARKET_CAP
from skfolio.containers import AssetPanel, InactivePolicy
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.optimization._base import BaseOptimization
from skfolio.typing import BoolArray, FloatArray, ObjArray
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_non_negative_real,
    _validate_positive_real,
)
from skfolio.utils.validation import validate_asset_panel

__all__ = ["ParametricPortfolioPolicy"]


class ParametricPortfolioPolicy(BaseOptimization):
    r"""Parametric Portfolio Policy estimator.

    Maps asset characteristics directly to portfolio weights in one step, following
    Brandt, Santa-Clara and Valkanov (2009) [1]_. Instead of first estimating expected
    returns and a covariance matrix and then optimizing, the weights are parametrized
    as a linear function of the characteristics and the coefficients are chosen to
    maximize the investor's realized in-sample utility.

    For each observation :math:`t` and asset :math:`i` eligible at :math:`t`, the
    policy weight is

    .. math::

        w_{t,i} = b_{t,i} + \frac{1}{N_t} \theta^\top \tilde{z}_{t,i}

    where :math:`b_{t,i}` are the benchmark weights, :math:`N_t` is the number of
    eligible assets at :math:`t`, :math:`\theta` is the vector of policy coefficients
    and :math:`\tilde{z}_{t,i}` are the characteristics demeaned cross-sectionally over
    the eligible assets, so that the deviation from the benchmark is self-financing and
    the weights sum to one.

    The coefficients :math:`\theta` maximize the average utility of the realized
    portfolio return path

    .. math::

        \max_{\theta} \; \frac{1}{T} \sum_{t} u\left(\sum_{i} w_{t,i} \, r_{t+1,i}\right)

    with the constant relative risk aversion (CRRA) utility

    .. math::

        u(r) = \frac{(1 + r)^{1 - \gamma}}{1 - \gamma}

    (and :math:`u(r) = \log(1 + r)` when :math:`\gamma = 1`), where :math:`\gamma` is
    `risk_aversion`. The weights at :math:`t` use the characteristics known at
    :math:`t` and are evaluated on the returns of :math:`t + 1` (as-of convention), so
    the objective uses only information available at the time of each decision.

    Because the portfolio return is linear in :math:`\theta` and the utility is
    concave, the objective is concave in :math:`\theta`. It is maximized with a damped
    Newton method using the analytic gradient and Hessian, with a backtracking line
    search that keeps :math:`1 + r > 0` at every step. Starting from
    :math:`\theta = 0`, which corresponds to holding the benchmark, the method converges
    to the global optimum.

    The characteristics are built with the same factor exposure estimators as
    :class:`~skfolio.prior.CharacteristicsFactorModel`
    (:class:`~skfolio.factor_exposure.FixedWeightedFactor`,
    :class:`~skfolio.factor_exposure.DerivedFactor`,
    :class:`~skfolio.factor_exposure.OneHotCategoricalFactors`, ...), including their
    cross-sectional outlier and scoring transformations. Multi-factor estimators
    contribute one coefficient per output column.

    Parameters
    ----------
    characteristics_exposures : list[tuple[str, BaseFactorExposure]]
        Characteristics as a list of `(name, estimator)` tuples, where each estimator
        is a :class:`~skfolio.factor_exposure.BaseFactorExposure` computed on the
        `characteristics` panel passed to `fit`. Single-factor estimators contribute a
        coefficient named after the tuple name; multi-factor estimators contribute one
        coefficient per entry of their `factor_names_`.

    risk_aversion : float, default=5.0
        Relative risk aversion :math:`\gamma > 0` of the CRRA utility. The default of
        5 follows the original paper. A value of 1 gives log utility.

    benchmark_mcap_power : float, default=1.0
        Exponent :math:`p` applied to market capitalizations when building the
        benchmark weights :math:`b_{t,i} \propto \text{mcap}_{t,i}^{p}` over the
        eligible assets. `1` gives value weights (as in the original paper), `0` gives
        equal weights and values in between shrink cap concentration. When different
        from `0`, the `characteristics` panel must contain a `market_cap` field.

    max_iter : int, default=100
        Maximum number of Newton iterations.

    tol : float, default=1e-10
        Convergence tolerance on the Newton decrement of the average utility.

    portfolio_params : dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

    fallback : BaseOptimization | "previous_weights" | list[BaseOptimization | "previous_weights"], optional
        Fallback estimator or a list of estimators to try, in order, when the primary
        optimization raises during `fit`. Alternatively, use `"previous_weights"`
        (alone or in a list) to fall back to the estimator's `previous_weights`.
        When a fallback succeeds, its fitted `weights_` are copied back to the primary
        estimator so that `fit` still returns the original instance. For traceability,
        the fitted fallback estimator is stored in `fallback_`, and the chain of
        attempts (including errors) is stored in `fallback_chain_`.
        The default (`None`) means no fallback.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Previous weights of the assets. Used only when `fallback="previous_weights"`
        and passed to the portfolio on `predict`.

    raise_on_failure : bool, default=True
        If True, any failure during `fit` is raised immediately, no `weights_` are
        set and the estimator is left unfitted.
        If False, errors are not raised; instead, a warning is emitted, `weights_`
        is set to `None`, the error is stored in `error_`, and `predict` returns a
        `FailedPortfolio`.

    Attributes
    ----------
    coef_ : ndarray of shape (n_characteristics,)
        Fitted policy coefficients :math:`\theta`, aligned with
        `characteristic_names_`.

    characteristic_names_ : ndarray of shape (n_characteristics,)
        Names of the characteristics associated with each coefficient.

    weights_ : ndarray of shape (n_assets,)
        Policy weights at the last observation, computed from the characteristics of
        that observation. Aligned with `feature_names_in_`.

    weights_history_ : ndarray of shape (n_observations, n_assets)
        In-sample policy weights at every observation, aligned with
        `feature_names_in_`. Observations with no eligible asset are NaN.

    utility_ : float
        Average in-sample utility of the fitted policy.

    benchmark_utility_ : float
        Average in-sample utility of the benchmark (:math:`\theta = 0`).

    n_iter_ : int
        Number of Newton iterations run.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has asset names that are all strings.

    Notes
    -----
    An asset is eligible at :math:`t` when it belongs to the investment universe
    (the columns of `X`), is active in the panel at :math:`t`, has finite
    characteristics at :math:`t` and, for the objective, a finite return at
    :math:`t + 1`. Non-eligible assets receive a zero weight. Observations with no
    eligible asset (for example the warm-up of rolling characteristics) are excluded
    from the objective and have NaN rows in `weights_history_`.

    References
    ----------
    .. [1] "Parametric Portfolio Policies: Exploiting Characteristics in the
        Cross-Section of Equity Returns".
        Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009).
        The Review of Financial Studies, 22(9), 3411-3447.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import BookToPrice, EWMomentum, LogMarketCap
    >>> from skfolio.factor_exposure import FixedWeightedFactor
    >>> from skfolio.optimization import ParametricPortfolioPolicy
    >>>
    >>> panel = make_synthetic_characteristics(
    ...     n_assets=100, n_observations=500, random_state=0
    ... )
    >>> X = panel.to_dataframe(fields="returns")
    >>>
    >>> model = ParametricPortfolioPolicy(
    ...     characteristics_exposures=[
    ...         ("size", FixedWeightedFactor(descriptors=[("mcap", LogMarketCap())])),
    ...         ("value", FixedWeightedFactor(descriptors=[("btp", BookToPrice())])),
    ...         ("momentum", FixedWeightedFactor(descriptors=[("mom", EWMomentum())])),
    ...     ],
    ...     risk_aversion=5.0,
    ... )
    >>> model.fit(X, characteristics=panel)
    ParametricPortfolioPolicy(...)
    >>> list(model.characteristic_names_)
    ['size', 'value', 'momentum']
    >>> model.coef_.shape
    (3,)
    >>> portfolio = model.predict(X)
    """

    coef_: FloatArray
    characteristic_names_: ObjArray
    weights_history_: FloatArray
    utility_: float
    benchmark_utility_: float
    n_iter_: int

    def __init__(
        self,
        characteristics_exposures: list[tuple[str, BaseFactorExposure]],
        risk_aversion: float = 5.0,
        benchmark_mcap_power: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-10,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        previous_weights: skt.MultiInput | None = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            previous_weights=previous_weights,
            raise_on_failure=raise_on_failure,
        )
        self.characteristics_exposures = characteristics_exposures
        self.risk_aversion = risk_aversion
        self.benchmark_mcap_power = benchmark_mcap_power
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        characteristics: AssetPanel | None = None,
    ) -> ParametricPortfolioPolicy:
        """Fit the Parametric Portfolio Policy estimator.

        Parameters
        ----------
        X : DataFrame of shape (n_observations, n_assets)
            Returns of the assets in the investment universe. The columns define the
            assets and their order, and must all be present in `characteristics`. The
            index must match `characteristics.observations`.

        y : Ignored
            Not used, present for API consistency by convention.

        characteristics : AssetPanel
            Point-in-time panel of asset characteristics on which the
            `characteristics_exposures` estimators are computed. It must cover at
            least the assets in `X` and contain a `market_cap` field when
            `benchmark_mcap_power != 0`.

        Returns
        -------
        self : ParametricPortfolioPolicy
            Fitted estimator.
        """
        self._validate_params()

        if characteristics is None:
            raise ValueError("`characteristics` must be provided to fit the policy.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` must be a pd.DataFrame.")

        returns = np.asarray(
            skv.validate_data(self, X, ensure_all_finite=False), dtype=float
        )
        n_observations = returns.shape[0]

        need_market_cap = self.benchmark_mcap_power != 0
        characteristics = validate_asset_panel(
            self,
            asset_panel=characteristics,
            required_fields=[_MARKET_CAP] if need_market_cap else None,
            reserved_fields=[_BENCHMARK_WEIGHTS],
            strictly_positive_when_active=[_MARKET_CAP] if need_market_cap else None,
            copy=True,  # shallow copy: the user's panel is not mutated
        )

        if characteristics.n_observations != n_observations:
            raise ValueError(
                "`X` and `characteristics` must have the same number of observations, "
                f"got {n_observations} and {characteristics.n_observations}."
            )
        if not np.array_equal(np.asarray(X.index), characteristics.observations):
            raise ValueError("`X.index` must match `characteristics.observations`.")

        asset_idx = {name: i for i, name in enumerate(characteristics.asset_names)}
        try:
            investment_idx = np.array([asset_idx[x] for x in X.columns], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"Asset {e.args[0]!r} from `X` is missing from `characteristics`."
            ) from e

        # Benchmark weights on the coverage universe, used by the cross-sectional
        # transformations of the exposure estimators (same convention as
        # `CharacteristicsFactorModel`) and, restricted to the eligible assets, as the
        # benchmark of the policy.
        active_mask = characteristics.active_mask
        benchmark_mask = active_mask & characteristics.estimation_mask
        if need_market_cap:
            market_cap = characteristics[_MARKET_CAP]
            benchmark_mask &= np.isfinite(market_cap)
            cap_weights = np.zeros_like(market_cap, dtype=float)
            cap_weights[benchmark_mask] = np.power(
                market_cap[benchmark_mask], self.benchmark_mcap_power
            )
        else:
            cap_weights = benchmark_mask.astype(float)
        characteristics.add_2d_field(
            name=_BENCHMARK_WEIGHTS,
            values=cap_weights,
            inactive_policy=InactivePolicy.ZERO,
        )

        # Characteristics (n_observations, n_coverage_assets, n_characteristics)
        exposures, names = self._compute_exposures(characteristics)

        # Restrict to the investment universe
        z = exposures[:, investment_idx, :]
        cap_weights = cap_weights[:, investment_idx]
        active_mask = active_mask[:, investment_idx]
        n_characteristics = z.shape[-1]

        # Eligibility for holding a position at t
        holdable = active_mask & np.isfinite(z).all(axis=-1) & (cap_weights > 0)

        # Demean characteristics cross-sectionally over the eligible assets so that
        # the policy tilt is self-financing, and scale by 1 / N_t.
        z_tilde = _demean_and_scale(z, holdable)

        # Benchmark weights normalized over the eligible assets
        benchmark_weights = np.where(holdable, cap_weights, 0.0)
        benchmark_sum = benchmark_weights.sum(axis=1, keepdims=True)
        benchmark_weights = np.divide(
            benchmark_weights,
            benchmark_sum,
            out=np.zeros_like(benchmark_weights),
            where=benchmark_sum > 0,
        )

        # Objective on the realized path: weights at t are evaluated on returns at
        # t + 1. Only pairs with a finite next return contribute.
        next_returns = returns[1:]
        contributes = holdable[:-1] & np.isfinite(next_returns)
        if not np.array_equal(contributes, holdable[:-1]):
            # An eligible asset with a missing next return is dropped from that
            # observation's portfolio: rebuild the tilt and benchmark on the reduced
            # set so that the objective weights stay self-financing.
            z_obj = _demean_and_scale(z[:-1], contributes)
            b_obj = np.where(contributes, cap_weights[:-1], 0.0)
            b_sum = b_obj.sum(axis=1, keepdims=True)
            b_obj = np.divide(b_obj, b_sum, out=np.zeros_like(b_obj), where=b_sum > 0)
        else:
            z_obj = z_tilde[:-1]
            b_obj = benchmark_weights[:-1]

        valid_obs = contributes.any(axis=1)
        if not valid_obs.any():
            raise ValueError(
                "No observation has an eligible asset with finite characteristics and "
                "a finite next-period return. Check the warm-up of the characteristics "
                "estimators and the coverage of `X`."
            )
        r = np.where(contributes, next_returns, 0.0)[valid_obs]
        # Benchmark return path and characteristic-weighted return path
        benchmark_returns = np.einsum("ti,ti->t", b_obj[valid_obs], r)
        tilt_returns = np.einsum("tik,ti->tk", z_obj[valid_obs], r)

        if np.any(1.0 + benchmark_returns <= 0.0):
            raise ValueError(
                "The benchmark portfolio has a return lower than or equal to -100% at "
                "some observation, which is outside the domain of the CRRA utility."
            )

        theta, n_iter, utility, benchmark_utility = _maximize_crra_utility(
            benchmark_returns=benchmark_returns,
            tilt_returns=tilt_returns,
            risk_aversion=self.risk_aversion,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # In-sample weight path and the weights for the next period
        weights_history = benchmark_weights + z_tilde @ theta
        no_holding = ~holdable.any(axis=1)
        weights_history[no_holding] = np.nan
        if no_holding[-1]:
            raise ValueError(
                "No asset is eligible at the last observation: the policy weights "
                "cannot be computed."
            )

        self.coef_ = theta
        self.characteristic_names_ = names
        self.n_characteristics_ = n_characteristics
        self.weights_history_ = weights_history
        self.utility_ = float(utility)
        self.benchmark_utility_ = float(benchmark_utility)
        self.n_iter_ = int(n_iter)
        self.weights_ = weights_history[-1].copy()
        return self

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if (
            self.characteristics_exposures is None
            or len(self.characteristics_exposures) == 0
        ):
            raise ValueError(
                "Invalid 'characteristics_exposures' attribute, it should be a "
                "non-empty list of (string, factor exposure) tuples."
            )
        names = []
        for item in self.characteristics_exposures:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "'characteristics_exposures' items must be (name, estimator) "
                    f"tuples, got {item!r}."
                )
            name, estimator = item
            if not isinstance(name, str):
                raise ValueError(f"Characteristic name must be a string, got {name!r}.")
            if not isinstance(estimator, BaseFactorExposure):
                raise ValueError(
                    f"Characteristic '{name}' must be a BaseFactorExposure estimator, "
                    f"got {type(estimator).__name__}."
                )
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"Characteristic names must be unique, got {names!r}.")
        _validate_positive_real(self.risk_aversion, "risk_aversion")
        _validate_non_negative_real(self.benchmark_mcap_power, "benchmark_mcap_power")
        _validate_non_negative_integer(self.max_iter, "max_iter")
        _validate_positive_real(self.tol, "tol")

    def _compute_exposures(
        self, characteristics: AssetPanel
    ) -> tuple[FloatArray, ObjArray]:
        """Compute and stack the characteristic exposures.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_coverage_assets, n_characteristics)
            Stacked exposures.

        names : ndarray of shape (n_characteristics,)
            Characteristic names, expanded for multi-factor estimators.
        """
        self.characteristics_exposures_ = [
            (name, sk.clone(estimator))
            for name, estimator in self.characteristics_exposures
        ]
        exposures = []
        names = []
        for name, estimator in self.characteristics_exposures_:
            exposure = np.asarray(estimator.fit_transform(characteristics), dtype=float)
            if exposure.ndim == 2:
                exposures.append(exposure[:, :, np.newaxis])
                names.append(name)
            elif exposure.ndim == 3:
                factor_names = getattr(estimator, "factor_names_", None)
                if factor_names is None or len(factor_names) != exposure.shape[-1]:
                    raise ValueError(
                        f"Multi-factor characteristic '{name}' must expose "
                        "`factor_names_` matching its number of output columns."
                    )
                exposures.append(exposure)
                names.extend(str(x) for x in factor_names)
            else:
                raise ValueError(
                    f"Characteristic '{name}' returned an exposure with "
                    f"{exposure.ndim} dimensions; expected 2 or 3."
                )
        return np.concatenate(exposures, axis=-1), np.array(names, dtype=object)


def _demean_and_scale(z: FloatArray, mask: BoolArray) -> FloatArray:
    r"""Demean characteristics cross-sectionally over `mask` and divide by the number
    of masked assets, giving the per-asset tilt :math:`\tilde{z}_{t,i} / N_t`.
    Entries outside `mask` are set to zero.
    """
    counts = mask.sum(axis=1)[:, np.newaxis, np.newaxis]
    masked = np.where(mask[:, :, np.newaxis], z, 0.0)
    means = np.divide(
        masked.sum(axis=1, keepdims=True),
        counts,
        out=np.zeros((z.shape[0], 1, z.shape[2])),
        where=counts > 0,
    )
    tilt = np.where(mask[:, :, np.newaxis], z - means, 0.0)
    return np.divide(tilt, counts, out=np.zeros_like(tilt), where=counts > 0)


def _crra(
    returns: FloatArray, risk_aversion: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the CRRA utility and its first two derivatives at `returns`."""
    wealth = 1.0 + returns
    if risk_aversion == 1.0:
        u = np.log(wealth)
    else:
        u = np.power(wealth, 1.0 - risk_aversion) / (1.0 - risk_aversion)
    du = np.power(wealth, -risk_aversion)
    d2u = -risk_aversion * np.power(wealth, -risk_aversion - 1.0)
    return u, du, d2u


def _maximize_crra_utility(
    benchmark_returns: FloatArray,
    tilt_returns: FloatArray,
    risk_aversion: float,
    max_iter: int,
    tol: float,
) -> tuple[FloatArray, int, float, float]:
    """Maximize the average CRRA utility of `benchmark_returns + tilt_returns @ theta`.

    The objective is concave in `theta`. It is maximized with a damped Newton method
    using the analytic gradient and Hessian, and a backtracking line search that keeps
    the portfolio wealth positive and enforces sufficient increase.

    Parameters
    ----------
    benchmark_returns : ndarray of shape (n_observations,)
        Realized benchmark returns.

    tilt_returns : ndarray of shape (n_observations, n_characteristics)
        Realized returns of the unit characteristic tilts.

    risk_aversion : float
        Relative risk aversion of the CRRA utility.

    max_iter : int
        Maximum number of Newton iterations.

    tol : float
        Tolerance on the Newton decrement.

    Returns
    -------
    theta : ndarray of shape (n_characteristics,)
        Optimal coefficients.

    n_iter : int
        Number of iterations run.

    utility : float
        Average utility at `theta`.

    benchmark_utility : float
        Average utility at `theta = 0`.
    """
    n_characteristics = tilt_returns.shape[1]
    theta = np.zeros(n_characteristics)

    def objective(t: FloatArray) -> float:
        portfolio_returns = benchmark_returns + tilt_returns @ t
        if np.any(1.0 + portfolio_returns <= 0.0):
            return -np.inf
        return float(np.mean(_crra(portfolio_returns, risk_aversion)[0]))

    benchmark_utility = objective(theta)
    utility = benchmark_utility
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        portfolio_returns = benchmark_returns + tilt_returns @ theta
        _, du, d2u = _crra(portfolio_returns, risk_aversion)
        gradient = tilt_returns.T @ du / len(du)
        hessian = (tilt_returns * d2u[:, np.newaxis]).T @ tilt_returns / len(du)
        try:
            direction = np.linalg.solve(-hessian, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(-hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(direction)) or gradient @ direction <= 0:
            # Not an ascent direction (numerically singular Hessian): use the gradient
            direction = gradient
        decrement = gradient @ direction
        if decrement <= tol:
            n_iter -= 1
            break
        # Backtracking line search (Armijo) inside the utility domain
        step = 1.0
        while step > 1e-12:
            candidate = theta + step * direction
            value = objective(candidate)
            if value >= utility + 1e-4 * step * decrement:
                break
            step *= 0.5
        else:
            break
        theta = candidate
        utility = value

    return theta, n_iter, utility, benchmark_utility
