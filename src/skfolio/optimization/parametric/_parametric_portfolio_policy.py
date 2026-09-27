"""Parametric Portfolio Policy optimization estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.optimize as sco
import sklearn as sk
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio._constants import _BENCHMARK_WEIGHTS, _MARKET_CAP
from skfolio.containers import AssetPanel, InactivePolicy
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.optimization._base import BaseOptimization
from skfolio.typing import BoolArray, FloatArray, IntArray, ObjArray
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_non_negative_real,
    _validate_positive_real,
    input_to_array,
)
from skfolio.utils.validation import validate_asset_panel

__all__ = ["ParametricPortfolioPolicy"]

_SMOOTH_SOLVER = "trust-exact"
_NON_SMOOTH_SOLVER = "L-BFGS-B"


@dataclass(frozen=True)
class _PolicyInputs:
    r"""Arrays a parametric policy is built from, aligned on the investment universe.

    Attributes
    ----------
    tilt : ndarray of shape (n_observations, n_assets, n_characteristics)
        Characteristics demeaned over the eligible assets and divided by their
        number, i.e. :math:`\tilde{z}_{t,i} / N_t`. Zero for non-eligible assets.

    benchmark : ndarray of shape (n_observations, n_assets)
        Benchmark weights normalized over the eligible assets. Zero elsewhere.

    holdable : ndarray of shape (n_observations, n_assets)
        Eligibility of each asset at each observation.
    """

    tilt: FloatArray
    benchmark: FloatArray
    holdable: BoolArray


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
    the weights sum to one. With `long_only=True` the weights are truncated at zero
    and renormalized, :math:`w^+_{t,i} = \max(w_{t,i}, 0) / \sum_j \max(w_{t,j}, 0)`,
    as in section 3.2 of [1]_.

    The coefficients :math:`\theta` maximize the average utility of the realized
    portfolio return path

    .. math::

        \max_{\theta} \; \frac{1}{T} \sum_{t} u\left(r^p_{t+1}\right), \qquad
        r^p_{t+1} = \sum_{i} w_{t,i} \, r_{t+1,i}
        - c \sum_i \lvert w_{t,i} - w_{t-1,i} \rvert

    with the constant relative risk aversion (CRRA) utility

    .. math::

        u(r) = \frac{(1 + r)^{1 - \gamma}}{1 - \gamma}

    (and :math:`u(r) = \log(1 + r)` when :math:`\gamma = 1`), where :math:`\gamma` is
    `risk_aversion` and :math:`c` is `transaction_costs`, a proportional cost per unit
    of turnover as in section 3.3 of [1]_. The weights at :math:`t` use only the
    characteristics known at :math:`t` and are evaluated on the returns of
    :math:`t + 1`, so the objective uses no information beyond the decision time.

    Without the long-only constraint and transaction costs the portfolio return is
    linear in :math:`\theta` and the objective is concave. It is then maximized with
    the trust-region method of `scipy.optimize.minimize` using the analytic gradient
    and Hessian, starting from :math:`\theta = 0` (the benchmark), which converges to
    the global optimum. Otherwise the objective is only piecewise smooth (the
    truncation and the turnover introduce kinks) and a quasi-Newton method with
    numerical gradients is used from the same starting point; it finds a local
    optimum.

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

    long_only : bool, default=False
        If True, the policy weights are truncated at zero and renormalized to sum to
        one at every observation, and the coefficients are fitted on the truncated
        policy.

    transaction_costs : float, default=0.0
        Proportional transaction cost :math:`c \ge 0` per unit of one-way turnover,
        in the unit of the returns. The cost of moving from the previous target
        weights to the current ones is deducted from the realized portfolio return in
        the objective. The turnover of the first observation is measured from
        `previous_weights` when provided, and is zero otherwise.

    solver : str, optional
        Method of `scipy.optimize.minimize`. The default (`None`) is `"trust-exact"`
        when the objective is smooth (no long-only constraint and no transaction
        costs), which uses the analytic gradient and Hessian, and `"L-BFGS-B"` with
        numerical gradients otherwise.

    solver_params : dict, optional
        Options passed to `scipy.optimize.minimize` through its `options` argument.

    max_iter : int, default=1000
        Maximum number of solver iterations (the `maxiter` option, unless overridden
        in `solver_params`).

    tol : float, default=1e-10
        Solver tolerance (the `tol` argument of `scipy.optimize.minimize`).

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
        Previous weights of the assets. Used as the starting point of the turnover
        when `transaction_costs` is positive, when `fallback="previous_weights"`, and
        passed to the portfolio on `predict`.

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
        Number of solver iterations.

    solver_result_ : scipy.optimize.OptimizeResult
        Result returned by `scipy.optimize.minimize`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has asset names that are all strings.

    Notes
    -----
    An asset is eligible at :math:`t` when it belongs to the investment universe
    (the columns of `X`), is active in the panel at :math:`t`, has finite
    characteristics at :math:`t` and a positive benchmark weight. Non-eligible assets
    receive a zero weight. Observations with no eligible asset (for example the
    warm-up of rolling characteristics) are excluded from the objective and have NaN
    rows in `weights_history_`. A missing return at :math:`t + 1` of an asset held at
    :math:`t` contributes zero to the realized portfolio return of that observation;
    the weights themselves never depend on the availability of future returns.

    With `long_only=True`, once the coefficients are large enough for the truncation
    to remove the whole benchmark component, the weights become the normalized
    positive part of the tilt and no longer depend on the scale of :math:`\theta`.
    The coefficients of a long-only policy are therefore identified only up to that
    plateau, and policies are better compared through their weights than through
    the size of their coefficients.

    The estimator is split into reusable steps: :meth:`_compute_exposures` builds the
    characteristics, :meth:`_policy_inputs` prepares the tilt, benchmark and
    eligibility arrays, :meth:`_policy_weights` maps coefficients to weights and
    :meth:`_fit_coefficients` runs the utility maximization, so that policies with a
    different coefficient estimation can reuse the infrastructure.

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
    solver_result_: sco.OptimizeResult

    def __init__(
        self,
        characteristics_exposures: list[tuple[str, BaseFactorExposure]],
        risk_aversion: float = 5.0,
        benchmark_mcap_power: float = 1.0,
        long_only: bool = False,
        transaction_costs: float = 0.0,
        solver: str | None = None,
        solver_params: dict | None = None,
        max_iter: int = 1000,
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
        self.long_only = long_only
        self.transaction_costs = transaction_costs
        self.solver = solver
        self.solver_params = solver_params
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
        n_assets = returns.shape[1]

        characteristics = self._validate_characteristics(characteristics, X)
        self._investment_idx_ = self._investment_index(characteristics, X.columns)

        # Characteristics (n_observations, n_coverage_assets, n_characteristics)
        exposures, names = self._compute_exposures(characteristics, method="fit")
        inputs = self._policy_inputs(characteristics, exposures, self._investment_idx_)

        previous_weights = None
        if self.previous_weights is not None:
            previous_weights = input_to_array(
                items=self.previous_weights,
                n_assets=n_assets,
                fill_value=0.0,
                dim=1,
                assets_names=getattr(self, "feature_names_in_", None),
                name="previous_weights",
            )

        # Objective on the realized path: weights at t are evaluated on the returns
        # of t + 1. Observations with no eligible asset are excluded.
        valid_obs = inputs.holdable[:-1].any(axis=1)
        if not valid_obs.any():
            raise ValueError(
                "No observation has an eligible asset with finite characteristics. "
                "Check the warm-up of the characteristics estimators and the "
                "coverage of `X`."
            )
        next_returns = returns[1:]

        theta, result = self._fit_coefficients(
            inputs=_PolicyInputs(
                tilt=inputs.tilt[:-1][valid_obs],
                benchmark=inputs.benchmark[:-1][valid_obs],
                holdable=inputs.holdable[:-1][valid_obs],
            ),
            next_returns=next_returns[valid_obs],
            previous_weights=previous_weights,
        )

        # In-sample weight path and the weights for the next period
        weights_history = self._policy_weights(theta, inputs, self.long_only)
        no_holding = ~inputs.holdable.any(axis=1)
        weights_history[no_holding] = np.nan
        if no_holding[-1]:
            raise ValueError(
                "No asset is eligible at the last observation: the policy weights "
                "cannot be computed."
            )

        self.coef_ = theta
        self.characteristic_names_ = names
        self.n_characteristics_ = len(names)
        self.weights_history_ = weights_history
        self.utility_ = float(-result.fun)
        self.benchmark_utility_ = float(result.benchmark_utility)
        self.n_iter_ = int(getattr(result, "nit", 0))
        self.solver_result_ = result
        self.weights_ = weights_history[-1].copy()
        return self

    def predict_weights(self, characteristics: AssetPanel) -> pd.DataFrame:
        """Apply the fitted coefficients to new characteristics.

        The characteristics are computed with the fitted exposure estimators,
        continuing their state through `partial_fit_transform` when they support
        online updates, so the panel should follow the observations seen in `fit`.
        The fitted policy weights are then applied without refitting.

        Parameters
        ----------
        characteristics : AssetPanel
            Point-in-time panel of asset characteristics covering the investment
            universe seen in `fit`.

        Returns
        -------
        weights : DataFrame of shape (n_observations, n_assets)
            Policy weights at every observation of the panel, aligned with
            `feature_names_in_`. Observations with no eligible asset are NaN.
        """
        skv.check_is_fitted(self, "coef_")
        asset_names = getattr(self, "feature_names_in_", None)
        characteristics = self._validate_characteristics(characteristics, X=None)
        if asset_names is None:
            investment_idx = self._investment_idx_
        else:
            investment_idx = self._investment_index(characteristics, asset_names)
        exposures, _ = self._compute_exposures(characteristics, method="partial_fit")
        inputs = self._policy_inputs(characteristics, exposures, investment_idx)
        weights = self._policy_weights(self.coef_, inputs, self.long_only)
        weights[~inputs.holdable.any(axis=1)] = np.nan
        return pd.DataFrame(
            weights, index=characteristics.observations, columns=asset_names
        )

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
        _validate_non_negative_real(self.transaction_costs, "transaction_costs")
        _validate_non_negative_integer(self.max_iter, "max_iter")
        _validate_positive_real(self.tol, "tol")
        if not isinstance(self.long_only, bool | np.bool_):
            raise ValueError(f"`long_only` must be a boolean, got {self.long_only!r}")
        if self.solver is not None and not isinstance(self.solver, str):
            raise ValueError(f"`solver` must be a string or None, got {self.solver!r}")

    def _validate_characteristics(
        self, characteristics: AssetPanel, X: pd.DataFrame | None
    ) -> AssetPanel:
        """Validate the panel and attach the benchmark weights used by the exposure
        estimators' cross-sectional transformations (same convention as
        `CharacteristicsFactorModel`). Returns a shallow copy.
        """
        need_market_cap = self.benchmark_mcap_power != 0
        characteristics = validate_asset_panel(
            self,
            asset_panel=characteristics,
            required_fields=[_MARKET_CAP] if need_market_cap else None,
            reserved_fields=[_BENCHMARK_WEIGHTS],
            strictly_positive_when_active=[_MARKET_CAP] if need_market_cap else None,
            reset=X is not None,
            copy=True,  # shallow copy: the user's panel is not mutated
        )
        if X is not None:
            if characteristics.n_observations != X.shape[0]:
                raise ValueError(
                    "`X` and `characteristics` must have the same number of "
                    f"observations, got {X.shape[0]} and "
                    f"{characteristics.n_observations}."
                )
            if not np.array_equal(np.asarray(X.index), characteristics.observations):
                raise ValueError("`X.index` must match `characteristics.observations`.")

        benchmark_mask = characteristics.active_mask & characteristics.estimation_mask
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
        return characteristics

    @staticmethod
    def _investment_index(characteristics: AssetPanel, asset_names) -> IntArray:
        """Positions of the investment universe in the coverage universe."""
        asset_idx = {name: i for i, name in enumerate(characteristics.asset_names)}
        try:
            return np.array([asset_idx[x] for x in asset_names], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"Asset {e.args[0]!r} from `X` is missing from `characteristics`."
            ) from e

    def _compute_exposures(
        self, characteristics: AssetPanel, method: str
    ) -> tuple[FloatArray, ObjArray]:
        """Compute and stack the characteristic exposures.

        Parameters
        ----------
        characteristics : AssetPanel
            Validated panel with benchmark weights attached.

        method : str
            `"fit"` clones the estimators and computes the exposures from a clean
            state. `"partial_fit"` continues the fitted estimators' state through
            `partial_fit_transform` when available, and falls back to
            `fit_transform` otherwise.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_coverage_assets, n_characteristics)
            Stacked exposures.

        names : ndarray of shape (n_characteristics,)
            Characteristic names, expanded for multi-factor estimators.
        """
        if method == "fit":
            self.characteristics_exposures_ = [
                (name, sk.clone(estimator))
                for name, estimator in self.characteristics_exposures
            ]
        exposures = []
        names = []
        for name, estimator in self.characteristics_exposures_:
            if method == "partial_fit" and hasattr(estimator, "partial_fit_transform"):
                exposure = estimator.partial_fit_transform(characteristics)
            else:
                exposure = estimator.fit_transform(characteristics)
            exposure = np.asarray(exposure, dtype=float)
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

    @staticmethod
    def _policy_inputs(
        characteristics: AssetPanel, exposures: FloatArray, investment_idx: IntArray
    ) -> _PolicyInputs:
        """Restrict the exposures to the investment universe and build the tilt,
        benchmark and eligibility arrays.

        An asset is eligible at `t` when it is active, has finite characteristics and
        a positive benchmark weight. The characteristics are demeaned over the
        eligible assets and divided by their number, and the benchmark weights are
        normalized over the eligible assets. Nothing depends on future observations.
        """
        z = exposures[:, investment_idx, :]
        cap_weights = characteristics[_BENCHMARK_WEIGHTS][:, investment_idx]
        active_mask = characteristics.active_mask[:, investment_idx]

        holdable = active_mask & np.isfinite(z).all(axis=-1) & (cap_weights > 0)

        counts = holdable.sum(axis=1)[:, np.newaxis, np.newaxis]
        masked = np.where(holdable[:, :, np.newaxis], z, 0.0)
        means = np.divide(
            masked.sum(axis=1, keepdims=True),
            counts,
            out=np.zeros((z.shape[0], 1, z.shape[2])),
            where=counts > 0,
        )
        tilt = np.where(holdable[:, :, np.newaxis], z - means, 0.0)
        tilt = np.divide(tilt, counts, out=np.zeros_like(tilt), where=counts > 0)

        benchmark = np.where(holdable, cap_weights, 0.0)
        benchmark_sum = benchmark.sum(axis=1, keepdims=True)
        benchmark = np.divide(
            benchmark,
            benchmark_sum,
            out=np.zeros_like(benchmark),
            where=benchmark_sum > 0,
        )
        return _PolicyInputs(tilt=tilt, benchmark=benchmark, holdable=holdable)

    @staticmethod
    def _policy_weights(
        theta: FloatArray, inputs: _PolicyInputs, long_only: bool
    ) -> FloatArray:
        """Map coefficients to the policy weights, shape (n_observations, n_assets).

        Non-eligible assets have a zero weight. With `long_only`, the weights are
        truncated at zero and renormalized to sum to one; if no weight is positive
        the benchmark is held.
        """
        weights = inputs.benchmark + inputs.tilt @ np.asarray(theta, dtype=float)
        if long_only:
            weights = np.maximum(weights, 0.0)
            total = weights.sum(axis=1, keepdims=True)
            weights = np.divide(
                weights, total, out=inputs.benchmark.copy(), where=total > 0
            )
        return weights

    def _fit_coefficients(
        self,
        inputs: _PolicyInputs,
        next_returns: FloatArray,
        previous_weights: FloatArray | None,
    ) -> tuple[FloatArray, sco.OptimizeResult]:
        """Maximize the average CRRA utility of the realized policy return path.

        Parameters
        ----------
        inputs : _PolicyInputs
            Policy inputs at the decision times, restricted to observations with at
            least one eligible asset.

        next_returns : ndarray of shape (n_observations, n_assets)
            Returns realized over the period following each decision time. Missing
            returns of held assets contribute zero.

        previous_weights : ndarray of shape (n_assets,) or None
            Weights held before the first observation, for the turnover.

        Returns
        -------
        theta : ndarray of shape (n_characteristics,)
            Optimal coefficients.

        result : scipy.optimize.OptimizeResult
            Solver result with the additional attribute `benchmark_utility`.
        """
        gamma = float(self.risk_aversion)
        cost = float(self.transaction_costs)
        smooth = not self.long_only and cost == 0.0
        n_characteristics = inputs.tilt.shape[-1]
        r = np.where(inputs.holdable & np.isfinite(next_returns), next_returns, 0.0)

        def portfolio_returns(theta: FloatArray) -> FloatArray:
            weights = self._policy_weights(theta, inputs, self.long_only)
            path = np.einsum("ti,ti->t", weights, r)
            if cost > 0.0:
                start = (
                    np.zeros(weights.shape[1])
                    if previous_weights is None
                    else previous_weights
                )
                previous = np.vstack([start[np.newaxis, :], weights[:-1]])
                turnover = np.abs(weights - previous).sum(axis=1)
                if previous_weights is None:
                    turnover[0] = 0.0
                path = path - cost * turnover
            return path

        def objective(theta: FloatArray) -> float:
            path = portfolio_returns(theta)
            if np.any(1.0 + path <= 0.0):
                return np.inf
            return -float(np.mean(_crra(path, gamma)[0]))

        theta0 = np.zeros(n_characteristics)
        benchmark_utility = -objective(theta0)
        if not np.isfinite(benchmark_utility):
            raise ValueError(
                "The benchmark portfolio has a return lower than or equal to -100% at "
                "some observation, which is outside the domain of the CRRA utility."
            )

        options = {"maxiter": int(self.max_iter)}
        if self.solver_params is not None:
            options.update(self.solver_params)
        method = self.solver
        if method is None:
            method = _SMOOTH_SOLVER if smooth else _NON_SMOOTH_SOLVER

        kwargs = {}
        if smooth:
            # The portfolio return is linear in theta: precompute its coefficients
            benchmark_returns = np.einsum("ti,ti->t", inputs.benchmark, r)
            tilt_returns = np.einsum("tik,ti->tk", inputs.tilt, r)
            n = len(benchmark_returns)

            def gradient(theta: FloatArray) -> FloatArray:
                path = benchmark_returns + tilt_returns @ theta
                if np.any(1.0 + path <= 0.0):
                    return np.full(n_characteristics, np.nan)
                return -(tilt_returns.T @ _crra(path, gamma)[1]) / n

            def hessian(theta: FloatArray) -> FloatArray:
                path = benchmark_returns + tilt_returns @ theta
                d2u = _crra(path, gamma)[2]
                return -((tilt_returns * d2u[:, np.newaxis]).T @ tilt_returns) / n

            kwargs["jac"] = gradient
            if method in (
                "trust-exact",
                "trust-ncg",
                "trust-krylov",
                "Newton-CG",
                "dogleg",
            ):
                kwargs["hess"] = hessian

        if self.max_iter == 0:
            # Hold the benchmark without calling the solver
            result = sco.OptimizeResult(
                x=theta0,
                fun=-benchmark_utility,
                nit=0,
                success=True,
                message="max_iter=0",
            )
        else:
            result = sco.minimize(
                objective,
                theta0,
                method=method,
                tol=self.tol,
                options=options,
                **kwargs,
            )
        if not np.isfinite(result.fun):
            raise ValueError(
                f"The solver failed to find a feasible policy: {result.message}"
            )
        result.benchmark_utility = benchmark_utility
        return np.asarray(result.x, dtype=float), result


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
