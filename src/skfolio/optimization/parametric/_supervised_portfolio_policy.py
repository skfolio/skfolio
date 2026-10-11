"""Supervised portfolio policies learned from subperiod allocations."""

# Copyright (c) 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import copy, deepcopy

import numpy as np
import pandas as pd
from sklearn.base import clone, is_regressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted, validate_data

from skfolio.containers import AssetPanel, AssetPanelView
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.moments import LedoitWolf
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.parametric._parametric_portfolio_policy import (
    ParametricPortfolioPolicy,
)
from skfolio.optimization.parametric._projection_context import projection_context
from skfolio.optimization.parametric._weight_projection import _WeightProjection
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_non_negative_real,
    input_to_array,
)


class SupervisedPortfolioPolicy(ParametricPortfolioPolicy):
    """Learn a characteristic policy from optimal subperiod portfolios.

    This experimental supervised alternative fits an independent clone of
    ``optimizer`` on each non-overlapping forward training window. Characteristics
    at the preceding decision date explain those target allocations through a shared
    regressor, using the benchmark-relative, demeaned policy of
    :class:`ParametricPortfolioPolicy`.

    Only complete windows inside the training sample produce labels. Labels use
    future returns *within that sample*, so ``weights_history_`` is an in-sample
    diagnostic, not a backtest. Evaluate on a later, untouched sample.
    Predictions are projected onto a static feasible portfolio set using the
    existing CLARABEL solver dependency.

    Parameters
    ----------
    characteristics_exposures : list of (str, BaseFactorExposure)
        Point-in-time characteristic estimators, as in ParametricPortfolioPolicy.
        They must compute historical exposures causally.
    optimizer : BaseOptimization
        Portfolio estimator cloned separately for each label window. Its output
        must contain finite, fully invested weights. Constraints on these labels
        are not automatically constraints on the learned policy.
    window_size : int, default=20
        Number of subsequent return observations used for each target portfolio.
        Must be at least two. Incomplete trailing windows are discarded.
    alpha : float, default=1.0
        Non-negative penalty for the default Ridge regressor. Ignored when
        ``regressor`` is supplied.
    regressor : regressor, optional
        Cloned scikit-learn compatible regressor or pipeline accepting a 2D
        feature matrix and predicting one value per asset/date. Defaults to Ridge
        without an intercept. Model-specific constraints belong in this estimator.
    benchmark_mcap_power : float, default=1.0
        Market-cap power used to construct the benchmark; zero gives equal weights.
    long_only : bool, default=False
        Enforce nonnegative weights through Euclidean projection after regression.
        Regression itself fits the unprojected benchmark-relative targets.
    portfolio_params : dict, optional
        Parameters forwarded to Portfolio by the inherited predict method.

    min_weights, max_weights : float, array-like or dict, optional
        Static per-asset limits enforced after prediction. Dictionary keys are
        fitted asset names; omitted entries default to zero and one respectively.
    groups : array-like of shape (n_groups, n_assets), optional
        Group labels in the fitted investment asset order.
    linear_constraints : list of str, optional
        Asset/group equations such as ``["Equity <= 0.6", "a >= 0.1"]``.
        Unknown group names and infeasible constraints raise an error.

    max_variance : float, optional
        Variance limit applied at prediction with explicitly supplied covariance.
        Units must match the covariance horizon. Training weights/history enforce
        static constraints only; standard predict is disabled with dynamic limits.

    max_turnover : float, optional
        Limit on sum(abs(target - pre-trade holdings)), without a one-half factor.
        Requires actual fully invested holdings for each prediction date.

    Attributes
    ----------
    coef_ : ndarray of shape (n_characteristics,)
        Characteristic coefficients, only when exposed by the fitted regressor.
    regressor_ : regressor
        Fitted clone predicting benchmark-relative deviations.
    target_weights_ : DataFrame
        Training target allocations indexed by their decision dates.
    training_windows_ : DataFrame
        Decision dates, first and last label return dates, for auditing timing.
    weights_ : ndarray of shape (n_assets,)
        Allocation obtained from the final training characteristics.
    weights_history_ : ndarray of shape (n_observations, n_assets)
        In-sample allocations under the final regressor. Unavailable rows are
        NaN. The regressor is fitted on the whole training sample.

    Notes
    -----
    This initial implementation requires a fixed, active investment universe and
    finite returns. Characteristic warmup windows are skipped. Prediction on new
    characteristics uses ``predict_weights``, which advances
    the fitted exposure estimators; pass successive, non-overlapping panels with
    the same coverage asset names and order as the training panel.
    This is a supervised prototype, not a reproduction of the full
    Chevalier, Coqueret and Raffinot method.
    """

    def __init__(
        self,
        characteristics_exposures: list[tuple[str, BaseFactorExposure]],
        optimizer: BaseOptimization,
        window_size: int = 20,
        alpha: float = 1.0,
        regressor=None,
        benchmark_mcap_power: float = 1.0,
        long_only: bool = False,
        portfolio_params: dict | None = None,
        min_weights=None,
        max_weights=None,
        groups=None,
        linear_constraints=None,
        max_variance: float | None = None,
        max_turnover: float | None = None,
    ):
        super().__init__(
            characteristics_exposures=characteristics_exposures,
            benchmark_mcap_power=benchmark_mcap_power,
            long_only=long_only,
            portfolio_params=portfolio_params,
        )
        self.optimizer = optimizer
        self.window_size = window_size
        self.alpha = alpha
        self.regressor = regressor
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.max_variance = max_variance
        self.max_turnover = max_turnover

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        characteristics: AssetPanel | None = None,
    ) -> SupervisedPortfolioPolicy:
        """Fit subperiod targets and the shared regression weight policy.

        Parameters
        ----------
        X : DataFrame of shape (n_observations, n_assets)
            Finite asset returns with a unique, increasing observation index.
        y : Ignored
            Not used, present for API consistency.
        characteristics : AssetPanel
            Point-in-time characteristics aligned with X.

        Returns
        -------
        self : SupervisedPortfolioPolicy
            Fitted estimator.
        """
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("_"):
                delattr(self, name)
        self._validate_params()
        if self.max_variance is not None:
            _validate_non_negative_real(self.max_variance, "max_variance")
        if self.max_turnover is not None:
            _validate_non_negative_real(self.max_turnover, "max_turnover")
        _validate_non_negative_integer(self.window_size, "window_size")
        if self.window_size < 2:
            raise ValueError("`window_size` must be at least 2.")
        _validate_non_negative_real(self.alpha, "alpha")
        if not isinstance(self.optimizer, BaseOptimization):
            raise ValueError("`optimizer` must be a BaseOptimization estimator.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` must be a pd.DataFrame.")
        if not X.index.is_unique or not X.index.is_monotonic_increasing:
            raise ValueError("`X.index` must be unique and increasing.")
        if characteristics is None:
            raise ValueError("`characteristics` must be provided.")
        validate_data(self, X)
        panel = self._validate_characteristics(characteristics, X)
        investment_idx = self._investment_index(panel, X.columns)
        if not panel.active_mask[:, investment_idx].all():
            raise ValueError(
                "The supervised prototype requires a fixed active universe."
            )
        exposures, names = self._compute_exposures(panel, method="fit")
        inputs = self._policy_inputs(panel, exposures, investment_idx)
        if not inputs.holdable[-1].all():
            raise ValueError("Final characteristics must be available for every asset.")

        projection = _WeightProjection(
            X.columns,
            self.long_only,
            self.min_weights,
            self.max_weights,
            self.groups,
            self.linear_constraints,
        )
        # Check feasibility before fitting label optimizers.
        projection.project(inputs.benchmark[-1:])

        decisions, targets = [], []
        for t in range(0, len(X) - self.window_size, self.window_size):
            if not inputs.holdable[t].all():
                continue
            estimator = clone(self.optimizer)
            estimator.fit(X.iloc[t + 1 : t + 1 + self.window_size])
            weights = np.asarray(estimator.weights_, dtype=float)
            if (
                weights.shape != (X.shape[1],)
                or not np.isfinite(weights).all()
                or not np.isclose(weights.sum(), 1.0)
            ):
                raise ValueError(
                    "Target optimizer must produce finite, fully invested weights."
                )
            decisions.append(t)
            targets.append(weights.copy())
        if not decisions:
            raise ValueError(
                "No complete training windows with available characteristics."
            )

        target_weights = np.asarray(targets)
        design = inputs.tilt[decisions].reshape(-1, len(names))
        response = (target_weights - inputs.benchmark[decisions]).ravel()
        regressor = (
            Ridge(alpha=self.alpha, fit_intercept=False, solver="svd")
            if self.regressor is None
            else clone(self.regressor)
        )
        if not is_regressor(regressor):
            raise ValueError("`regressor` must be a scikit-learn compatible regressor.")
        regressor.fit(design, response)
        weights_history = self._predict_policy(regressor, inputs, projection)
        self.weight_projection_ = projection

        self.regressor_ = regressor
        if hasattr(regressor, "coef_"):
            self.coef_ = np.asarray(regressor.coef_).copy()
        self.characteristic_names_ = names
        self.n_characteristics_ = len(names)
        self._investment_idx_ = investment_idx
        self.target_weights_ = pd.DataFrame(
            target_weights, index=X.index[decisions], columns=X.columns
        )
        self.training_windows_ = pd.DataFrame(
            {
                "start": X.index[np.asarray(decisions) + 1],
                "end": X.index[np.asarray(decisions) + self.window_size],
            },
            index=X.index[decisions],
        )
        self.weights_history_ = weights_history
        self.weights_ = weights_history[-1].copy()
        self.training_returns_ = X.copy()
        self.last_observation_ = X.index[-1]
        self.last_prediction_observation_ = X.index[-1]
        return self

    def _predict_policy(
        self,
        regressor,
        inputs,
        projection,
        covariance=None,
        max_variance=None,
        holdings=None,
        max_turnover=None,
    ):
        """Predict asset-level deviations and restore the portfolio budget."""
        valid = inputs.holdable.all(axis=1)
        weights = np.full(inputs.benchmark.shape, np.nan)
        if not valid.any():
            return weights
        n_rows, n_assets, n_features = inputs.tilt[valid].shape
        deviations = np.asarray(
            regressor.predict(inputs.tilt[valid].reshape(-1, n_features)), dtype=float
        )
        if (
            deviations.shape != (n_rows * n_assets,)
            or not np.isfinite(deviations).all()
        ):
            raise ValueError("Regressor must predict one finite value per asset/date.")
        deviations = deviations.reshape(n_rows, n_assets)
        deviations -= deviations.mean(axis=1, keepdims=True)
        proposed = inputs.benchmark[valid] + deviations
        weights[valid] = projection.project(
            proposed,
            covariance=None if covariance is None else covariance[valid],
            max_variance=max_variance,
            holdings=None if holdings is None else holdings[valid],
            max_turnover=max_turnover,
        )
        return weights

    def predict_weights(
        self,
        characteristics: AssetPanel | AssetPanelView,
        *,
        covariance=None,
        holdings=None,
        update_state: bool = False,
    ) -> pd.DataFrame:
        """Predict allocations from a subsequent characteristic panel.

        Parameters
        ----------
        characteristics : AssetPanel
            Subsequent, non-overlapping panel with the fitted coverage asset order.
            The regressor is not refitted.
        covariance : array-like or DataFrame, optional
            Covariance available at each decision. Shape (t,n,n), or (n,n) to
            explicitly share one estimate. Named matrices are aligned to assets.
            The caller is responsible for using only then-available information.
        holdings : array-like or DataFrame, optional
            Actual pre-trade holdings of shape (t,n), after weight drift. A vector
            is accepted for a single date only; holdings are never broadcast.
            Named columns are aligned; DataFrame dates must match the panel.

        update_state : bool, default=False
            Commit exposure state only after a successful prediction. The default
            leaves fitted state unchanged and permits repeatable predictions.
            Streaming callers must explicitly opt in and pass increasing dates.

        Returns
        -------
        weights : DataFrame
            Allocations in fitted investment asset order. Rows with incomplete
            characteristics are NaN.
        """
        check_is_fitted(self, "regressor_")
        dates = pd.Index(characteristics.observations)
        if (
            len(dates) == 0
            or not dates.is_unique
            or not dates.is_monotonic_increasing
            or dates[0] <= self.last_prediction_observation_
        ):
            raise ValueError("Prediction dates must follow the fitted exposure state.")
        worker = copy(self)
        worker.characteristics_exposures_ = deepcopy(self.characteristics_exposures_)
        covariance, holdings = projection_context(
            self.feature_names_in_,
            characteristics.observations,
            covariance=covariance,
            holdings=holdings,
        )
        if self.max_variance is not None and covariance is None:
            raise ValueError("`covariance` is required for a risk limit.")
        if self.max_turnover is not None and holdings is None:
            raise ValueError("`holdings` is required for a turnover limit.")
        if isinstance(characteristics, AssetPanelView):
            characteristics = characteristics.to_panel(deep=False)
        panel = worker._validate_characteristics(characteristics, X=None)
        investment_idx = self._investment_index(panel, self.feature_names_in_)
        if not panel.active_mask[:, investment_idx].all():
            raise ValueError(
                "The supervised prototype requires a fixed active universe."
            )
        exposures, _ = worker._compute_exposures(panel, method="partial_fit")
        inputs = self._policy_inputs(panel, exposures, investment_idx)
        result = pd.DataFrame(
            self._predict_policy(
                self.regressor_,
                inputs,
                self.weight_projection_,
                covariance=covariance,
                max_variance=self.max_variance,
                holdings=holdings,
                max_turnover=self.max_turnover,
            ),
            index=panel.observations,
            columns=self.feature_names_in_,
        )

        if update_state:
            self.characteristics_exposures_ = worker.characteristics_exposures_
            self.last_prediction_observation_ = dates[-1]
        return result

    def predict(self, X):
        """Evaluate the final static allocation when no dynamic limits are enabled.

        Parameters
        ----------
        X : array-like
            Evaluation returns.

        Returns
        -------
        portfolio : Portfolio
            Portfolio using the final static training allocation.
        """
        if self.max_variance is not None or self.max_turnover is not None:
            raise ValueError(
                "Use predict_weights with explicit risk and holdings inputs."
            )
        return super().predict(X)

    def evaluate_sequential(
        self,
        X: pd.DataFrame,
        characteristics: AssetPanel,
        *,
        initial_holdings,
        covariance=None,
        transaction_costs=None,
        prior_estimator: BasePrior | None = None,
        covariance_window: int = 60,
        rebalance_every: int = 1,
    ) -> MultiPeriodPortfolio:
        """Evaluate scheduled rebalances over successive return observations.

        Parameters
        ----------
        X : DataFrame
            Realized returns for successive holding periods. Each row ends AFTER
            its corresponding characteristic decision date. From row two onward,
            the decision date must equal the preceding return's end date.
        characteristics : AssetPanel
            One causal characteristic row per decision, strictly after training.
        initial_holdings : array-like
            Fully invested pre-trade weights at the first decision in fitted asset
            order. No initial allocation or initial trade is silently assumed.
        covariance : array-like or DataFrame, optional
            Point-in-time covariance input, as in predict_weights.
        transaction_costs : float, array-like or dict, optional
            Nonnegative proportional costs per asset, charged on absolute traded
            weights. Named rates align to fitted assets; omitted names cost zero.
            Overrides portfolio_params["transaction_costs"] when supplied.

        prior_estimator : BasePrior, optional
            Covariance prior cloned and refitted on trailing observed returns at
            each rebalance when a risk limit is enabled without explicit covariance.
            Defaults to EmpiricalPrior with LedoitWolf covariance shrinkage.
            Priors requiring additional metadata are not supported here.
        covariance_window : int, default=60
            Maximum number of past return rows supplied to each prior fit; at
            least two. No return from the following holding period is included.

        rebalance_every : int, default=1
            Number of return observations between rebalances (for example, 5 or
            21 for trading-week/month approximations on daily data). The first
            decision always rebalances. Holdings drift between decisions; limits
            apply at rebalance time, not continuously during the holding period.

        Returns
        -------
        portfolio : MultiPeriodPortfolio
            One Portfolio per holding period. Its previous_weights, weights,
            ending_weights and turnover expose the execution sequence.

        Notes
        -----
        Runs on a deep copy of fitted state, so evaluation is repeatable and a
        partial failure cannot advance the original estimator. Costs follow
        Portfolio accounting: deducted from period returns, while ending weights
        follow asset-return drift. No separate cash account is modeled. Management
        fees are not supported. Costs affect evaluation, not regressor fitting.
        """
        check_is_fitted(self, "regressor_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(
            self.feature_names_in_
        ):
            raise ValueError(
                "Evaluation returns must have fitted asset columns in order."
            )
        dates = pd.Index(characteristics.observations)
        if (
            len(X) == 0
            or len(X) != len(dates)
            or not X.index.is_unique
            or not X.index.is_monotonic_increasing
            or not dates.is_unique
            or not dates.is_monotonic_increasing
            or dates[0] <= self.last_observation_
            or not (dates < X.index).all()
            or not dates[1:].equals(X.index[:-1])
        ):
            raise ValueError(
                "Decision dates must precede returns and form consecutive out-of-sample periods."
            )
        if not np.isfinite(X.to_numpy()).all() or (X.to_numpy() <= -1).any():
            raise ValueError(
                "Evaluation requires finite returns strictly greater than -1."
            )
        cov, held = projection_context(
            self.feature_names_in_, dates[:1], holdings=initial_holdings
        )
        cov, _ = projection_context(
            self.feature_names_in_, dates, covariance=covariance
        )
        _validate_non_negative_integer(rebalance_every, "rebalance_every")
        if rebalance_every < 1:
            raise ValueError("`rebalance_every` must be at least one.")
        _validate_non_negative_integer(covariance_window, "covariance_window")
        if covariance_window < 2:
            raise ValueError("`covariance_window` must be at least two.")
        if prior_estimator is not None and not isinstance(prior_estimator, BasePrior):
            raise ValueError("`prior_estimator` must be a BasePrior estimator.")
        if prior_estimator is not None and cov is not None:
            raise ValueError("Supply either covariance or prior_estimator, not both.")
        risk_prior = (
            EmpiricalPrior(covariance_estimator=LedoitWolf())
            if prior_estimator is None
            else prior_estimator
        )
        params = dict(self.portfolio_params or {})
        costs = params.pop("transaction_costs", None)
        if transaction_costs is not None:
            costs = transaction_costs
        if costs is None:
            costs = 0.0
        if isinstance(costs, dict) and set(costs) - set(self.feature_names_in_):
            raise ValueError("Transaction cost names must belong to fitted assets.")
        cost_rates = (
            np.full(len(self.feature_names_in_), costs, dtype=float)
            if np.isscalar(costs)
            else input_to_array(
                costs,
                len(self.feature_names_in_),
                0,
                1,
                self.feature_names_in_,
                "transaction_costs",
            )
        )
        if not np.isfinite(cost_rates).all() or (cost_rates < 0).any():
            raise ValueError("Transaction costs must be finite and nonnegative.")
        forbidden = {
            "weights",
            "previous_weights",
            "weight_drift",
            "X",
            "management_fees",
            "sample_weight",
        }
        if forbidden.intersection(params):
            raise ValueError(
                "Sequential evaluation controls holdings/drift; management fees and sample weights are unsupported."
            )
        # Rebuild CVXPY objects: copied native solver/canonicalization caches are
        # not safe to reuse. Only fitted statistical/model state is deep-copied.
        runner = deepcopy(self, {id(self.weight_projection_): None})
        runner.weight_projection_ = _WeightProjection(
            self.feature_names_in_,
            self.long_only,
            self.min_weights,
            self.max_weights,
            self.groups,
            self.linear_constraints,
        )
        holdings = held[0]
        portfolios = []
        for t in range(len(X)):
            if t % rebalance_every == 0:
                decision_covariance = None if cov is None else cov[t]
                if self.max_variance is not None and cov is None:
                    history = pd.concat([self.training_returns_, X.iloc[:t]]).tail(
                        covariance_window
                    )
                    fitted_prior = clone(risk_prior).fit(history)
                    decision_covariance = fitted_prior.return_distribution_.covariance
                targets = (
                    runner.predict_weights(
                        characteristics.isel(observations=slice(t, t + 1)),
                        covariance=decision_covariance,
                        holdings=holdings,
                        update_state=True,
                    )
                    .iloc[0]
                    .to_numpy()
                )
            else:
                # Advance descriptors even when no trade is scheduled, so rolling
                # characteristics keep the full observation stream.
                observation = characteristics.isel(observations=slice(t, t + 1))
                if isinstance(observation, AssetPanelView):
                    observation = observation.to_panel(deep=False)
                observation = runner._validate_characteristics(observation, X=None)
                if not observation.active_mask[:, runner._investment_idx_].all():
                    raise ValueError(
                        "Sequential evaluation requires a fixed active universe."
                    )
                runner._compute_exposures(observation, method="partial_fit")
                runner.last_prediction_observation_ = dates[t]
                targets = holdings.copy()
            if not np.isfinite(targets).all():
                raise ValueError(
                    "All characteristics must be available at each rebalance."
                )
            period = Portfolio(
                X.iloc[t : t + 1],
                weights=targets,
                previous_weights=holdings,
                weight_drift=True,
                transaction_costs=cost_rates,
                **params,
            )
            holdings = period.ending_weights.copy()
            if not np.isfinite(holdings).all() or 1 + float(period.returns[0]) <= 0:
                raise ValueError("Nonpositive wealth prevents continued evaluation.")
            portfolios.append(period)
        return MultiPeriodPortfolio(
            portfolios=portfolios, check_observations_order=True, compounded=True
        )
