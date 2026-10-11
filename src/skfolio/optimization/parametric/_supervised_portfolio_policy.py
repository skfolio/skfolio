"""Supervised portfolio policies learned from subperiod allocations."""

# Copyright (c) 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import copy, deepcopy
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone, is_regressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted, validate_data

from skfolio.containers import AssetPanel, AssetPanelView
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.moments import LedoitWolf
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.parametric._execution import (
    PortfolioExecution,
    _UnconstrainedProposal,
)
from skfolio.optimization.parametric._parametric_portfolio_policy import (
    ParametricPortfolioPolicy,
    _PolicyInputs,
)
from skfolio.optimization.parametric._prior_metadata import slice_prior_metadata
from skfolio.optimization.parametric._projection_context import projection_context
from skfolio.optimization.parametric._weight_projection import _WeightProjection
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_non_negative_real,
    input_to_array,
)


class _PolicyRegressor(Protocol):
    """Prediction interface shared by regressors and regression pipelines."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one benchmark-relative weight per feature row."""
        ...


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

    allow_universe_changes : bool, default=False
        Allow eligibility changes within the registered asset columns. Excluded
        assets receive zero target weight; constraints are never silently relaxed.
        Newly named assets require explicit admission through expand_universe.

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
    The registered asset columns are fixed per fit. Eligibility may change when
    allow_universe_changes is enabled; label windows still require observed returns
    for each decision-eligible asset. Characteristic warmup windows are skipped. Prediction on new
    characteristics uses repeatable ``predict_weights``. Streaming callers can
    opt into state advancement with ``update_state=True`` and must pass successive,
    non-overlapping panels with the fitted coverage asset names and order.
    This is a supervised prototype, not a reproduction of the full
    Chevalier, Coqueret and Raffinot method.
    """

    def __init__(
        self,
        characteristics_exposures: list[tuple[str, BaseFactorExposure]],
        optimizer: BaseOptimization,
        window_size: int = 20,
        alpha: float = 1.0,
        regressor: BaseEstimator | None = None,
        benchmark_mcap_power: float = 1.0,
        long_only: bool = False,
        portfolio_params: dict | None = None,
        min_weights: float | ArrayLike | dict[str, float] | None = None,
        max_weights: float | ArrayLike | dict[str, float] | None = None,
        groups: ArrayLike | None = None,
        linear_constraints: list[str] | None = None,
        max_variance: float | None = None,
        max_turnover: float | None = None,
        allow_universe_changes: bool = False,
    ) -> None:
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
        self.allow_universe_changes = allow_universe_changes

    def fit(
        self,
        X: pd.DataFrame,
        y: None = None,
        characteristics: AssetPanel | None = None,
        optimizer_fit_params: dict | None = None,
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
        optimizer_fit_params : dict, optional
            Dated metadata passed to each cloned label optimizer, sliced to its
            exact forward training window. Configure nested sklearn metadata
            requests on the supplied optimizer and prior as usual.

        Returns
        -------
        self : SupervisedPortfolioPolicy
            Fitted estimator.
        """
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("_"):
                delattr(self, name)
        self._validate_params()
        if not isinstance(self.allow_universe_changes, bool):
            raise ValueError("allow_universe_changes must be a boolean.")
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
        validate_data(self, X, ensure_all_finite=not self.allow_universe_changes)
        panel = self._validate_characteristics(characteristics, X)
        investment_idx = self._investment_index(panel, X.columns)
        if (
            not self.allow_universe_changes
            and not panel.active_mask[:, investment_idx].all()
        ):
            raise ValueError(
                "The supervised prototype requires a fixed active universe."
            )
        exposures, names = self._compute_exposures(panel, method="fit")
        inputs = self._policy_inputs(panel, exposures, investment_idx)
        if not inputs.holdable[-1].any() or (
            not self.allow_universe_changes and not inputs.holdable[-1].all()
        ):
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
        projection.project(inputs.benchmark[-1:], eligible=inputs.holdable[-1:])

        decisions, targets = [], []
        for t in range(0, len(X) - self.window_size, self.window_size):
            if not inputs.holdable[t].any() or (
                not self.allow_universe_changes and not inputs.holdable[t].all()
            ):
                continue
            estimator = clone(self.optimizer)
            eligible = inputs.holdable[t]
            label_returns = X.iloc[t + 1 : t + 1 + self.window_size, eligible]
            if not np.isfinite(label_returns.to_numpy()).all():
                raise ValueError(
                    "Label windows require observed returns for every decision-eligible asset."
                )
            estimator.fit(
                label_returns,
                **slice_prior_metadata(optimizer_fit_params, label_returns.index),
            )
            weights = np.asarray(estimator.weights_, dtype=float)
            if (
                weights.shape != (int(eligible.sum()),)
                or not np.isfinite(weights).all()
                or not np.isclose(weights.sum(), 1.0)
            ):
                raise ValueError(
                    "Target optimizer must produce finite, fully invested weights."
                )
            decisions.append(t)
            target = np.zeros(X.shape[1])
            target[eligible] = weights
            targets.append(target)
        if not decisions:
            raise ValueError(
                "No complete training windows with available characteristics."
            )

        target_weights = np.asarray(targets)
        sample_mask = inputs.holdable[decisions]
        design = inputs.tilt[decisions][sample_mask]
        response = (target_weights - inputs.benchmark[decisions])[sample_mask]
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

    def expand_universe(
        self,
        X: pd.DataFrame,
        characteristics: AssetPanel,
        *,
        min_weights: float | ArrayLike | dict[str, float] | None = None,
        max_weights: float | ArrayLike | dict[str, float] | None = None,
        groups: ArrayLike | None = None,
    ) -> SupervisedPortfolioPolicy:
        """Return an expanded policy without retraining its regressor.

        Parameters
        ----------
        X : DataFrame
            Expanded return history, starting at the original training start and
            ending at last_prediction_observation_. Include every registered asset
            and at least one new asset. Missing pre-entry returns remain NaN.
        characteristics : AssetPanel
            Expanded characteristic history aligned with X. Exposure estimators
            are rebuilt by replaying this history, so provide enough observations
            to warm up rolling descriptors. Feature definitions must remain stable.
        min_weights : float, dict or array-like, optional
            Replacement lower bounds. Otherwise reuse scalar or named bounds;
            positional bounds require an explicit expanded replacement.
        max_weights : float, dict or array-like, optional
            Replacement upper bounds, with the same rules as min_weights.
        groups : array-like, optional
            Expanded group labels. Required when the original policy has groups.

        Returns
        -------
        policy : SupervisedPortfolioPolicy
            Independent adapted policy. The original model is unchanged. Its
            historical label and weight diagnostics retain their original universe.

        Notes
        -----
        This is an explicit preprocessing reset, not an incremental update to
        arbitrary learned factor models. Replaying a learned transformation can
        change feature meaning even if names match; use fixed feature definitions
        or refit the whole policy in that case. The regression mapping and its
        fitted parameters are preserved. New assets still need finite descriptors
        before admission, and observed past returns or explicit covariance before
        risk-constrained allocation. No historical missing return is filled.
        """
        check_is_fitted(self, "regressor_")
        if not self.allow_universe_changes:
            raise ValueError("Enable allow_universe_changes before expanding assets.")
        old_names = list(self.feature_names_in_)
        if (
            not isinstance(X, pd.DataFrame)
            or not X.columns.is_unique
            or not all(isinstance(name, str) for name in X.columns)
            or not set(old_names) < set(X.columns)
            or X.empty
            or not X.index.is_unique
            or not X.index.is_monotonic_increasing
            or X.index[0] != self.training_returns_.index[0]
            or X.index[-1] != self.last_prediction_observation_
        ):
            raise ValueError(
                "Expanded history must retain all assets, add new names and span "
                "the training start through the current prediction date."
            )
        original = self.training_returns_
        if not original.index.isin(X.index).all() or not np.array_equal(
            X.loc[original.index, old_names].to_numpy(),
            original.loc[:, old_names].to_numpy(),
            equal_nan=True,
        ):
            raise ValueError("Existing return history must be preserved exactly.")
        if np.isinf(X.to_numpy()).any() or (X.to_numpy() < -1).any():
            raise ValueError("Returns cannot be infinite or below -100 percent.")
        worker = deepcopy(self, {id(self.weight_projection_): None})
        for name, replacement in (
            ("min_weights", min_weights),
            ("max_weights", max_weights),
        ):
            bound = getattr(self, name) if replacement is None else replacement
            if (
                replacement is None
                and bound is not None
                and not (np.isscalar(bound) or isinstance(bound, dict))
            ):
                raise ValueError(f"Supply expanded positional {name}.")
            if isinstance(bound, dict) and set(bound) - set(X.columns):
                raise ValueError(f"Unknown asset in {name}.")
            setattr(worker, name, deepcopy(bound))
        if self.groups is not None and groups is None:
            raise ValueError("Supply expanded groups for the new asset order.")
        worker.groups = deepcopy(groups)
        validate_data(worker, X, reset=True, ensure_all_finite=False)
        panel = worker._validate_characteristics(characteristics, X)
        positions = worker._investment_index(panel, X.columns)
        exposures, names = worker._compute_exposures(panel, method="fit")
        if list(names) != list(self.characteristic_names_):
            raise ValueError("Expanded exposures must preserve fitted feature names.")
        inputs = worker._policy_inputs(panel, exposures, positions)
        new_positions = [i for i, name in enumerate(X.columns) if name not in old_names]
        if not inputs.holdable[-1, new_positions].all():
            raise ValueError("New assets need warmed-up finite characteristics.")
        projection = _WeightProjection(
            X.columns,
            worker.long_only,
            worker.min_weights,
            worker.max_weights,
            worker.groups,
            worker.linear_constraints,
        )
        final_inputs = _PolicyInputs(
            tilt=inputs.tilt[-1:],
            benchmark=inputs.benchmark[-1:],
            holdable=inputs.holdable[-1:],
        )
        worker.weights_ = worker._predict_policy(
            worker.regressor_, final_inputs, projection
        )[-1].copy()
        worker.weight_projection_ = projection
        worker._investment_idx_ = positions
        worker.training_returns_ = X.copy()
        return worker

    def _predict_policy(
        self,
        regressor: _PolicyRegressor,
        inputs: _PolicyInputs,
        projection: _WeightProjection | _UnconstrainedProposal,
        covariance: np.ndarray | None = None,
        max_variance: float | None = None,
        holdings: np.ndarray | None = None,
        max_turnover: float | None = None,
    ) -> np.ndarray:
        """Predict asset-level deviations and restore the portfolio budget."""
        valid = (
            inputs.holdable.any(axis=1)
            if self.allow_universe_changes
            else inputs.holdable.all(axis=1)
        )
        weights = np.full(inputs.benchmark.shape, np.nan)
        if not valid.any():
            return weights
        eligible = inputs.holdable[valid]
        predicted = np.asarray(
            regressor.predict(inputs.tilt[valid][eligible]), dtype=float
        )
        if (
            predicted.shape != (int(eligible.sum()),)
            or not np.isfinite(predicted).all()
        ):
            raise ValueError("Regressor must predict one finite value per asset/date.")
        deviations = np.zeros(eligible.shape)
        deviations[eligible] = predicted
        means = deviations.sum(axis=1) / eligible.sum(axis=1)
        deviations = np.where(eligible, deviations - means[:, None], 0)
        proposed = inputs.benchmark[valid] + deviations
        weights[valid] = projection.project(
            proposed,
            covariance=None if covariance is None else covariance[valid],
            max_variance=max_variance,
            holdings=None if holdings is None else holdings[valid],
            max_turnover=max_turnover,
            eligible=eligible,
        )
        return weights

    def predict_weights(
        self,
        characteristics: AssetPanel | AssetPanelView,
        *,
        covariance: ArrayLike | pd.DataFrame | None = None,
        holdings: ArrayLike | pd.DataFrame | None = None,
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
        if (
            not self.allow_universe_changes
            and not panel.active_mask[:, investment_idx].all()
        ):
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

    def predict(self, X: pd.DataFrame) -> Portfolio | Population:
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
        initial_holdings: float | ArrayLike | dict[str, float],
        covariance: ArrayLike | pd.DataFrame | None = None,
        transaction_costs: float | ArrayLike | dict[str, float] | None = None,
        prior_estimator: BasePrior | None = None,
        prior_fit_params: dict | None = None,
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
        prior_fit_params : dict, optional
            Dated prior fit inputs, e.g. {"factors": factor_returns}. Series,
            DataFrames and AssetPanels are sliced to the exact history window.
            Missing dates, duplicate dates and undated arrays raise an error.
            Configure child metadata requests on the supplied prior as usual.
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
        if self.allow_universe_changes:
            raise ValueError("Use evaluate_execution for changing universes.")
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
        if initial_holdings is None:
            raise ValueError("`initial_holdings` must be supplied explicitly.")
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
        if prior_fit_params and (cov is not None or self.max_variance is None):
            raise ValueError("Prior metadata requires automatic risk estimation.")
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
        assert held is not None
        holdings = held[0]
        portfolios = []
        for t in range(len(X)):
            if t % rebalance_every == 0:
                decision_covariance = None if cov is None else cov[t]
                if self.max_variance is not None and cov is None:
                    history = pd.concat([self.training_returns_, X.iloc[:t]]).tail(
                        covariance_window
                    )
                    fitted_prior = clone(risk_prior).fit(
                        history, **slice_prior_metadata(prior_fit_params, history.index)
                    )
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

    def evaluate_execution(
        self,
        X: pd.DataFrame,
        characteristics: AssetPanel,
        *,
        initial_holdings: float | ArrayLike | dict[str, float],
        initial_cash: float,
        cash_weight: float = 0.05,
        cash_returns: float | ArrayLike | pd.Series = 0.0,
        transaction_costs: float | ArrayLike | dict[str, float] | None = None,
        covariance: ArrayLike | pd.DataFrame | None = None,
        prior_estimator: BasePrior | None = None,
        prior_fit_params: dict | None = None,
        covariance_window: int = 60,
        rebalance_every: int = 1,
        eligibility: pd.DataFrame | None = None,
    ) -> PortfolioExecution:
        """Evaluate a strategy with explicit cash and funded transaction fees.

        Parameters
        ----------
        X : DataFrame
            Holding-period returns with the registered asset columns. Decisions
            precede return dates as in evaluate_sequential. A held asset must have
            an observed return; unheld assets may have missing returns.
        characteristics : AssetPanel
            Daily causal characteristics following the fitted exposure state.
        initial_holdings : array-like or dict
            Signed risky-asset fractions of initial wealth. Negative positions
            require long_only=False.
        initial_cash : float
            Initial cash fraction. Together with risky holdings it must sum to one.
        cash_weight : float, default=0.05
            Target cash fraction of post-fee wealth. When no assets are eligible,
            execution holds all cash. Short positions are allowed when
            long_only=False; cash borrowing is not modeled.
        cash_returns : float, array-like or Series, default=0.0
            Cash return per holding period, with no implicit annualization. Series
            dates must match X.
        transaction_costs : float, array-like or dict, optional
            Proportional rates below one, funded through actual trades. Defaults
            to portfolio_params["transaction_costs"], or zero.
        covariance : array-like or DataFrame, optional
            Explicit decision-time covariance, as in predict_weights.
        prior_estimator : BasePrior, optional
            Causal risk prior. Defaults to an empirical LedoitWolf prior.
        prior_fit_params : dict, optional
            Dated metadata sliced to the covariance history window.
        covariance_window : int, default=60
            Maximum history length before dropping incomplete eligible-asset rows.
            At least two complete historical rows are needed for risk estimation.
        rebalance_every : int, default=1
            Number of observations between scheduled trades. Eligibility changes
            force an additional rebalance, without relaxing turnover limits.
        eligibility : DataFrame, optional
            Boolean decision-date/asset schedule, intersected with active_mask.
            It can exclude all assets and move entirely to cash.

        Returns
        -------
        execution : PortfolioExecution
            Targets, actual trades, ending weights and cash/wealth/cost ledger.

        Notes
        -----
        Targets and risk/weight constraints use post-fee wealth. Trades and turnover
        use pre-trade wealth. Funding is solved iteratively and must reconcile;
        failure raises without changing the original model. Exit trades assume
        execution at the known decision valuation. Suspended or untradeable exits
        must not be represented as ordinary eligibility changes. New named assets
        require explicit admission to the registry. Short positions carry no
        borrowing fees; collateral, margin and tradability are outside this
        accounting model. This method does not use Portfolio's legacy
        additive cost convention; fees are actually deducted from available cash.
        """
        from skfolio.optimization.parametric._execution import evaluate_execution

        return evaluate_execution(
            self,
            X,
            characteristics,
            initial_holdings=initial_holdings,
            initial_cash=initial_cash,
            cash_weight=cash_weight,
            cash_returns=cash_returns,
            transaction_costs=transaction_costs,
            covariance=covariance,
            prior_estimator=prior_estimator,
            prior_fit_params=prior_fit_params,
            covariance_window=covariance_window,
            rebalance_every=rebalance_every,
            eligibility=eligibility,
        )
