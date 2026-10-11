"""Supervised linear portfolio policies learned from subperiod allocations."""

# Copyright (c) 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone, is_regressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted, validate_data

from skfolio.containers import AssetPanel
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.parametric._parametric_portfolio_policy import (
    ParametricPortfolioPolicy,
)
from skfolio.optimization.parametric._weight_projection import _WeightProjection
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_non_negative_real,
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
        In-sample allocations under the final coefficients. Unavailable rows are
        NaN. These allocations use coefficients fitted on the whole sample.

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

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        characteristics: AssetPanel | None = None,
    ) -> SupervisedPortfolioPolicy:
        """Fit subperiod targets and the shared linear weight policy.

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
        return self

    def _predict_policy(self, regressor, inputs, projection):
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
        weights[valid] = projection.project(proposed)
        return weights

    def predict_weights(self, characteristics: AssetPanel) -> pd.DataFrame:
        """Predict allocations from a subsequent characteristic panel.

        Parameters
        ----------
        characteristics : AssetPanel
            Subsequent, non-overlapping panel with the fitted coverage asset order.
            Exposure estimators advance their state; the regressor is not refitted.

        Returns
        -------
        weights : DataFrame
            Allocations in fitted investment asset order. Rows with incomplete
            characteristics are NaN.
        """
        check_is_fitted(self, "regressor_")
        panel = self._validate_characteristics(characteristics, X=None)
        investment_idx = self._investment_index(panel, self.feature_names_in_)
        if not panel.active_mask[:, investment_idx].all():
            raise ValueError(
                "The supervised prototype requires a fixed active universe."
            )
        exposures, _ = self._compute_exposures(panel, method="partial_fit")
        inputs = self._policy_inputs(panel, exposures, investment_idx)
        return pd.DataFrame(
            self._predict_policy(self.regressor_, inputs, self.weight_projection_),
            index=panel.observations,
            columns=self.feature_names_in_,
        )
