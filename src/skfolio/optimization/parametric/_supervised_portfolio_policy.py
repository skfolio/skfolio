"""Supervised linear portfolio policies learned from subperiod allocations."""

# Copyright (c) 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import Ridge
from sklearn.utils.validation import validate_data

from skfolio.containers import AssetPanel
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.parametric._parametric_portfolio_policy import (
    ParametricPortfolioPolicy,
)
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_non_negative_real,
)


class SupervisedPortfolioPolicy(ParametricPortfolioPolicy):
    """Learn a linear characteristic policy from optimal subperiod portfolios.

    This experimental supervised alternative fits an independent clone of
    ``optimizer`` on each non-overlapping forward training window. Characteristics
    at the preceding decision date explain those target allocations through ridge
    regression, using the benchmark-relative, demeaned policy of
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
        Non-negative ridge penalty on the shared characteristic coefficients.
    benchmark_mcap_power : float, default=1.0
        Market-cap power used to construct the benchmark; zero gives equal weights.
    long_only : bool, default=False
        Clip negative predicted weights and renormalize after regression.
        Regression itself fits the unprojected benchmark-relative targets.
    portfolio_params : dict, optional
        Parameters forwarded to Portfolio by the inherited predict method.

    Attributes
    ----------
    coef_ : ndarray of shape (n_characteristics,)
        Shared characteristic coefficients.
    regressor_ : Ridge
        Fitted regression without an intercept.
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
    characteristics uses the parent's ``predict_weights`` method, which advances
    the fitted exposure estimators; pass successive, non-overlapping panels with
    the same coverage asset names and order as the training panel.
    This is a linear supervised prototype, not a reproduction of the full
    Chevalier, Coqueret and Raffinot method.
    """

    def __init__(
        self,
        characteristics_exposures: list[tuple[str, BaseFactorExposure]],
        optimizer: BaseOptimization,
        window_size: int = 20,
        alpha: float = 1.0,
        benchmark_mcap_power: float = 1.0,
        long_only: bool = False,
        portfolio_params: dict | None = None,
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
        regressor = Ridge(alpha=self.alpha, fit_intercept=False, solver="svd")
        regressor.fit(design, response)
        weights_history = self._policy_weights(regressor.coef_, inputs, self.long_only)
        weights_history[~inputs.holdable.all(axis=1)] = np.nan

        self.regressor_ = regressor
        self.coef_ = regressor.coef_.copy()
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
