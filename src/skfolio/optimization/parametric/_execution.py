"""Cash-funded portfolio accounting with an auditable wealth ledger."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skfolio.containers import AssetPanel, AssetPanelView
from skfolio.moments import LedoitWolf
from skfolio.optimization.parametric._prior_metadata import slice_prior_metadata
from skfolio.optimization.parametric._projection_context import projection_context
from skfolio.optimization.parametric._weight_projection import _WeightProjection
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.tools import _validate_non_negative_integer, input_to_array

if TYPE_CHECKING:
    from skfolio.optimization.parametric._supervised_portfolio_policy import (
        SupervisedPortfolioPolicy,
    )


@dataclass(frozen=True)
class PortfolioExecution:
    """Execution ledger, with all monetary fractions relative to pre-trade wealth.

    Attributes
    ----------
    ledger : DataFrame
        Period returns, compounded wealth, cash weights, costs and turnover.
        Target cash is relative to post-fee wealth; ending cash is after returns.
    targets : DataFrame
        Risky target weights relative to post-fee wealth.
    trades : DataFrame
        Signed risky-asset trades as fractions of pre-trade wealth.
    ending_weights : DataFrame
        Risky asset weights after the holding-period returns.
    """

    ledger: pd.DataFrame
    targets: pd.DataFrame
    trades: pd.DataFrame
    ending_weights: pd.DataFrame


class _UnconstrainedProposal:
    """Extract model predictions before the execution constraint stage."""

    def project(self, proposed: np.ndarray, **kwargs: object) -> np.ndarray:
        """Return raw policy weights before the accounting projection."""
        return proposed


def _funded_projection(
    projection: _WeightProjection,
    proposed: np.ndarray,
    holdings: np.ndarray,
    rates: np.ndarray,
    eligible: np.ndarray,
    cash_weight: float,
    covariance: np.ndarray | None,
    max_variance: float | None,
    max_turnover: float | None,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Solve fee funding to a fixed point and verify actual traded amounts."""
    capital = 1.0
    budget = 1 - cash_weight if eligible.any() else 0.0
    for _ in range(200):
        weights = projection.project(
            (proposed * budget)[None, :],
            eligible=eligible[None, :],
            budget=budget,
            covariance=None if covariance is None else covariance[None, :, :],
            max_variance=max_variance,
            holdings=(holdings / capital)[None, :],
            max_turnover=None if max_turnover is None else max_turnover / capital,
        )[0]
        weights[~eligible] = 0
        trades = capital * weights - holdings
        cost = float(rates @ np.abs(trades))
        if 1 - cost <= 0:
            raise ValueError("Transaction costs exhaust available wealth.")
        if abs(capital - (1 - cost)) <= 1e-10:
            dollars = capital * weights
            cash = float(1 - dollars.sum() - cost)
            if cash < -1e-8:
                raise ValueError("Insufficient cash to fund transaction costs.")
            if max_turnover is not None and np.abs(trades).sum() > max_turnover + 1e-6:
                raise ValueError("Funded trades exceed the turnover limit.")
            if abs(cash - (1 - cost) * (1 - budget)) > 1e-7:
                raise ValueError("Funded cash allocation failed reconciliation.")
            return dollars, max(cash, 0.0), trades, cost
        capital = 1 - cost
    raise ValueError(
        "Cash funding did not converge; no execution result was committed."
    )


def evaluate_execution(
    model: SupervisedPortfolioPolicy,
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
    """Run explicit cash accounting without mutating the fitted estimator."""
    check_is_fitted(model, "regressor_")
    names = [str(name) for name in model.feature_names_in_]
    if (
        not isinstance(X, pd.DataFrame)
        or not X.columns.is_unique
        or set(X.columns) != set(names)
    ):
        raise ValueError("Returns must contain exactly the registered asset names.")
    X = X.loc[:, names]
    dates = pd.Index(characteristics.observations)
    if (
        len(X) == 0
        or len(X) != len(dates)
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not X.index.is_unique
        or not X.index.is_monotonic_increasing
        or dates[0] <= model.last_prediction_observation_
        or not (dates < X.index).all()
        or not dates[1:].equals(X.index[:-1])
    ):
        raise ValueError(
            "Execution needs consecutive, increasing out-of-sample decision dates."
        )
    if eligibility is not None:
        if (
            not isinstance(eligibility, pd.DataFrame)
            or not eligibility.index.equals(dates)
            or not eligibility.columns.is_unique
            or set(eligibility.columns) != set(names)
            or not all(
                pd.api.types.is_bool_dtype(dtype) for dtype in eligibility.dtypes
            )
            or eligibility.isna().any().any()
        ):
            raise ValueError(
                "Eligibility must be a boolean DataFrame aligned with decision dates and assets."
            )
        eligibility = eligibility.loc[:, names].to_numpy(dtype=bool)
    if np.isinf(X.to_numpy()).any() or (X.to_numpy() < -1).any():
        raise ValueError("Asset returns cannot be infinite or below -100 percent.")
    for name, value in (("initial_cash", initial_cash), ("cash_weight", cash_weight)):
        if not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite fraction between zero and one.")
    for name, value in (
        ("rebalance_every", rebalance_every),
        ("covariance_window", covariance_window),
    ):
        _validate_non_negative_integer(value, name)
        if value < (2 if name == "covariance_window" else 1):
            raise ValueError(f"Invalid {name}.")

    def asset_vector(
        value: float | ArrayLike | dict[str, float], name: str, nonnegative: bool = True
    ) -> np.ndarray:
        """Align named or positional holdings and costs to registered assets."""
        if isinstance(value, dict) and set(value) - set(names):
            raise ValueError(f"Unknown asset in {name}.")
        array = (
            np.full(len(names), value, dtype=float)
            if np.isscalar(value)
            else input_to_array(value, len(names), 0, 1, np.asarray(names), name)
        )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite.")
        if nonnegative and (array < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative.")
        return array

    holdings = asset_vector(
        initial_holdings, "initial_holdings", nonnegative=model.long_only
    )
    if not np.isclose(holdings.sum() + initial_cash, 1, atol=1e-9, rtol=0):
        raise ValueError("Initial risky holdings plus cash must sum to one.")
    params = dict(model.portfolio_params or {})
    if set(params) - {"transaction_costs"}:
        raise ValueError(
            "Cash execution only reads transaction_costs from portfolio_params."
        )
    costs = (
        transaction_costs
        if transaction_costs is not None
        else params.get("transaction_costs", 0.0)
    )
    rates = asset_vector(costs, "transaction_costs")
    if (rates >= 1).any():
        raise ValueError("Transaction cost rates must be below one.")
    if isinstance(cash_returns, pd.Series):
        if not cash_returns.index.equals(X.index):
            raise ValueError("Cash return dates must match holding-period returns.")
        cash_returns = cash_returns.to_numpy()
    cash_rates = np.broadcast_to(np.asarray(cash_returns, dtype=float), (len(X),))
    if not np.isfinite(cash_rates).all() or (cash_rates <= -1).any():
        raise ValueError("Cash returns must be finite and greater than -100 percent.")
    cov, _ = projection_context(names, dates, covariance=covariance)
    if prior_estimator is not None and not isinstance(prior_estimator, BasePrior):
        raise ValueError("prior_estimator must be a BasePrior.")
    if cov is not None and (prior_estimator is not None or prior_fit_params):
        raise ValueError("Supply explicit covariance or a prior, not both.")
    if prior_fit_params and model.max_variance is None:
        raise ValueError("Prior metadata requires a risk constraint.")
    prior = (
        prior_estimator
        if prior_estimator is not None
        else EmpiricalPrior(covariance_estimator=LedoitWolf())
    )
    runner = deepcopy(model, {id(model.weight_projection_): None})
    projection = _WeightProjection(
        names,
        model.long_only,
        model.min_weights,
        model.max_weights,
        model.groups,
        model.linear_constraints,
        solver=model.projection_solver,
        solver_params=model.projection_solver_params,
    )
    cash, wealth, previous_eligible = float(initial_cash), 1.0, None
    records, target_rows, trade_rows, ending_rows = [], [], [], []
    for t in range(len(X)):
        panel = characteristics.isel(observations=slice(t, t + 1))
        if isinstance(panel, AssetPanelView):
            panel = panel.to_panel(deep=False)
        panel = runner._validate_characteristics(panel, X=None)
        positions = runner._investment_index(panel, names)
        active = panel.active_mask[0, positions].copy()
        if eligibility is not None:
            active &= eligibility[t]
        if not model.allow_universe_changes and not active.all():
            raise ValueError("Enable allow_universe_changes for eligibility changes.")
        exposures, _ = runner._compute_exposures(panel, method="partial_fit")
        if eligibility is not None:
            exposures = exposures.copy()
            exposures[0, positions[~active], :] = np.nan
        inputs = runner._policy_inputs(panel, exposures, positions)
        eligible = inputs.holdable[0]
        if (active & ~eligible).any():
            raise ValueError(
                "Active assets need finite characteristics and benchmark eligibility."
            )
        rebalance = (
            t % rebalance_every == 0
            or previous_eligible is None
            or not np.array_equal(eligible, previous_eligible)
        )
        if rebalance:
            proposal = (
                runner._predict_policy(
                    runner.regressor_, inputs, _UnconstrainedProposal()
                )[0]
                if eligible.any()
                else np.zeros(len(names))
            )
            decision_cov = None if cov is None else cov[t]
            if model.max_variance is not None and cov is None:
                decision_cov = np.zeros((len(names), len(names)))
                if eligible.any() and cash_weight < 1:
                    history = pd.concat([model.training_returns_, X.iloc[:t]]).tail(
                        covariance_window
                    )
                    history = history.loc[:, np.asarray(names)[eligible]].dropna()
                    if len(history) < 2:
                        raise ValueError(
                            "Eligible assets need at least two complete past return rows for risk estimation."
                        )
                    fitted = clone(prior).fit(
                        history, **slice_prior_metadata(prior_fit_params, history.index)
                    )
                    decision_cov[np.ix_(eligible, eligible)] = (
                        fitted.return_distribution_.covariance
                    )
                    validated_cov, _ = projection_context(
                        names, dates[t : t + 1], covariance=decision_cov
                    )
                    assert validated_cov is not None
                    decision_cov = validated_cov[0]
            dollars, cash_dollars, trades, cost = _funded_projection(
                projection,
                proposal,
                holdings,
                rates,
                eligible,
                cash_weight,
                decision_cov,
                model.max_variance,
                model.max_turnover,
            )
        else:
            dollars, cash_dollars = holdings.copy(), cash
            trades, cost = np.zeros(len(names)), 0.0
        observed = X.iloc[t].to_numpy()
        if np.isnan(observed[np.abs(dollars) > 1e-10]).any():
            raise ValueError(
                "A held asset has an unobserved return; supply an explicit valuation or exit."
            )
        end_values = dollars * (1 + np.nan_to_num(observed, nan=0.0))
        end_cash = cash_dollars * (1 + cash_rates[t])
        growth = float(end_values.sum() + end_cash)
        if growth <= 0 or not np.isfinite(growth):
            raise ValueError("Execution exhausted portfolio wealth.")
        holdings, cash = end_values / growth, float(end_cash / growth)
        wealth *= growth
        records.append(
            {
                "return": growth - 1,
                "wealth": wealth,
                "cost": cost,
                "turnover": float(np.abs(trades).sum()),
                "rebalanced": rebalance,
                "target_cash": cash_dollars / (1 - cost),
                "ending_cash": cash,
            }
        )
        target_rows.append(dollars / (1 - cost))
        trade_rows.append(trades)
        ending_rows.append(holdings.copy())
        previous_eligible = eligible.copy()
    return PortfolioExecution(
        ledger=pd.DataFrame(records, index=X.index),
        targets=pd.DataFrame(target_rows, index=X.index, columns=names),
        trades=pd.DataFrame(trade_rows, index=X.index, columns=names),
        ending_weights=pd.DataFrame(ending_rows, index=X.index, columns=names),
    )
