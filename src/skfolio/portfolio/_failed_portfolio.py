"""FailedPortfolio sentinel class."""


# Copyright (c) 2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt

import skfolio.typing as skt
from skfolio.portfolio._portfolio import Portfolio


class FailedPortfolio(Portfolio):
    r"""
    Portfolio object returned when an optimization step fails. It acts as a sentinel
    value that marks the failure and stores failure diagnostics (`optimization_error`,
    `fallback_chain`).

    `FailedPortfolio` preserves full API compatibility with `Portfolio` so it
    can seamlessly pass through risk measures, aggregation, rolling computations and
    plotting without raising. All returns, weights, composition, and derived measures
    are NaN.

    .. note::

        In backtesting workflows, when an optimization estimator is configured
        with `raise_on_failure=False`, a `FailedPortfolio` is returned on failed
        rebalancings. This lets the process complete without raising while preserving
        the full timeline for downstream analysis and diagnostics.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.
        If `X` is a DataFrame, the columns will be considered as assets names
        and the indices will be considered as observations. Otherwise, we use
        `["x0", "x1", ..., "x(n_assets - 1)"]` as asset names and
        `[0, 1, ..., n_observations]` as observations.

    optimization_error : str, optional
        Stringified error message explaining why the optimization failed.
        Propagated from the optimization estimator when `raise_on_failure=False`.
        `None` means the reason is unknown or not provided.

    name : str, optional
        Name of the portfolio. The default (`None`) is to use the object id.

    tag : str, optional
        Tag given to the portfolio. Tags are used to manipulate groups of
        Portfolios from a `Population`.

    fallback_chain : list[tuple[str, str]] | None, optional
        Sequence describing the optimization fallback attempts. Each element is
        a pair `(estimator_repr, outcome)` where:

        * `estimator_repr` is the string representation of the primary
          estimator or a fallback (e.g. `"EqualWeighted()"`,
          `"previous_weights"`).
        * `outcome` is `"success"` if that step produced a valid solution,
          otherwise the stringified error message.

        For successful fits without any fallback, this is `None`. When
        fallbacks are provided and the primary fails, the chain starts with
        `(primary_repr, primary_error)` and is followed by one entry per
        fallback that was attempted, ending with the first `"success"` or the
        last error if all fail. This is set by the optimization estimator and
        propagated to the resulting portfolio objects (including
        `FailedPortfolio`).

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    risk_free_rate : float, default=0.0
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    annualized_factor : float, default=252.0
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    fitness_measures : list[measures], optional
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    compounded : bool, default=False
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    sample_weight : ndarray of shape (n_observations, ), optional
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    min_acceptable_return : float | None, optional
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    value_at_risk_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    entropic_risk_measure_theta : float, default=1.0
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    entropic_risk_measure_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    cvar_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    evar_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    drawdown_at_risk_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    cdar_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    edar_beta : float, default=0.95
        Accepted for API compatibility with `Portfolio` but not used by
        `FailedPortfolio`.

    Notes
    -----
    All performance, risk, and contribution measures are computed from NaN
    returns and NaN weights in a `FailedPortfolio`. As a result, these
    parameters do not affect the outcome: NaNs are carried over to metrics,
    contributions, plots, and rolling computations. This class exists solely to
    preserve API and type compatibility while signaling a failed optimization.
    """

    __slots__ = {
        # read-write
        "optimization_error",
    }

    def __init__(
        self,
        X: npt.ArrayLike,
        name: str | None = None,
        tag: str | None = None,
        optimization_error: str | None = None,
        fallback_chain: list[tuple[str, str]] | None = None,
        previous_weights: skt.MultiInput = None,
        transaction_costs: skt.MultiInput = None,
        management_fees: skt.MultiInput = None,
        risk_free_rate: float = 0,
        annualized_factor: float = 252,
        fitness_measures: list[skt.Measure] | None = None,
        compounded: bool = False,
        sample_weight: np.ndarray | None = None,
        min_acceptable_return: float | None = None,
        value_at_risk_beta: float = 0.95,
        entropic_risk_measure_theta: float = 1,
        entropic_risk_measure_beta: float = 0.95,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        drawdown_at_risk_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
    ):
        super().__init__(
            X=X,
            weights=None,
            previous_weights=previous_weights,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            risk_free_rate=risk_free_rate,
            name=name,
            tag=tag,
            annualized_factor=annualized_factor,
            fitness_measures=fitness_measures,
            compounded=compounded,
            sample_weight=sample_weight,
            min_acceptable_return=min_acceptable_return,
            value_at_risk_beta=value_at_risk_beta,
            entropic_risk_measure_theta=entropic_risk_measure_theta,
            entropic_risk_measure_beta=entropic_risk_measure_beta,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            drawdown_at_risk_beta=drawdown_at_risk_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            fallback_chain=fallback_chain,
        )

        self.optimization_error = optimization_error
