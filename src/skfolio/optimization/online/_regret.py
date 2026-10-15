"""Regret utilities for online portfolio selection."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._base import BaseOnlineOptimization
from skfolio.utils.tools import AutoEnum


class RegretType(AutoEnum):
    """Available historical comparator conventions."""

    STATIC = "static"
    DYNAMIC_UNIVERSAL = "dynamic_universal"
    DYNAMIC_WORST_CASE = "dynamic_worst_case"


def plot_regret_curve(
    regret: ArrayLike,
    *,
    fig: go.Figure | None = None,
    average: bool = True,
    label: str | None = None,
) -> go.Figure:
    """Plot a regret curve over time using plotly.

    Parameters
    ----------
    regret : array-like of shape (T,)
        Regret values per time step.
    fig : plotly figure, optional
        Figure to plot on; created if None.
    average : bool, default=True
        If True, the label defaults to 'Average regret'; otherwise 'Regret'.
    label : str, optional
        Custom label for the curve.
    """
    r = np.asarray(regret, dtype=float)
    if fig is None:
        fig = go.Figure()
    if label is None:
        label = "Average regret" if average else "Regret"

    fig.add_trace(go.Scatter(x=np.arange(1, r.size + 1), y=r, mode="lines", name=label))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=label,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.3)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.3)"),
        showlegend=True,
    )

    return fig


def _losses_from_weights(
    relatives: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorized per-period negative log-wealth losses.

    relatives : (T, n), weights : (T, n) or (n,)
    Returns: (T,)
    """
    R = np.asarray(relatives, dtype=float)
    W = np.asarray(weights, dtype=float)
    if W.shape not in ((R.shape[1],), R.shape):
        raise ValueError("weights must have shape (n_assets,) or (T, n_assets)")
    gross = 1.0 + np.sum(W * (R - 1.0), axis=1)
    if not np.all(np.isfinite(gross)) or np.any(gross <= 0):
        raise ValueError("Portfolio gross returns must be finite and strictly positive")
    return -np.log(gross)


def _running_regret(
    online_losses: np.ndarray,
    comp_losses: np.ndarray,
    *,
    average: bool | str = False,
    window: int | None = None,
) -> np.ndarray:
    """Compute running (or windowed) regret curve with optional averaging.

    average can be:
    - False or "none": no averaging (default)
    - True or "running": divide by t (or by window size when windowed)
    - "final": return a constant array equal to the final average value
    """
    if average not in (False, True, "none", "running", "final"):
        raise ValueError("Unknown average mode")
    T = online_losses.size
    avg_mode = (
        "running"
        if average is True
        else (average if isinstance(average, str) else "none")
    )

    if window is not None:
        if (
            isinstance(window, bool)
            or not isinstance(window, (int, np.integer))
            or not (1 <= window <= T)
        ):
            raise ValueError("window must be between 1 and T.")
        wr_raw = np.zeros(T, dtype=float)
        wsize = int(window)
        for t in range(wsize - 1, T):
            o = float(np.sum(online_losses[t - wsize + 1 : t + 1]))
            c = float(np.sum(comp_losses[t - wsize + 1 : t + 1]))
            wr_raw[t] = o - c
        if avg_mode == "running":
            return wr_raw / float(wsize)
        if avg_mode == "final":
            final_avg = wr_raw[-1] / float(wsize)
            return np.full(T, final_avg, dtype=float)
        return wr_raw

    # No window: cumulative
    rr_raw = np.cumsum(online_losses - comp_losses)
    if avg_mode == "running":
        return rr_raw / np.arange(1, T + 1, dtype=float)
    if avg_mode == "final":
        final_avg = rr_raw[-1] / float(T)
        return np.full(T, final_avg, dtype=float)
    return rr_raw


def _worst_case_dynamic_weights(relatives: np.ndarray) -> np.ndarray:
    """One-hot comparators on argmax asset per round for log-wealth loss.

    relatives: (T, n) strictly positive gross relatives.
    Return W*: (T, n) with one-hot rows. Ensures per-round minimal loss.
    """
    R = np.asarray(relatives, dtype=float)
    if R.ndim != 2:
        raise ValueError("relatives must be (T, n)")
    idx = np.argmax(R, axis=1)
    T, n = R.shape
    W = np.zeros((T, n), dtype=float)
    W[np.arange(T), idx] = 1.0
    return W


def _solve_best_constant_weights(
    relatives: np.ndarray,
    *,
    init: np.ndarray | None = None,
    tol: float = 1e-9,
    max_iter: int = 2000,
) -> np.ndarray:
    r"""Compute best constant rebalanced portfolio (BestConstantRebalancedPortfolio) for log wealth."""
    R = np.asarray(relatives, dtype=float)
    if R.ndim != 2:
        raise ValueError("relatives must have shape (T, n)")
    T, n = R.shape
    if T == 0 or n == 0:
        return np.zeros(n, dtype=float)

    weights = cp.Variable(n)
    if init is not None:
        initial = np.asarray(init, dtype=float)
        if initial.shape != (n,):
            raise ValueError("init must have shape (n_assets,)")
        weights.value = initial
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.log(R @ weights))),
        [weights >= 0, cp.sum(weights) == 1],
    )
    problem.solve(
        solver="CLARABEL",
        max_iter=max_iter,
        tol_gap_abs=tol,
        tol_gap_rel=tol,
        tol_feas=tol,
    )
    if problem.status != cp.OPTIMAL:
        import warnings

        warnings.warn(
            f"BestConstantRebalancedPortfolio solver did not converge to optimal status: {problem.status}",
            UserWarning,
            stacklevel=2,
        )
    if weights.value is None:
        raise RuntimeError("BestConstantRebalancedPortfolio optimization failed")
    # Remove tiny solver residuals at the simplex boundary.
    result = np.maximum(np.asarray(weights.value, dtype=float), 0.0)
    return result / result.sum()


def _universal_dynamic_weights(
    relatives: np.ndarray,
    *,
    path_length: float | None = None,
    path_penalty: float | None = None,
    norm: str = "l1",
    solver: str = "CLARABEL",
    solver_params: dict | None = None,
) -> np.ndarray:
    r"""
    Solve universal dynamic comparator sequence u_1..u_T on the simplex.

        maximize     sum_t log( r_t^T u_t )
        subject to   u_t >= 0, sum_i u_{t,i} = 1  for each t
                     sum_{t=2}^T ||u_t - u_{t-1}||_p <= path_length    (constraint)
                     or penalized with path_penalty * sum ||u_t - u_{t-1}||_p

    - Use norm='l1' (default) or 'l2'. L1 matches many path-length results in the literature.
    - If path_length=0, this reduces to STATIC BestConstantRebalancedPortfolio (one u_t constant), which we can detect.

    This implements the canonical universal dynamic regret comparator used in OCO,
    where non-stationarity is captured by the path-length PT, a key regularity in
    dynamic regret bounds for OGD and SWORD families.

    References
    ----------
    Zhao, Peng et al, "Adaptivity and Non-stationarity: Problem-dependent Dynamic Regret for Online Convex Optimization", JMLR 25 (2024), Eq. 2
    """
    R = np.asarray(relatives, dtype=float)
    if R.ndim != 2:
        raise ValueError("relatives must be (T, n) gross relatives")
    T, n = R.shape
    if T == 0:
        return np.zeros((0, n), dtype=float)
    if norm not in ("l1", "l2"):
        raise ValueError("norm must be 'l1' or 'l2'")
    if path_length is None and path_penalty is None:
        raise ValueError(
            "Provide either path_length or path_penalty for universal dynamic comparator."
        )
    if path_length is not None and path_penalty is not None:
        raise ValueError("Provide only one of path_length and path_penalty")
    if path_length is not None and (not np.isfinite(path_length) or path_length < 0):
        raise ValueError("path_length must be nonnegative.")
    if path_penalty is not None and (not np.isfinite(path_penalty) or path_penalty < 0):
        raise ValueError("path_penalty must be nonnegative.")

    # Special case: path_length == 0 -> STATIC comparator (one constant u) = BestConstantRebalancedPortfolio
    if path_length is not None and path_length <= 1e-18:
        w = _solve_best_constant_weights(R)
        return np.repeat(w[None, :], T, axis=0)

    # Decision variable: one weight vector per t
    W = cp.Variable((T, n))
    constraints: list[cp.constraints.Constraint] = [W >= 0, cp.sum(W, axis=1) == 1]

    diffs = W[1:, :] - W[:-1, :] if T >= 2 else None
    if diffs is not None:
        if norm == "l1":
            path_expr = cp.sum(cp.norm1(diffs, axis=1))
        else:
            path_expr = cp.sum(cp.norm2(diffs, axis=1))
    else:
        path_expr = cp.Constant(0.0)

    # Objective: maximize sum log(R[t]^T W_t). This is concave; maximizing is DCP-valid.
    # R is strictly positive, so we don't need extra positivity constraints for the affine inner product.
    obj_terms = [cp.log(R[t, :] @ W[t, :]) for t in range(T)]
    if path_length is not None:
        constraints.append(path_expr <= float(path_length))
        objective = cp.Maximize(cp.sum(obj_terms))
    else:
        assert path_penalty is not None
        objective = cp.Maximize(cp.sum(obj_terms) - float(path_penalty) * path_expr)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver or "CLARABEL", **(solver_params or {}))

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or W.value is None:
        raise RuntimeError(
            "Universal dynamic comparator optimization failed to converge."
        )

    return np.asarray(W.value, dtype=float)


def regret(
    estimator: BaseOptimization,
    X: ArrayLike,
    *,
    comparator: BaseOptimization | None = None,
    regret_type: RegretType | str = RegretType.STATIC,
    average: bool | str = False,
    window: int | None = None,
    dynamic_config: dict | None = None,
) -> np.ndarray:
    """Compute regret curves over time using the negative log-wealth loss: ft(w) = -log(w^T r_t).

    Parameters
    ----------
    estimator : BaseOptimization
        Online estimator evaluated from its initial allocation on a copy using
        ``fit_predict``. Other estimators must expose pre-observation
        ``all_weights_`` of shape (T, n) after fitting. Inputs are not mutated.
    X : array-like of shape (T, n)
        Net returns per period. Internally converted to gross relatives via `1 + r`.
    comparator : BaseOptimization, optional
        For STATIC, fits the supplied comparator on a copy. Equal-weight
        comparisons can use :class:`~skfolio.optimization.EqualWeighted` directly;
        no ``partial_fit`` or wealth history is required. If None, solves
        the best fixed long-only, fully invested log-wealth portfolio.
        Dynamic modes use their own comparators and ignore this argument.
    regret_type : RegretType or str, default=RegretType.STATIC
        STATIC, DYNAMIC_UNIVERSAL or DYNAMIC_WORST_CASE, or their string values.
    average : {False, "none", True, "running", "final"}, default=False
        Averaging mode for the returned curve:
        - False or "none": return cumulative (or windowed) regret curve
        - True or "running": return running-average curve (divide by t, or by window)
        - "final": return a constant array equal to final average (R_T/T, or last window average)
    window : int, optional
        If provided, compute sliding-window regret with the given window size.
    dynamic_config : dict, optional
        Additional options for universal dynamic regret comparator:
            - path_length: float >= 0           (sum_t ||u_t - u_{t-1}||_p <= path_length)
            - path_penalty: float >= 0          (Lagrangian penalty instead of constraint)
            - norm: "l1" (default) or "l2"      (p in the path length)
            - solver: str or None               (cvxpy solver; default "CLARABEL")
            - solver_params: dict               (forwarded to cvxpy)

    Notes
    -----
    Online evaluation includes configured costs and actual drifted holdings.
    Hindsight comparators are evaluated gross of costs, with residual cash earning
    zero. They do not inherit the online estimator's constraints. This is an
    empirical comparison, not a proof of a regret bound. A supplied comparator
    contributes its weights only, not its cost model. Penalty mode selects the
    comparator with a path penalty but reports unpenalized log-wealth differences.
    Sliding windows sum these same losses; comparators are not refitted per window.

    - For path_length=0, DYNAMIC_UNIVERSAL reduces to STATIC comparator (BestConstantRebalancedPortfolio).
    - For L1 norm and sufficiently large path_length (≥ 2*(T-1)), the universal dynamic comparator approaches the worst-case dynamic comparator (can change fully every round).

    Returns
    -------
    ndarray of shape (T,)
        Regret curve (cumulative or running-averaged, or windowed).

    References
    ----------
    - Static regret, universal dynamic regret, and worst-case dynamic regret definitions are standard
      in OCO (see, e.g., Zhao et al. 2024, Eqs. (1)-(3)) (https://jmlr.org/papers/volume25/21-0748/21-0748.pdf).
    - Path length PT is the canonical non-stationarity measure in dynamic regret bounds; SWORD
      algorithms achieve problem-dependent bounds in terms of PT and gradient variation VT (and small loss FT) under smoothness, within the universal dynamic regret framework.
    """
    rt = RegretType(regret_type)

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.all(np.isfinite(values)):
        raise ValueError("X must be a nonempty finite 2D return matrix")
    if np.any(values <= -1):
        raise ValueError(
            "Returns must be greater than -1; bankrupt assets are unsupported"
        )
    X = X if hasattr(X, "shape") else values
    relatives = 1.0 + values
    est = deepcopy(estimator)
    if isinstance(est, BaseOnlineOptimization):
        portfolio = est.fit_predict(X)
        gross = 1.0 + np.asarray(portfolio.returns)
        if not np.all(np.isfinite(gross)) or np.any(gross <= 0):
            raise ValueError(
                "Portfolio gross returns must be finite and strictly positive"
            )
        online_losses = -np.log(gross)
    else:
        est.fit(X)
        if not hasattr(est, "all_weights_"):
            raise ValueError("estimator must expose pre-observation all_weights_")
        weights = np.asarray(est.all_weights_, dtype=float)
        if weights.shape != values.shape:
            raise ValueError("all_weights_ must have shape (T, n_assets)")
        online_losses = _losses_from_weights(relatives, weights)

    # Comparator weights and losses
    if rt == RegretType.STATIC:
        if comparator is not None:
            comp = deepcopy(comparator).fit(X)
            w_star = np.asarray(comp.weights_, dtype=float)
        else:
            w_star = _solve_best_constant_weights(relatives)
        comp_losses = _losses_from_weights(relatives, w_star)

    elif rt == RegretType.DYNAMIC_WORST_CASE:
        W_wc = _worst_case_dynamic_weights(relatives)
        comp_losses = _losses_from_weights(relatives, W_wc)

    elif rt == RegretType.DYNAMIC_UNIVERSAL:
        cfg = dynamic_config or {}
        solver = cfg.get("solver", None)
        solver_params = cfg.get("solver_params", None)
        W_ud = _universal_dynamic_weights(
            relatives,
            path_length=cfg.get("path_length", None),
            path_penalty=cfg.get("path_penalty", None),
            norm=cfg.get("norm", "l1"),
            solver=solver or "CLARABEL",
            solver_params=solver_params,
        )
        comp_losses = _losses_from_weights(relatives, W_ud)

    else:
        raise ValueError(
            "Unknown regret_type. Use STATIC, DYNAMIC_UNIVERSAL or DYNAMIC_WORST_CASE."
        )

    # Running (or windowed) regret curve
    return _running_regret(
        online_losses=online_losses,
        comp_losses=comp_losses,
        average=average,
        window=window,
    )
