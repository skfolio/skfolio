"""Exponentiated Gradient portfolio allocation."""

# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import skfolio.typing as skt
from skfolio.optimization.online._base import BaseOnlineOptimization
from skfolio.optimization.online._eg_update import entropy_update, project_entropy
from skfolio.typing import ArrayLike, FloatArray


class ExponentiatedGradient(BaseOnlineOptimization):
    """Entropy mirror-descent EG with an optional bounded-simplex constraint.

    For price relatives x = 1 + r and previous target b::

        b_next[i] proportional to b[i] * exp(eta * x[i] / (b @ x)).

    learning_rate is a finite nonnegative scalar or callable of the zero-based
    observation count (including warmup). This is OMD, not Carlo's default FTRL.
    min_weights/max_weights use skfolio's scalar, array or asset-name dictionary
    convention. They are fixed during a stream; call fit to change them.

    Parameters
    ----------
    learning_rate : float or callable, default=0.05
        Nonnegative step size or a pure function of the zero-based observation
        count, including warmup. Callable results must be finite and nonnegative.
    initial_weights : array-like of shape (n_assets,), optional
        Strictly positive reference allocation summing to one, projected into the
        bounds before the first observation. Defaults to equal weights.
    previous_weights : float, dict or array-like, optional
        Actual holdings forwarded to portfolio evaluation. Never used as the
        algorithm's previous target.
    portfolio_params : dict, optional
        Portfolio evaluation parameters, for example ``{"weight_drift": True}``.
    min_weights : float, dict or array-like, default=0.0
        Lower bounds. Dictionary keys are asset names; unspecified assets use zero.
    max_weights : float, dict or array-like, default=1.0
        Upper bounds. Dictionary keys are asset names; unspecified assets use one.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Target for the period following the last observed return.
    initial_weights_ : ndarray of shape (n_assets,)
        Initial allocation after projection into the bounds.
    n_observations_ : int
        Number of successfully processed observations, including warmup.
    min_weights_ : ndarray of shape (n_assets,)
        Resolved lower bounds.
    max_weights_ : ndarray of shape (n_assets,)
        Resolved upper bounds.

    Notes
    -----
    Only a fixed asset universe, finite returns greater than -1, and a long-only
    unit budget are supported. Bounds use a KL projection, not an L2 projection.
    A 1e-16 floor before taking logarithms allows numerically zero reference
    weights to recover; this is not exact support preservation.

    Use :func:`~skfolio.model_selection.online_predict` with ``test_size=1`` for
    causal evaluation. Inherited ``predict`` holds the final target; inherited
    ``fit_predict`` is not a sequential backtest. Risk constraints, turnover
    limits, optional priors and fallback recovery are not implemented.

    The update structure follows the entropy mirror-descent configuration in
    Carlo Nicolini's online portfolio work. Its default FTRL configuration has
    different semantics, particularly with nonuniform initial allocations.

    Examples
    --------
    >>> from collections.abc import Callable

    import numpy as np

    import skfolio.typing as skt
    >>> from skfolio.optimization import ExponentiatedGradient
    >>> X = np.array([[0.01, -0.01], [0.02, 0.0]])
    >>> model = ExponentiatedGradient().fit(X)
    >>> bool(model.weights_[0] > model.weights_[1])
    True
    """

    def __init__(
        self,
        *,
        learning_rate: float | Callable[[int], float] = 0.05,
        initial_weights: ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        portfolio_params: dict | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
    ) -> None:
        """Initialize the update rule, allocation bounds and evaluation parameters."""
        super().__init__(initial_weights, previous_weights, portfolio_params)
        self.learning_rate = learning_rate
        self.min_weights = min_weights
        self.max_weights = max_weights

    def _validate_params(self) -> None:
        """Validate the constant rate; scheduled rates are checked per update."""
        if callable(self.learning_rate):
            return
        self._validate_rate(self.learning_rate)

    @staticmethod
    def _validate_rate(rate: object) -> None:
        """Require a finite nonnegative learning rate."""
        if (
            isinstance(rate, (bool, np.bool_))
            or not isinstance(rate, (int, float, np.integer, np.floating))
            or not np.isfinite(rate)
            or rate < 0
        ):
            raise ValueError("learning_rate must be finite and nonnegative.")

    def _initialize(self, n_assets: int) -> None:
        """Validate bounds and project the initial allocation in KL geometry."""
        lower = self._bounds(self.min_weights, n_assets, 0, "min_weights")
        upper = self._bounds(self.max_weights, n_assets, 1, "max_weights")
        if (
            np.any(lower < 0)
            or np.any(upper > 1)
            or np.any(lower > upper)
            or lower.sum() > 1
            or upper.sum() < 1
        ):
            raise ValueError("Weight bounds must define a feasible long-only simplex.")
        super()._initialize(n_assets)
        self.weights_ = project_entropy(np.log(self.weights_), lower, upper)
        self.initial_weights_ = self.weights_.copy()
        self.min_weights_ = lower
        self.max_weights_ = upper

    def _bounds(
        self, value: skt.MultiInput | None, n_assets: int, default: float, name: str
    ) -> FloatArray:
        """Resolve skfolio scalar, array or named bounds to an asset vector."""
        value = default if value is None else value
        cleaned = self._clean_input(
            value, n_assets=n_assets, fill_value=default, name=name
        )
        result = np.asarray(cleaned, dtype=float)
        if result.ndim == 0:
            result = np.full(n_assets, result.item())
        if result.shape != (n_assets,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain finite bounds for each asset.")
        return result.copy()

    def _validate_stream_params(self) -> None:
        """Reject changed bounds until the estimator is explicitly refitted."""
        for name, default in (("min_weights", 0), ("max_weights", 1)):
            current = self._bounds(
                getattr(self, name), self.n_features_in_, default, name
            )
            if not np.array_equal(current, getattr(self, name + "_")):
                raise ValueError(f"{name} changed during a stream; call fit to reset.")

    _state_attributes = (
        *BaseOnlineOptimization._state_attributes,
        "min_weights_",
        "max_weights_",
    )

    def _solve_update(self, returns_t: FloatArray) -> FloatArray:
        """Return an entropy mirror-descent step without advancing state."""
        rate = (
            self.learning_rate(self.n_observations_)
            if callable(self.learning_rate)
            else self.learning_rate
        )
        self._validate_rate(rate)
        return entropy_update(
            self.weights_, returns_t, rate, self.min_weights_, self.max_weights_
        )
