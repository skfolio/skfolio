"""First-order Online Convex Optimization engine."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from skfolio.optimization.online._mirror_maps import (
    BaseMirrorMap,
    DynamicMirrorMap,
)
from skfolio.optimization.online._projection import BaseProjector

Predictor = Callable[[int, np.ndarray | None, np.ndarray | None], np.ndarray]
EtaSchedule = float | np.ndarray | Callable[[int], float]


class FirstOrderOCO:
    r"""Implements a unified engine for first order online convex optimization algorithms supporting both Online Mirror Descent (OMD) and Follow-the-Regularized-Leader (FTRL) through COMID (Composite-Objective Mirror Descent); Duchi et al., COLT 2010).
    The helper `_composite_update` solves min ⟨η_t ĝ, w⟩ + ψ(w) with ψ un-linearised, exactly the COMID step.

    This engine provides the core logic for first-order online convex optimization
    algorithms, framed through the modern lens of FTRL-Proximal and dual averaging
    that connects OMD and FTRL. It can be configured to run in either mode and
    supports optimistic updates via gradient predictors and flexible learning rate
    schedules.

    OMD vs. FTRL
    ------------
    Two update modes are supported:
    - **OMD** (Online Mirror Descent) takes a local proximal step around the current iterate `x_t` using the
      current gradient `g_t`. Its update is:
      :math:`x_{t+1} = \arg\min_{x \in K} \{ \eta_t \langle g_t, x \rangle + D_\psi(x, x_t) \}`

    - **FTRL** (Follow-the-Regularized-Leader) chooses the point that minimizes the regularized cumulative linearized loss over all past gradients:
      :math:`x_{t+1} = \arg\min_{x \in K} \{ \sum_{s=1}^{t} \langle g_s, x \rangle + \frac{1}{\eta_t}\psi(x) \}`
      This implementation uses a formulation where eta is multiplied with the linear term:
      :math:`x_{t+1} = \arg\min_{x \in K} \{ \eta_t \sum_{s=1}^{t} \langle g_s, x \rangle + \psi(x) \}`

    Equivalence
    -----------
    OMD and FTRL can be made mathematically equivalent. This engine implements the
    FTRL-Proximal formulation, which can exactly reproduce OMD by choosing a
    time-varying proximal regularizer. The core solver method
    `_composite_update` implements this unified objective.

    Optimistic Updates
    ------------------
    The engine supports optimistic updates following Rakhlin & Sridharan (2013).
    When a predictor is provided, the effective gradient at round t becomes:
        g_eff = g_t - tilde{g}_t + tilde{g}_{t+1}
    where:
    - g_t is the actual gradient at round t
    - tilde{g}_t is the prediction made at round t-1 (stored from previous round)
    - tilde{g}_{t+1} is the new prediction made at round t (returned by predictor)

    This formulation ensures the regret bound scales with ||g_t - tilde{g}_t||^2
    (prediction error) rather than ||g_t||^2, providing better performance when
    gradients are predictable.

    **OMD mode**: Uses the full effective gradient in the local proximal step
    **FTRL mode**: The telescoping sum handles the prediction terms implicitly
                   through cumulative gradient tracking

    See Orabona (2020+), "A Modern Introduction to Online Learning", Chapter 6,
    Section 6.9 for detailed analysis.
    See also Joulani et al. (2017) "A modular analysis of adaptive (non-)convex
    optimization: Optimism, composite objectives, and variational bounds"
    for the original algorithm.

    Learning Rate Schedule
    ----------------------
    The learning rate `eta` can be a constant float, a pre-computed numpy array,
    or a callable `lambda t: f(t)` that returns the learning rate for round `t`.

    References
    ----------
    .. [1] McMahan, H. B. (2011). "Follow-the-Regularized-Leader and Mirror
           Descent: Equivalence Theorems and L1 Regularization". PMLR.

    .. [2] Xiao, L. (2009). "Dual Averaging Method for Regularized Stochastic
           Learning and Online Optimization". JMLR.

    .. [3] Chiang, C., et al. "Online optimization with gradual variations." JMLR 2012.
    """

    def __init__(
        self,
        mirror_map: BaseMirrorMap,
        projector: BaseProjector,
        eta: EtaSchedule = 1.0,
        predictor: Predictor | None = None,
        mode: Literal["omd", "ftrl"] = "omd",
        skip_auto_update: bool = False,
        discount: float | None = None,
    ) -> None:
        self.map = mirror_map
        self.projector = projector
        self.eta = eta
        self.predictor = predictor
        self.mode = mode
        self.skip_auto_update = skip_auto_update
        self.discount = discount
        self._t = 0
        self._G_sum: np.ndarray | None = None
        self._x_t: np.ndarray | None = None
        self._last_grad: np.ndarray | None = None
        self._prev_prediction: np.ndarray | None = None

    def _get_eta(self, t: int) -> float:
        """Evaluate the learning-rate schedule at observation t."""
        if callable(self.eta):
            return float(np.asarray(self.eta(t)))
        if isinstance(self.eta, np.ndarray):
            if t < len(self.eta):
                return float(self.eta[t])
            return float(self.eta[-1])
        return float(self.eta)

    def _composite_update(
        self, eta_grad_term: np.ndarray, reg_center: np.ndarray | None
    ) -> np.ndarray:
        """
        Expectation: eta_grad_term is already multiplied by eta_t (i.e. eta_grad_term = eta_t * <sum or current>).
        Solves dual: z = - eta_grad_term + grad_psi(reg_center)  (if reg_center provided)
        then x = grad_psi_star(z); follow with geometry-specific normalization & external projector.
        """
        if reg_center is None:
            z = -eta_grad_term
        else:
            z = -eta_grad_term + self.map.grad_psi(reg_center)
        x = self.map.grad_psi_star(z)
        x = self.map.project_geom(x)
        x = self.projector.project(x)
        return x

    def step(self, g: np.ndarray) -> np.ndarray:
        """Consume a gradient and return the next portfolio decision."""
        g = np.asarray(g, dtype=float)
        if self._G_sum is None:
            self._G_sum = np.zeros_like(g)

        # Ensure dimensional init for dynamic maps before any grad_psi calls
        if isinstance(self.map, DynamicMirrorMap):
            try:
                self.map.ensure_dim(g.shape[0])
            except Exception:
                pass

        if self._x_t is None:
            d = g.shape[0]
            self._x_t = self.projector.project(np.ones(d) / d)

        # Initialize previous prediction to zeros on first step when predictor is used
        if self._prev_prediction is None and self.predictor is not None:
            self._prev_prediction = np.zeros_like(g)

        eta_t = self._get_eta(self._t)

        m_t = np.zeros_like(g)
        if self.predictor is not None:
            m_t = self.predictor(self._t, self._x_t, self._last_grad)
            if m_t.size == 0:
                m_t = np.zeros_like(g)
            else:
                m_t = np.asarray(m_t, dtype=float)
                if m_t.shape != g.shape:
                    raise ValueError("Predictor returned vector with wrong shape.")

        if self.discount is not None:
            self._G_sum *= self.discount
        self._G_sum += g

        # Special handling for BurgMirrorMap (Prod/Soft-Bayes algorithm)
        # The Burg regularizer requires solving for λ in the constrained OMD problem,
        # which cannot be done through the standard grad_psi_star interface
        from skfolio.optimization.online._mirror_maps import BurgMirrorMap

        if isinstance(self.map, BurgMirrorMap):
            # Compute effective gradient for optimistic updates
            effective_grad = g.copy()
            if self._prev_prediction is not None:
                effective_grad -= self._prev_prediction
            effective_grad += m_t
            # Use specialized constrained_inverse method for Burg update
            x_next, _ = self.map.constrained_inverse(self._x_t, eta_t, effective_grad)
            # Apply external projector for additional constraints
            x_next = self.projector.project(x_next)
        elif self.mode == "omd":
            # Compute effective gradient: g_t - tilde{g}_t + tilde{g}_{t+1}
            # where tilde{g}_t is the prediction made at round t-1 (stored in _prev_prediction)
            # and tilde{g}_{t+1} is the new prediction made at round t (m_t)
            effective_grad = g.copy()
            if self._prev_prediction is not None:
                effective_grad -= self._prev_prediction
            effective_grad += m_t
            lin = eta_t * effective_grad
            if isinstance(self.map, DynamicMirrorMap):
                _ = self.map.grad_psi(self._x_t)  # preserves Spy test (pre-geometry)
                center = self.map.grad_psi_after(
                    self._x_t, g
                )  # aggregated center (H_{t+1})
                z = -lin + center
                x_next = self.map.grad_psi_star_after(
                    z, g
                )  # aggregated inverse (H_{t+1})
                x_next = self.map.project_geom(x_next)
                x_next = self.projector.project(x_next)
            else:
                x_next = self._composite_update(eta_grad_term=lin, reg_center=self._x_t)

        elif self.mode == "ftrl":
            lin = eta_t * (self._G_sum + m_t)
            x_next = self._composite_update(eta_grad_term=lin, reg_center=None)
        else:
            raise ValueError("mode must be 'omd' or 'ftrl'")

        # Update mirror map state unless skip_auto_update is True
        # (used for Ada-BARRONS which needs manual component-specific updates)
        if not self.skip_auto_update and isinstance(self.map, DynamicMirrorMap):
            self.map.update_state(g)

        self._last_grad = g.copy()

        # Update previous prediction for next round (optimistic OMD)
        if self.predictor is not None:
            self._prev_prediction = m_t.copy()

        self._x_t = x_next
        self._t += 1

        return x_next


class SwordMeta:
    r"""
    Meta-aggregator for SWORD experts using exponential weights.

    - Maintains K experts (FTRL engines), e.g., [SWORD-Var, SWORD-Small, (EG for SWORD++)].
    - At round t, with observed gradient g_t and experts' current decisions x_t^{(k)},
      updates alpha via alpha_k <- alpha_k * exp(-eta_meta * <g_t, x_t^{(k)}>), normalize.
    - Advances each expert with .step(g_t) to get x_{t+1}^{(k)}.
    - Plays the projected mixture x_{t+1} = Proj( sum_k alpha_k x_{t+1}^{(k)} ).

    Notes
    -----
    - Experts are created with IdentityProjector to avoid constraint duplication.
    - Final mixture is projected using the full AutoProjector to satisfy constraints.
    """

    def __init__(
        self,
        experts: list[FirstOrderOCO],
        projector: BaseProjector,
        eta_meta: EtaSchedule = 1.0,
    ) -> None:
        if len(experts) < 2:
            raise ValueError("SwordMeta requires at least 2 experts.")
        self.experts = experts
        self.projector = projector
        self.eta_meta = eta_meta
        self._alpha: np.ndarray | None = None
        self._t: int = 0

    def _get_eta_meta(self, t: int) -> float:
        """Evaluate the mixture learning-rate schedule at observation t."""
        if callable(self.eta_meta):
            return float(np.asarray(self.eta_meta(t)))
        if isinstance(self.eta_meta, np.ndarray):
            if t < len(self.eta_meta):
                return float(self.eta_meta[t])
            return float(self.eta_meta[-1])
        return float(self.eta_meta)

    def step(self, g: np.ndarray) -> np.ndarray:
        """Consume a gradient and return the next portfolio decision."""
        g = np.asarray(g, dtype=float)
        d = g.shape[0]
        K = len(self.experts)
        if self._alpha is None:
            self._alpha = np.ones(K, dtype=float) / float(K)

        # Ensure experts have a current iterate; if not, set to uniform feasible seed
        for e in self.experts:
            if e._x_t is None:
                e._x_t = np.ones(d, dtype=float) / float(d)

        # Compute linearized losses at current decisions x_t^{(k)}
        losses = np.array([float(np.dot(g, e._x_t)) for e in self.experts], dtype=float)

        # Exponential-weights update for alpha
        eta_m = self._get_eta_meta(self._t)
        # numerically stable update
        u = -eta_m * losses
        u -= np.max(u)
        new_alpha_raw = self._alpha * np.exp(u)
        s = float(new_alpha_raw.sum())
        self._alpha = (
            (new_alpha_raw / s) if s > 0 else (np.ones_like(new_alpha_raw) / K)
        )

        # Advance each expert to its x_{t+1}^{(k)} under g_t
        X_next = [e.step(g) for e in self.experts]  # each is shape (d,)

        # Mixture and final projection
        w_mix = np.sum(
            np.array([a * x for a, x in zip(self._alpha, X_next, strict=False)]), axis=0
        )
        w_proj = self.projector.project(w_mix)
        self._t += 1
        return w_proj
