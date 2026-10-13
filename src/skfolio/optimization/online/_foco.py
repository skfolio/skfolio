"""First-order Online Convex Optimization engine."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
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
from skfolio.optimization.online._projection import BaseProjector, IdentityProjector

Predictor = Callable[[int, np.ndarray | None, np.ndarray | None], np.ndarray]
EtaSchedule = float | np.ndarray | Callable[[int], float]


class FirstOrderOCO:
    r"""First-order online optimization with mirror descent or FTRL updates.

    Parameters
    ----------
    mirror_map : BaseMirrorMap
        Regularizer geometry and its dual transforms.
    projector : BaseProjector
        Feasible-set projector or native constrained-update solver.
    eta : float, ndarray or callable, default=1.0
        Learning rate, indexed schedule or function of the zero-based round.
    predictor : callable, optional
        Predict a gradient from the round, current weights and previous gradient.
    mode : {"omd", "ftrl"}, default="omd"
        OMD uses a local mirror step. FTRL uses accumulated gradients.
    skip_auto_update : bool, default=False
        Leave adaptive mirror-state updates to the caller.
    discount : float, optional
        Multiply accumulated gradients by this factor before adding the new one.

    Notes
    -----
    OMD uses the effective gradient ``g + prediction - previous_prediction``.
    FTRL uses the accumulated gradient plus the current prediction. Native
    constrained solvers retain the mirror geometry in the objective.

    With adaptive maps, OMD computes its step using geometry that includes the
    current gradient. FTRL computes its decision before updating that geometry.
    The family adapter can own those updates when ``skip_auto_update=True``.

    References
    ----------
    .. [1] McMahan, H. B. (2011). "Follow-the-Regularized-Leader and Mirror
           Descent: Equivalence Theorems and L1 Regularization". PMLR.
    .. [2] Xiao, L. (2009). "Dual Averaging Method for Regularized Stochastic
           Learning and Online Optimization". JMLR.
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

        if not isinstance(self.projector, IdentityProjector):
            if self.mode == "omd":
                effective = g + m_t
                if self._prev_prediction is not None:
                    effective = effective - self._prev_prediction
                center = (
                    self.map.grad_psi_after(self._x_t, g)
                    if isinstance(self.map, DynamicMirrorMap)
                    else self.map.grad_psi(self._x_t)
                )
                dual = center - eta_t * effective
                pending = g
            elif self.mode == "ftrl":
                dual = -eta_t * (self._G_sum + m_t)
                # Preserve the source FTRL convention: current regularizer,
                # then update its state after the decision.
                pending = None
            else:
                raise ValueError("mode must be 'omd' or 'ftrl'")
            x_next = self.projector.solve_update(self.map, dual, pending)
        else:
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
                x_next, _ = self.map.constrained_inverse(
                    self._x_t, eta_t, effective_grad
                )
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
                    _ = self.map.grad_psi(
                        self._x_t
                    )  # preserves Spy test (pre-geometry)
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
                    x_next = self._composite_update(
                        eta_grad_term=lin, reg_center=self._x_t
                    )

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
