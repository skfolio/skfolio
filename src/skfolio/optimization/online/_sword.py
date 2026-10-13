"""Experimental SWORD expert aggregation for online portfolio selection."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.optimization.online._foco import EtaSchedule, FirstOrderOCO
from skfolio.optimization.online._projection import BaseProjector


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
