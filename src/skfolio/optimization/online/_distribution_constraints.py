"""Causal distribution constraints using the existing convex risk expressions."""

# Copyright (c) 2023-2026 skfolio contributors
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from skfolio.measures import RiskMeasure
from skfolio.optimization.convex import MeanRisk
from skfolio.prior import ReturnDistribution
from skfolio.utils.tools import args_names

RISK_LIMITS = (
    RiskMeasure.STANDARD_DEVIATION,
    RiskMeasure.SEMI_VARIANCE,
    RiskMeasure.SEMI_DEVIATION,
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.WORST_REALIZATION,
    RiskMeasure.CVAR,
    RiskMeasure.EVAR,
    RiskMeasure.MAX_DRAWDOWN,
    RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasure.CDAR,
    RiskMeasure.EDAR,
    RiskMeasure.ULCER_INDEX,
    RiskMeasure.GINI_MEAN_DIFFERENCE,
)


@dataclass
class DistributionConstraintConfig:
    """Scalar limits and aligned cost inputs for one causal decision."""

    min_return: float | None = None
    risk_limits: dict[RiskMeasure, float] = field(default_factory=dict)
    min_acceptable_return: float | None = None
    risk_free_rate: float = 0.0
    cvar_beta: float = 0.95
    evar_beta: float = 0.95
    cdar_beta: float = 0.95
    edar_beta: float = 0.95
    transaction_costs: float | np.ndarray = 0.0
    management_fees: float | np.ndarray = 0.0
    previous_weights: np.ndarray | None = None

    def validate(self) -> None:
        """Check scalar limits before constructing or updating an optimizer."""
        if self.min_return is not None and (
            np.ndim(self.min_return) != 0 or not np.isfinite(self.min_return)
        ):
            raise ValueError("min_return must be a finite scalar")
        for measure, limit in self.risk_limits.items():
            if measure not in RISK_LIMITS:
                raise ValueError(f"Unsupported distribution risk limit: {measure}")
            if np.ndim(limit) != 0 or not np.isfinite(limit) or limit < 0:
                raise ValueError(f"max_{measure.value} must be finite and nonnegative")
        for name in ("risk_free_rate", "min_acceptable_return"):
            value = getattr(self, name)
            if value is not None and (np.ndim(value) != 0 or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite scalar")
        for name in ("cvar_beta", "evar_beta", "cdar_beta", "edar_beta"):
            value = getattr(self, name)
            if np.ndim(value) != 0 or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")

    def ready(self, distribution: ReturnDistribution, n_assets: int) -> bool:
        """Check fixed-universe shapes and whether causal statistics are usable."""
        returns = np.asarray(distribution.returns)
        mu = np.asarray(distribution.mu)
        if returns.ndim != 2 or returns.shape[1] != n_assets or mu.shape != (n_assets,):
            raise ValueError(
                "The causal prior distribution must match the fixed universe"
            )
        needs_sample = any(
            m != RiskMeasure.STANDARD_DEVIATION for m in self.risk_limits
        )
        if len(returns) < (2 if needs_sample else 1):
            return False
        if not np.isfinite(returns).all() or not np.isfinite(mu).all():
            return False
        if RiskMeasure.STANDARD_DEVIATION in self.risk_limits:
            covariance = np.asarray(distribution.covariance)
            if covariance.shape != (n_assets, n_assets):
                raise ValueError(
                    "The causal prior covariance must match the fixed universe"
                )
            if not np.isfinite(covariance).all():
                return False
        return True


def distribution_constraints(
    *,
    w: cp.Variable,
    distribution: ReturnDistribution | None,
    config: DistributionConstraintConfig,
) -> list[cp.Constraint]:
    """Build MeanRisk-compatible constraints without fitting another estimator."""
    config.validate()
    if distribution is None or not config.ready(distribution, w.shape[0]):
        raise ValueError(
            "Distribution constraints require a ready causal prior distribution"
        )
    if (
        np.any(np.asarray(config.transaction_costs) != 0)
        and config.previous_weights is None
    ):
        raise ValueError("Cost-aware distribution constraints require previous_weights")
    # A fresh expression context keeps caches local to this variable and decision.
    context = MeanRisk(
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        previous_weights=config.previous_weights,
        min_acceptable_return=config.min_acceptable_return,
        risk_free_rate=config.risk_free_rate,
        cvar_beta=config.cvar_beta,
        evar_beta=config.evar_beta,
        cdar_beta=config.cdar_beta,
        edar_beta=config.edar_beta,
    )
    context._set_scale_constraints(default=1.0)
    factor = cp.Constant(1.0)
    constraints = []
    if config.min_return is not None:
        expected = (
            context._cvx_expected_return(return_distribution=distribution, w=w)
            - context._cvx_transaction_cost(
                return_distribution=distribution, w=w, factor=factor
            )
            - context._cvx_management_fee(return_distribution=distribution, w=w)
        )
        constraints.append(expected >= config.min_return)
    for measure, limit in config.risk_limits.items():
        function = getattr(context, f"_{measure.value}_risk")
        inputs = {"return_distribution": distribution, "w": w, "factor": factor}
        arguments = {
            name: inputs[name] if name in inputs else getattr(context, name)
            for name in args_names(function)
        }
        risk, auxiliary = function(**arguments)
        constraints.extend(auxiliary)
        constraints.append(risk <= limit)
    return constraints
