"""Custom typing module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from skfolio.measures import ExtraRiskMeasure, PerfMeasure, RatioMeasure, RiskMeasure

if TYPE_CHECKING:
    from skfolio.optimization._base import BaseOptimization

__all__ = [
    "CvxMeasure",
    "ExpressionFunction",
    "Factor",
    "Fallback",
    "Groups",
    "Inequality",
    "LinearConstraints",
    "Measure",
    "MultiInput",
    "Names",
    "ParametersValues",
    "Result",
    "RiskResult",
    "Tags",
    "Target",
]

Measure = PerfMeasure | RiskMeasure | ExtraRiskMeasure | RatioMeasure
CvxMeasure = PerfMeasure | RiskMeasure | RatioMeasure
MultiInput = float | dict[str, float] | npt.ArrayLike
Groups = dict[str, list[str]] | np.ndarray | list[list[str]]
LinearConstraints = np.ndarray | list[str]
Inequality = np.ndarray | list
Target = float | np.ndarray
ParametersValues = list[tuple[cp.Parameter, float | np.ndarray]]
Factor = cp.Variable | cp.Constant
Result = np.ndarray | tuple[float | tuple[float, float] | np.ndarray, np.ndarray]
RiskResult = tuple[
    cp.Expression | cp.Variable | cp.trace, list[cp.Expression | cp.SOC | cp.PSD]
]
ExpressionFunction = Callable[[cp.Variable, any], cp.Expression]
Figure = go.Figure
Fallback: TypeAlias = Union[
    "BaseOptimization",
    list[Union["BaseOptimization", Literal["previous_weights"]]],
    Literal["previous_weights"],
    None,
]

# Population
Names = str | list[str]
Tags = str | list[str]
