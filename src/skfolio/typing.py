"""Custom typing module."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from collections.abc import Callable

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from skfolio.measures import ExtraRiskMeasure, PerfMeasure, RatioMeasure, RiskMeasure

__all__ = [
    "Groups",
    "Inequality",
    "LinearConstraints",
    "MultiInput",
    "Target",
    "ParametersValues",
    "Factor",
    "Result",
    "RiskResult",
    "ExpressionFunction",
    "Measure",
    "CvxMeasure",
    "Names",
    "Tags",
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

# Population
Names = str | list[str]
Tags = str | list[str]
