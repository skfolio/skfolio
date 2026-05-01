"""Custom typing module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

if TYPE_CHECKING:
    from skfolio.measures import (
        ExtraRiskMeasure,
        PerfMeasure,
        RatioMeasure,
        RiskMeasure,
    )
    from skfolio.optimization._base import BaseOptimization

__all__ = [
    "AnyArray",
    "ArrayLike",
    "BoolArray",
    "CvxMeasure",
    "ExpressionFunction",
    "Factor",
    "Fallback",
    "FloatArray",
    "Groups",
    "Inequality",
    "IntArray",
    "LinearConstraints",
    "Measure",
    "MultiInput",
    "Names",
    "ParametersValues",
    "Result",
    "RiskResult",
    "Scoring",
    "StrArray",
    "StrArray",
    "Tags",
    "Target",
]

# Numpy
ArrayLike: TypeAlias = npt.ArrayLike
BoolArray: TypeAlias = npt.NDArray[np.bool_]
FloatArray: TypeAlias = npt.NDArray[np.floating]
IntArray: TypeAlias = npt.NDArray[np.integer]
ObjArray: TypeAlias = npt.NDArray[np.object_]
StrArray: TypeAlias = npt.NDArray[np.str_]
AnyArray: TypeAlias = npt.NDArray[Any]

# Skfolio
Measure: TypeAlias = Union[
    "PerfMeasure", "RiskMeasure", "ExtraRiskMeasure", "RatioMeasure"
]
CvxMeasure: TypeAlias = Union["PerfMeasure", "RiskMeasure", "RatioMeasure"]
Scoring: TypeAlias = Callable | dict[str, Callable] | Measure | None
MultiInput = float | dict[str, float] | ArrayLike
Groups = dict[str, list[str]] | IntArray | StrArray | list[list[str]]
LinearConstraints = FloatArray | list[str]
Inequality = FloatArray | list
Target = float | FloatArray
ParametersValues = list[tuple[cp.Parameter, float | FloatArray]]
Factor = cp.Variable | cp.Constant
Result = FloatArray | tuple[float | tuple[float, float] | FloatArray, FloatArray]
RiskResult = tuple[cp.Expression | cp.Variable, list[cp.Expression | cp.SOC | cp.PSD]]
ExpressionFunction = Callable[[cp.Variable, Any], cp.Expression]
Figure = go.Figure
Names = str | list[str]
Tags = str | list[str]
Fallback: TypeAlias = Union[
    "BaseOptimization",
    list[Union["BaseOptimization", Literal["previous_weights"]]],
    Literal["previous_weights"],
    None,
]
