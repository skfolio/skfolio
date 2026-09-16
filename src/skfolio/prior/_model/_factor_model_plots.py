"""Shared plotting helpers of the factor model diagnostics."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from skfolio.typing import (
    FloatArray,
    StrArray,
)
from skfolio.utils.figure import format_plot_label


def _multi_line_plot(df: pd.DataFrame, title: str, yaxis_title: str) -> go.Figure:
    """Create a multi-line time series plot with a toggleable legend."""
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col].values,
                mode="lines",
                name=format_plot_label(col),
                line=dict(color=colors[i % len(colors)], width=1.5),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title=yaxis_title,
    )
    return fig


def _rolling_title(
    metric: str,
    window: int | None,
    *,
    context: str | None = None,
) -> str:
    """Format a default title for raw or rolling time-series plots."""
    if window is None:
        return f"{metric} ({context})" if context is not None else metric
    suffix = (
        f"{window} observations"
        if context is None
        else f"{context}, {window} observations"
    )
    return f"Rolling {metric} ({suffix})"


def _plot_single_ts(
    series: pd.Series,
    title: str,
    yaxis_title: str,
    *,
    window: int | None = None,
    show_raw: bool = False,
    show_mean: bool = True,
    ref_value: float | None = None,
    ref_label: str | None = None,
    mean_fmt: str = ".2f",
    tick_format: str | None = None,
    raw_trace_name: str | None = None,
) -> go.Figure:
    """Single time-series plot with optional rolling mean and reference lines.

    Parameters
    ----------
    series : pd.Series
        Raw time series (DatetimeIndex).

    title : str
        Figure title.

    yaxis_title : str
        Y-axis label.

    window : int, optional
        If given, smooth with a rolling mean once a complete window is available.

    show_raw : bool, default=False
        When `True` and `window` is set, also plot the raw series as
        a faded line behind the smoothed one. The raw trace is added
        first (`fig.data[0]`) so callers and tests can rely on its
        position.

    show_mean : bool, default=True
        Whether to overlay a horizontal line at the full-sample mean
        with a side annotation.

    ref_value : float, optional
        Y-value for a dashed reference line (e.g. Gaussian expectation).

    ref_label : str, optional
        Annotation text for the reference line.

    mean_fmt : str, default=".2f"
        Format string for the mean annotation value.

    tick_format : str, optional
        Y-axis tick format (e.g. `".2%"`).

    raw_trace_name : str, optional
        Trace name used for the raw series when `show_raw` is `True`.
    """
    raw = series.values.copy()
    smoothed = series.rolling(window=window).mean() if window is not None else series

    mean_val = np.nanmean(raw)

    fig = go.Figure()
    if show_raw and window is not None:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=raw,
                mode="lines",
                name=raw_trace_name or series.name,
                line=dict(color="rgba(31, 119, 180, 0.35)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                name="Rolling Mean",
                line=dict(color="rgb(31, 119, 180)", width=2),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                name=series.name,
                line=dict(color="rgb(31, 119, 180)", width=1.5),
            )
        )
    if ref_value is not None:
        fig.add_hline(
            y=ref_value,
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )
        if ref_label is not None:
            ref_yanchor = (
                "middle"
                if not show_mean
                else ("bottom" if ref_value >= mean_val else "top")
            )
            fig.add_annotation(
                xref="paper",
                yref="y",
                x=1.0,
                y=ref_value,
                text=ref_label,
                showarrow=False,
                xanchor="left",
                yanchor=ref_yanchor,
                xshift=8,
            )
    if show_mean:
        fig.add_hline(
            y=mean_val,
            line_width=1,
            line_dash="dot",
            line_color="rgb(255, 127, 14)",
        )
        mean_yanchor = (
            "middle"
            if ref_value is None
            else ("bottom" if mean_val >= ref_value else "top")
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.0,
            y=mean_val,
            text=f"Mean: {mean_val:{mean_fmt}}",
            showarrow=False,
            xanchor="left",
            yanchor=mean_yanchor,
            xshift=8,
        )
    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title=yaxis_title,
        margin=dict(r=120),
    )
    if tick_format is not None:
        fig.update_yaxes(tickformat=tick_format)
    return fig


def _heatmap(
    matrix: FloatArray,
    labels: list[str],
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Create an annotated correlation-style heatmap."""
    text = [
        [f"{matrix[i, j]:.2f}" for j in range(len(labels))] for i in range(len(labels))
    ]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _add_family_outlines(
    fig: go.Figure,
    families: StrArray | None,
    idx: slice | list[int],
) -> None:
    """Draw rectangles around contiguous family blocks on a heatmap."""
    if families is None:
        return
    if idx == slice(None):
        fam_labels = [str(family) for family in families]
    else:
        fam_labels = [str(families[i]) for i in idx]
    unique_families = dict.fromkeys(fam_labels)
    if len(unique_families) <= 1:
        return
    blocks = []
    start = 0
    for end in range(1, len(fam_labels) + 1):
        if end == len(fam_labels) or fam_labels[end] != fam_labels[start]:
            blocks.append((start, end))
            start = end
    if len(blocks) > len(unique_families):
        return
    for start, end in blocks:
        fig.add_shape(
            type="rect",
            x0=start - 0.5,
            y0=start - 0.5,
            x1=end - 0.5,
            y1=end - 0.5,
            line=dict(color="gold", width=2),
        )
