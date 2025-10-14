"""Figure module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st


def plot_kde_distributions(
    X: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    percentile_cutoff: float | None = None,
    title: str = "Distribution of Asset Returns",
    unweighted_suffix: str = "",
    weighted_suffix: str = "with Sample Weight",
) -> go.Figure:
    """
    Plot the Kernel Density Estimate (KDE) of return distributions for multiple assets.

    Parameters
    ----------
    X : DataFrame of shape (n_observations, n_assets)
        Return data where each column corresponds to an asset and each row to an
        observation.

    sample_weight : ndarray of shape (n_observations,), optional
        Weights to apply to each observation when computing the KDE.
        If None, equal weighting is used.

    percentile_cutoff : float, default=None
        Percentile cutoff for tail truncation (percentile), in percent.
        - If a float p is provided, the distribution support is truncated at
          the p-th and (100 - p)-th percentiles.
        - If None, no truncation is applied (uses full min/max of returns).

    title : str, default="Distribution of Asset Returns"
        Title for the Plotly figure.

    unweighted_suffix : str, default=""
        Suffix to append to asset names for unweighted KDE traces.

    weighted_suffix : str, default="weighted"
        Suffix to append to asset names for weighted KDE traces.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly Figure object containing overlaid KDE plots for each asset,
        with separate traces for weighted and unweighted distributions if weights
        are provided.
    """
    asset_names = X.columns.tolist()
    X = X.values
    colors = px.colors.qualitative.Plotly

    traces: list[go.Scatter] = []

    for i, asset in enumerate(asset_names):
        x = X[:, i]
        color = colors[i % len(colors)]
        visible = True if i == 0 else "legendonly"

        # Unweighted: solid line
        traces.append(
            kde_trace(
                x=x,
                sample_weight=None,
                percentile_cutoff=percentile_cutoff,
                name=f"{asset} {unweighted_suffix}".strip(),
                line_color=color,
                fill_opacity=0.17,
                line_dash="solid",
                line_width=1,
                visible=visible,
            )
        )

        # Weighted: dashed, thicker line
        if sample_weight is not None:
            traces.append(
                kde_trace(
                    x=x,
                    sample_weight=sample_weight,
                    percentile_cutoff=percentile_cutoff,
                    name=f"{asset} {weighted_suffix}".strip(),
                    line_color=color,
                    fill_opacity=0.17,
                    line_dash="dash",
                    line_width=1.5,
                    visible=visible,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title="Returns",
        yaxis_title="Probability Density",
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def kde_trace(
    x: np.ndarray,
    sample_weight: np.ndarray | None,
    percentile_cutoff: float | None,
    name: str,
    line_color: str,
    fill_opacity: float,
    line_dash: str,
    line_width: float,
    visible: bool,
) -> go.Scatter:
    """
    Create a Plotly Scatter trace representing a Gaussian kernel density estimate (KDE),
    with customizable line style and fill opacity.

    Parameters
    ----------
    x : ndarray of shape (n_observations,)
        One-dimensional array of sample values for which the KDE is computed.

    sample_weight : ndarray of shape (n_observations,), optional
        Weights to apply to each observation when computing the KDE.
        If None, equal weighting is used.

    percentile_cutoff : float, default=None
        Percentile cutoff for tail truncation (percentile), in percent.
        - If a float p is provided, the distribution support is truncated at
          the p-th and (100 - p)-th percentiles.
        - If None, no truncation is applied (uses full min/max of returns).

    name : str
        Legend name for this trace.

    line_color : str
        Color of the KDE line (hex or named CSS color).

    fill_opacity : float
        Opacity of the filled area under the curve (0 to 1).

    line_dash : str
        Dash style for the line ("solid", "dash", "dot", etc.).

    line_width : float
        Width of the line.

    visible : bool
        Initial visibility of the trace in the Plotly figure.

    Returns
    -------
    go.Scatter
        A Plotly Scatter trace with the KDE line and shaded area under the curve.
    """
    if percentile_cutoff is None:
        lower, upper = x.min(), x.max()
    else:
        lower = np.percentile(x, percentile_cutoff)
        upper = np.percentile(x, 100.0 - percentile_cutoff)

    xs = np.linspace(lower, upper, 500)
    ys = st.gaussian_kde(x, weights=sample_weight)(xs)

    # build RGBA fill color from the line_color hex
    r, g, b = tuple(int(line_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    fill_color = f"rgba({r},{g},{b},{fill_opacity})"

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=name,
        visible=visible,
        line=dict(color=line_color, dash=line_dash, width=line_width),
        fill="tozeroy",
        fillcolor=fill_color,
        opacity=1.0,
    )
